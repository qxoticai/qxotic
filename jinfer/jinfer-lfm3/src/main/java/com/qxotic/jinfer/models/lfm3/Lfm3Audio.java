package com.qxotic.jinfer.models.lfm3;

import static com.qxotic.jinfer.models.lfm3.Lfm3.f32;
import static com.qxotic.jinfer.models.lfm3.Lfm3.require;
import static com.qxotic.jinfer.models.lfm3.Lfm3.weight;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.GeluErf;
import com.qxotic.jinfer.kernels.LogMel;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.ObjIntConsumer;

/**
 * The LFM3-Omni audio encoder: a Whisper-style log10 mel (128 bands over 25 ms windows every 10
 * ms), an 8x downsampler of three stride-2 3x3 convolutions (the last two depthwise then pointwise,
 * each followed by SiLU), then a causal LFM block stack - short convolutions, and GQA over a
 * sliding window of the last {@code window} positions - and a LayerNorm GELU MLP into the language
 * model's width. The whole clip encodes at once, as llama.cpp's mtmd does.
 */
final class Lfm3Audio implements MediaProjector<Media.Audio> {

    static final int SAMPLE_RATE = 16_000;
    private static final int N_FFT = 400, HOP = 160;
    private static final int SHORTCONV_PARTS = 3; // the short conv in_proj: B | C | x
    private static final int SHORTCONV_TAPS = 3;
    private static final float LN_10 = (float) Math.log(10);

    private final int nMel, channels, dim, heads, kvHeads, headDim, ffnDim, window;
    private final int projectorHidden, modelDim;
    private final float eps;
    private final LogMel logMel;
    private final float[] conv0Taps; // [3 x 3][channels], F16 values: ggml's im2col is F16
    private final float[] conv0Bias, depthwise1, depthwise2, bias1, bias2, bias3, bias4;
    private final MemoryView<MemorySegment> pointwise1, pointwise2, downOut, downOutBias;
    private final MemoryView<MemorySegment> postNorm, projectorNorm, projectorNormBias;
    private final MemoryView<MemorySegment> projectorUp,
            projectorUpBias,
            projectorDown,
            projectorDownBias;
    private final RoPE.Schedule rope;
    private final Layer[] layers;

    /** Exactly one of the short-conv and attention weight groups is set. */
    record Layer(
            MemoryView<MemorySegment> norm1,
            MemoryView<MemorySegment> convIn,
            MemoryView<MemorySegment> convTaps,
            MemoryView<MemorySegment> convOut,
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> queryNorm,
            MemoryView<MemorySegment> keyNorm,
            MemoryView<MemorySegment> norm2,
            MemoryView<MemorySegment> ffnGate,
            MemoryView<MemorySegment> ffnUp,
            MemoryView<MemorySegment> ffnDown) {}

    Lfm3Audio(GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors, int modelDim) {
        require(
                "lfm3a".equals(gguf.getString("clip.audio.projector_type")),
                "unsupported audio projector " + gguf.getString("clip.audio.projector_type"));
        nMel = gguf.getValue(int.class, "clip.audio.num_mel_bins");
        dim = gguf.getValue(int.class, "clip.audio.embedding_length");
        heads = gguf.getValue(int.class, "clip.audio.attention.head_count");
        kvHeads = gguf.getValue(int.class, "clip.audio.attention.head_count_kv");
        ffnDim = gguf.getValue(int.class, "clip.audio.feed_forward_length");
        eps = gguf.getValue(float.class, "clip.audio.attention.layer_norm_epsilon");
        window = gguf.getValue(int.class, "clip.audio.window_size");
        this.modelDim = modelDim;
        headDim = dim / heads;
        require(dim % heads == 0 && heads % kvHeads == 0 && window > 0, "audio encoder geometry");
        require(
                gguf.getValue(int.class, "clip.audio.projection_dim") == modelDim,
                "audio projection width != model width " + modelDim);
        logMel =
                new LogMel(
                        new LogMel.Spec(
                                N_FFT, HOP, nMel, hann(), filterbank(nMel), 0f, 2f, 1e-12f, 0f));

        channels =
                Math.toIntExact(
                        ModelLoader.require(tensors, "a.conv1d.0.weight").shape().flatAt(0));
        conv0Taps =
                Towers.roundToHalf(
                        Towers.tapMajor(
                                floats(tensors, "a.conv1d.0.weight", channels, 1, 3, 3), channels));
        depthwise1 =
                Towers.tapMajor(floats(tensors, "a.conv1d.2.weight", channels, 1, 3, 3), channels);
        depthwise2 =
                Towers.tapMajor(floats(tensors, "a.conv1d.5.weight", channels, 1, 3, 3), channels);
        conv0Bias = floats(tensors, "a.conv1d.0.bias", channels, 1, 1);
        bias1 = floats(tensors, "a.conv1d.2.bias", channels, 1, 1);
        bias2 = floats(tensors, "a.conv1d.3.bias", channels, 1, 1);
        bias3 = floats(tensors, "a.conv1d.5.bias", channels, 1, 1);
        bias4 = floats(tensors, "a.conv1d.6.bias", channels, 1, 1);
        Shape square = Shape.flat(channels, channels);
        pointwise1 = f32(tensors, "a.conv1d.3.weight", channels, channels, 1, 1).view(square);
        pointwise2 = f32(tensors, "a.conv1d.6.weight", channels, channels, 1, 1).view(square);
        int flat = channels * reduce(reduce(reduce(nMel)));
        downOut = weight(tensors, "a.pre_encode.out.weight", dim, flat);
        downOutBias = f32(tensors, "a.pre_encode.out.bias", dim);

        int count = gguf.getValue(int.class, "clip.audio.block_count");
        layers = new Layer[count];
        int kvDim = kvHeads * headDim;
        for (int l = 0; l < count; l++) {
            String p = "a.blk." + l + ".";
            boolean conv = tensors.containsKey(p + "conv_dw.weight");
            layers[l] =
                    new Layer(
                            f32(tensors, p + "ln1.weight", dim),
                            conv ? weight(tensors, p + "conv_pw1.weight", 3 * dim, dim) : null,
                            conv ? f32(tensors, p + "conv_dw.weight", dim, SHORTCONV_TAPS) : null,
                            conv ? weight(tensors, p + "conv_pw2.weight", dim, dim) : null,
                            conv ? null : weight(tensors, p + "attn_q.weight", dim, dim),
                            conv ? null : weight(tensors, p + "attn_k.weight", kvDim, dim),
                            conv ? null : weight(tensors, p + "attn_v.weight", kvDim, dim),
                            conv ? null : weight(tensors, p + "attn_out.weight", dim, dim),
                            conv ? null : f32(tensors, p + "attn_q_norm.weight", headDim),
                            conv ? null : f32(tensors, p + "attn_k_norm.weight", headDim),
                            f32(tensors, p + "ln2.weight", dim),
                            weight(tensors, p + "ffn_gate.weight", ffnDim, dim),
                            weight(tensors, p + "ffn_up.weight", ffnDim, dim),
                            weight(tensors, p + "ffn_down.weight", dim, ffnDim));
        }
        postNorm = f32(tensors, "a.post_ln.weight", dim);
        projectorHidden =
                Math.toIntExact(ModelLoader.require(tensors, "mm.a.mlp.1.bias").shape().flatAt(0));
        projectorNorm = f32(tensors, "mm.a.mlp.0.weight", dim);
        projectorNormBias = f32(tensors, "mm.a.mlp.0.bias", dim);
        projectorUp = weight(tensors, "mm.a.mlp.1.weight", projectorHidden, dim);
        projectorUpBias = f32(tensors, "mm.a.mlp.1.bias", projectorHidden);
        projectorDown = weight(tensors, "mm.a.mlp.3.weight", modelDim, projectorHidden);
        projectorDownBias = f32(tensors, "mm.a.mlp.3.bias", modelDim);
        rope = RoPE.plain(headDim, 1e6); // the converter pins the encoder's rope_theta
    }

    /** A stride-2, pad-1, 3-tap axis: {@code ceil(n / 2)}. */
    private static int reduce(int n) {
        return (n + 1) / 2;
    }

    /** Mel frames of {@code samples}: the tail zero-padded so no sample is left out. */
    private static int frames(int samples) {
        if (samples <= N_FFT) return 1;
        return (samples - N_FFT + HOP - 1) / HOP + 1;
    }

    @Override
    public int positions(Media.Audio audio) {
        requireRate(audio);
        return reduce(reduce(reduce(frames(audio.pcm().length / audio.channels()))));
    }

    @Override
    public String planId() {
        return "lfm3a mel=" + nMel + " window=" + window;
    }

    @Override
    public void project(Media.Audio audio, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        if (maxChunkSize <= 0) throw new IllegalArgumentException("maxChunkSize must be positive");
        int rows = positions(audio);
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            MemoryView<MemorySegment> out = encode(audio, scratch, null);
            for (int first = 0; first < rows; first += maxChunkSize)
                sink.accept(out.slice(0, first, Math.min(rows, first + maxChunkSize)));
        } finally {
            Arenas.close(scratch);
            Reference.reachabilityFence(this);
        }
    }

    /** The codec resamples, so any other rate is refused, never guessed at. */
    private static void requireRate(Media.Audio audio) {
        if (audio.sampleRate() != SAMPLE_RATE)
            throw new IllegalArgumentException(
                    "LFM3 audio is 16 kHz, got "
                            + audio.sampleRate()
                            + " Hz - decode it with the audio codec, which resamples");
    }

    /** Mono 16 kHz PCM, the channels averaged. */
    static float[] mono(Media.Audio audio) {
        requireRate(audio);
        int c = audio.channels(), n = audio.pcm().length / c;
        if (c == 1) return audio.pcm();
        float[] mono = new float[n];
        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int k = 0; k < c; k++) sum += audio.pcm()[i * c + k];
            mono[i] = sum / c;
        }
        return mono;
    }

    /** {@code (log10(mel) + 4) / 4}, frame-major {@code [frames][nMel]}. */
    float[] mel(float[] pcm) {
        float[] mel = logMel.frames(pcm, 0, pcm.length, 0, frames(pcm.length));
        for (int i = 0; i < mel.length; i++) mel[i] = (mel[i] / LN_10 + 4f) / 4f;
        return mel;
    }

    /**
     * The rows of {@code audio}; {@code beforeLayer} (nullable) sees each block's normed input, and
     * finally the post-stack norm.
     */
    MemoryView<MemorySegment> encode(
            Media.Audio audio,
            MemoryArena<MemorySegment> arena,
            ObjIntConsumer<MemoryView<MemorySegment>> beforeLayer) {
        // the downsampler, channel-last [time][freq][channel]; the mel enters F16, as im2col has it
        float[] mel = Towers.roundToHalf(mel(mono(audio)));
        int frames = mel.length / nMel;
        int t1 = reduce(frames), f1 = reduce(nMel), t2 = reduce(t1), f2 = reduce(f1);
        int t3 = reduce(t2), f3 = reduce(f2), n = t3;

        float[] a = conv3x3(mel, frames, nMel, 1, conv0Taps, conv0Bias, true);
        float[] b = conv3x3(a, t1, f1, channels, depthwise1, bias1, false);
        float[] c = pointwise(b, t2 * f2, pointwise1, bias2, arena);
        float[] d = conv3x3(c, t2, f2, channels, depthwise2, bias3, false);
        float[] e = pointwise(d, t3 * f3, pointwise2, bias4, arena);
        // ggml flattens each time step frequency-fastest: [channel][freq]
        float[] flat = new float[n * channels * f3];
        for (int t = 0; t < n; t++)
            for (int f = 0; f < f3; f++)
                for (int ch = 0; ch < channels; ch++)
                    flat[(t * channels + ch) * f3 + f] = e[(t * f3 + f) * channels + ch];

        Scratch s = new Scratch(n, channels * f3, arena);
        Views.copyFromArray(s.flat, 0, flat, 0, flat.length, "downsampled");
        MatMul.gemm(downOut, s.flat, s.x, n);
        Ops.addRowBiasInPlace(s.x, 0, downOutBias, 0, n, dim);
        RoPE.fill(s.cos, s.sin, 0, n, headDim / 2, rope);

        for (int l = 0; l < layers.length; l++) {
            Layer layer = layers[l];
            Norms.rmsnormRowsGgml(s.normed, s.x, layer.norm1, n, dim, eps);
            if (beforeLayer != null) beforeLayer.accept(s.normed, l);
            if (layer.convIn != null) shortConv(layer, s);
            else attention(layer, s);
            Ops.addRows(s.x, s.mixed, n, dim);
            Norms.rmsnormRowsGgml(s.normed, s.x, layer.norm2, n, dim, eps);
            Towers.ffn(
                    layer.ffnGate,
                    layer.ffnUp,
                    layer.ffnDown,
                    s.normed,
                    s.hidden,
                    s.hidden2,
                    s.mixed,
                    n,
                    ffnDim);
            Ops.addRows(s.x, s.mixed, n, dim);
        }
        Norms.rmsnormRowsGgml(s.x, s.x, postNorm, n, dim, eps);
        if (beforeLayer != null) beforeLayer.accept(s.x, layers.length);

        Norms.layerNormRows(s.x, s.x, projectorNorm, projectorNormBias, n, dim, 1e-5f);
        MatMul.gemm(projectorUp, s.x, s.projected, n);
        Ops.addRowBiasInPlace(s.projected, 0, projectorUpBias, 0, n, projectorHidden);
        GeluErf.rows(s.projected, n, projectorHidden);
        MatMul.gemm(projectorDown, s.projected, s.output, n);
        Ops.addRowBiasInPlace(s.output, 0, projectorDownBias, 0, n, modelDim);
        return s.output;
    }

    /** The encoder's buffers, allocated before its first block runs. */
    private final class Scratch {
        final MemoryView<MemorySegment> flat, x, normed, mixed, wide, query, key, value;
        final MemoryView<MemorySegment> half, convState, cos, sin;
        final MemoryView<MemorySegment> hidden, hidden2, projected, output;
        final int rows;

        Scratch(int rows, int flatWidth, MemoryArena<MemorySegment> arena) {
            this.rows = rows;
            int kvDim = kvHeads * headDim;
            flat = Views.allocateF32(arena, rows, flatWidth);
            x = Views.allocateF32(arena, rows, dim);
            normed = Views.allocateF32(arena, rows, dim);
            mixed = Views.allocateF32(arena, rows, dim);
            wide = Views.allocateF32(arena, rows, SHORTCONV_PARTS * dim);
            query = Views.allocateF32(arena, rows, dim);
            key = Views.allocateF32(arena, rows, kvDim);
            value = Views.allocateF32(arena, rows, kvDim);
            half = Views.allocateF16(arena, rows, kvDim);
            // a clip starts from silence: zeros
            convState = Views.allocateF32(arena, (long) (SHORTCONV_TAPS - 1) * dim);
            cos = Views.allocateF32(arena, rows, headDim / 2);
            sin = Views.allocateF32(arena, rows, headDim / 2);
            hidden = Views.allocateF32(arena, rows, ffnDim);
            hidden2 = Views.allocateF32(arena, rows, ffnDim);
            projected = Views.allocateF32(arena, rows, projectorHidden);
            output = Views.allocateF32(arena, rows, modelDim);
        }
    }

    private void shortConv(Layer layer, Scratch s) {
        MatMul.gemm(layer.convIn, s.normed, s.wide, s.rows);
        Ops.fillInPlace(s.convState, 0, (SHORTCONV_TAPS - 1) * dim, 0f);
        Convolutions.shortConvScan(
                layer.convTaps,
                s.convState,
                s.wide,
                s.normed,
                s.rows,
                dim,
                SHORTCONV_TAPS,
                SHORTCONV_PARTS);
        MatMul.gemm(layer.convOut, s.normed, s.mixed, s.rows);
    }

    private void attention(Layer layer, Scratch s) {
        int kvDim = kvHeads * headDim;
        MatMul.gemm(layer.query, s.normed, s.query, s.rows);
        MatMul.gemm(layer.key, s.normed, s.key, s.rows);
        MatMul.gemm(layer.value, s.normed, s.value, s.rows);
        Parallel.forLoop(
                s.rows,
                row -> {
                    Towers.headNormRope(
                            s.query,
                            (long) row * dim,
                            heads,
                            headDim,
                            layer.queryNorm,
                            eps,
                            row,
                            s.cos,
                            s.sin);
                    Towers.headNormRope(
                            s.key,
                            (long) row * kvDim,
                            kvHeads,
                            headDim,
                            layer.keyNorm,
                            eps,
                            row,
                            s.cos,
                            s.sin);
                });
        // ggml's flash attention reads K and V in F16: round them so, keep them F32 for speed;
        // no cache, the clip is its own context
        Towers.roundToHalf(s.key, s.half, s.rows * kvDim);
        Towers.roundToHalf(s.value, s.half, s.rows * kvDim);
        FlashAttention.slidingWindowPrefill(
                s.query,
                s.normed,
                s.key,
                s.value,
                s.key,
                s.value,
                heads,
                0,
                s.rows,
                headDim,
                kvDim,
                dim,
                kvDim,
                heads / kvHeads,
                1f / (float) Math.sqrt(headDim),
                window,
                0,
                null);
        MatMul.gemm(layer.output, s.normed, s.mixed, s.rows);
    }

    /**
     * A stride-2 3x3 convolution over {@code [time][freq][in]} into {@code [time/2][freq/2][out]}:
     * full ({@code in = 1}, {@code taps [9][out]}) or depthwise ({@code taps [9][channels]}). Time
     * is padded with one zero frame on each side (llama.cpp's left state of a fresh clip, and its
     * right pad), frequency by one zero bin on each side.
     */
    private float[] conv3x3(
            float[] in,
            int time,
            int freq,
            int inChannels,
            float[] taps,
            float[] bias,
            boolean full) {
        int timeOut = reduce(time), freqOut = reduce(freq);
        float[] out = new float[timeOut * freqOut * channels];
        Parallel.forLoop(
                timeOut,
                t -> {
                    for (int f = 0; f < freqOut; f++) {
                        int o = (t * freqOut + f) * channels;
                        for (int kt = 0; kt < 3; kt++) {
                            int st = 2 * t + kt - 1;
                            if (st < 0 || st >= time) continue;
                            for (int kf = 0; kf < 3; kf++) {
                                int sf = 2 * f + kf - 1;
                                if (sf < 0 || sf >= freq) continue;
                                int tap = (kf * 3 + kt) * channels,
                                        src = (st * freq + sf) * inChannels;
                                if (full) {
                                    float v = in[src];
                                    for (int ch = 0; ch < channels; ch++)
                                        out[o + ch] += taps[tap + ch] * v;
                                } else {
                                    for (int ch = 0; ch < channels; ch++)
                                        out[o + ch] += taps[tap + ch] * in[src + ch];
                                }
                            }
                        }
                        for (int ch = 0; ch < channels; ch++) out[o + ch] += bias[ch];
                        // the full conv is followed by SiLU, a depthwise one by its pointwise conv
                        if (full)
                            for (int ch = 0; ch < channels; ch++)
                                out[o + ch] = Activations.silu(out[o + ch]);
                    }
                });
        return out;
    }

    /** A 1x1 convolution: a GEMM over the positions, then the bias and SiLU. */
    private float[] pointwise(
            float[] in,
            int positions,
            MemoryView<MemorySegment> w,
            float[] bias,
            MemoryArena<MemorySegment> arena) {
        MemoryView<MemorySegment> x = Views.allocateF32(arena, positions, channels);
        MemoryView<MemorySegment> y = Views.allocateF32(arena, positions, channels);
        Views.copyFromArray(x, 0, in, 0, in.length, "pointwise in");
        MatMul.gemm(w, x, y, positions);
        float[] out = new float[in.length];
        Views.copyToArray(y, 0, out, 0, out.length, "pointwise out");
        Parallel.forLoop(
                positions,
                p -> {
                    for (int ch = 0, o = p * channels; ch < channels; ch++)
                        out[o + ch] = Activations.silu(out[o + ch] + bias[ch]);
                });
        return out;
    }

    // --- loading ---

    /** Periodic Hann, as llama.cpp tabulates it: {@code 0.5 (1 - cosf(2 pi i / n))}. */
    private static float[] hann() {
        float[] w = new float[N_FFT];
        for (int i = 0; i < N_FFT; i++)
            w[i] = (float) (0.5 * (1.0 - (float) Math.cos((float) (2.0 * Math.PI * i / N_FFT))));
        return w;
    }

    /**
     * librosa's default, as llama.cpp computes it: the Slaney mel scale from 0 Hz to Nyquist,
     * Slaney area-normalized, every expression in its order.
     */
    private static float[] filterbank(int nMel) {
        int bins = N_FFT / 2 + 1;
        double binHz = (double) SAMPLE_RATE / N_FFT, hiMel = hzToMel(0.5f * SAMPLE_RATE);
        double[] hz = new double[nMel + 2];
        for (int i = 0; i < nMel + 2; i++) hz[i] = melToHz(hiMel * ((double) i / (nMel + 1)));
        float[] out = new float[nMel * bins];
        for (int m = 0; m < nMel; m++) {
            double left = hz[m], center = hz[m + 1], right = hz[m + 2];
            double norm = 2.0 / Math.max(1e-30, right - left);
            for (int k = 0; k < bins; k++) {
                double f = k * binHz, w = 0;
                if (f >= left && f <= center) w = (f - left) / Math.max(1e-30, center - left);
                else if (f > center && f <= right)
                    w = (right - f) / Math.max(1e-30, right - center);
                out[m * bins + k] = (float) (w * norm);
            }
        }
        return out;
    }

    private static final double LIN_SLOPE = 3 / 200., MIN_LOG_MEL = 1000 * LIN_SLOPE;
    private static final double LOG_STEP = Math.log(6.4) / 27.0;

    private static double hzToMel(double hz) {
        return hz < 1000 ? hz * LIN_SLOPE : MIN_LOG_MEL + Math.log(hz / 1000) / LOG_STEP;
    }

    private static double melToHz(double mel) {
        return mel < MIN_LOG_MEL
                ? mel / LIN_SLOPE
                : 1000 * Math.exp((mel - MIN_LOG_MEL) * LOG_STEP);
    }

    private static float[] floats(
            Map<String, MemoryView<MemorySegment>> tensors, String name, long... shape) {
        return Views.toFloatArray(f32(tensors, name, shape), name);
    }
}
