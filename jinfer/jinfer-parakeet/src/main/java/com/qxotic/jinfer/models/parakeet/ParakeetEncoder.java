package com.qxotic.jinfer.models.parakeet;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.Workspace;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.LogMel;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.function.ObjIntConsumer;

/**
 * NVIDIA Parakeet FastConformer encoder (NeMo {@code ConformerEncoder}, {@code
 * general.architecture=parakeet}): NeMo mel front end, 8x depthwise-separable conv subsampling,
 * then macaron Conformer blocks with Transformer-XL relative-position attention. Ported against
 * parakeet.cpp; layer semantics are documented in the port plan and verified against its fixtures.
 */
public final class ParakeetEncoder {
    private static final float NORM_EPS = 1e-5f;
    private static final float BATCH_NORM_EPS = 1e-5f;

    record Config(
            int dModel,
            int layers,
            int heads,
            int ffDim,
            int convKernel,
            int subsamplingChannels,
            int nMels,
            int nFft,
            int hop,
            float preemphasis,
            float magPower,
            float logZeroGuard,
            boolean xscaling) {
        static Config from(GGUF gguf) {
            return new Config(
                    gguf.getValue(int.class, "parakeet.encoder.d_model"),
                    gguf.getValue(int.class, "parakeet.encoder.n_layers"),
                    gguf.getValue(int.class, "parakeet.encoder.n_heads"),
                    gguf.getValue(int.class, "parakeet.encoder.ff_dim"),
                    gguf.getValue(int.class, "parakeet.encoder.conv_kernel"),
                    gguf.getValue(int.class, "parakeet.encoder.subsampling_conv_channels"),
                    gguf.getValue(int.class, "parakeet.preprocessor.n_mels"),
                    gguf.getValue(int.class, "parakeet.preprocessor.n_fft"),
                    gguf.getValue(int.class, "parakeet.preprocessor.hop_length"),
                    gguf.getValue(float.class, "parakeet.preprocessor.preemph"),
                    gguf.getValue(float.class, "parakeet.preprocessor.mag_power"),
                    gguf.getValue(float.class, "parakeet.preprocessor.log_zero_guard"),
                    Boolean.TRUE.equals(gguf.getValue(boolean.class, "parakeet.encoder.xscaling")));
        }
    }

    /** A projection; {@code bias} is null on the bias-free checkpoints (tdt-0.6b-v2/v3). */
    record Linear(
            MemoryView<MemorySegment> weight, MemoryView<MemorySegment> bias, int in, int out) {
        /** {@code output[rows][out] = input[rows][in] · weightᵀ + bias}. */
        void apply(MemoryView<MemorySegment> input, MemoryView<MemorySegment> output, int rows) {
            MatMul.gemm(weight, input, in, output, out, out, rows, in);
            if (bias != null) Ops.addRowBiasInPlace(output, 0, bias, 0, rows, out);
        }
    }

    /** LayerNorm over rows of {@code weight.size()} channels. */
    record Norm(MemoryView<MemorySegment> weight, MemoryView<MemorySegment> bias) {
        void apply(MemoryView<MemorySegment> output, MemoryView<MemorySegment> input, int rows) {
            int channels = Math.toIntExact(weight.shape().size());
            Norms.layerNormRows(output, input, weight, bias, rows, channels, NORM_EPS);
        }
    }

    record Block(
            Norm ff1Norm,
            Linear ff1Up,
            Linear ff1Down,
            Norm attnNorm,
            Linear query,
            Linear key,
            Linear value,
            Linear position,
            Linear output,
            MemoryView<MemorySegment> posBiasU,
            MemoryView<MemorySegment> posBiasV,
            Norm convNorm,
            Linear pointwise1,
            float[] depthwise,
            float[] bnScale,
            float[] bnShift,
            Linear pointwise2,
            Norm ff2Norm,
            Linear ff2Up,
            Linear ff2Down,
            Norm outNorm) {}

    /** Encoder output, frame-major {@code [frames, dModel]}. */
    public record Output(float[] data, int frames) {}

    private final Config config;
    private final LogMel logMel;
    private final float[] conv0Taps, conv0Bias;
    private final float[] dw1Taps, dw1Bias, dw2Taps, dw2Bias;
    private final Linear pw1, pw2, preOut;
    private final Block[] blocks;

    private ParakeetEncoder(
            Config config,
            LogMel logMel,
            float[] conv0Taps,
            float[] conv0Bias,
            float[] dw1Taps,
            float[] dw1Bias,
            Linear pw1,
            float[] dw2Taps,
            float[] dw2Bias,
            Linear pw2,
            Linear preOut,
            Block[] blocks) {
        this.config = config;
        this.logMel = logMel;
        this.conv0Taps = conv0Taps;
        this.conv0Bias = conv0Bias;
        this.dw1Taps = dw1Taps;
        this.dw1Bias = dw1Bias;
        this.pw1 = pw1;
        this.dw2Taps = dw2Taps;
        this.dw2Bias = dw2Bias;
        this.pw2 = pw2;
        this.preOut = preOut;
        this.blocks = blocks;
    }

    Config config() {
        return config;
    }

    public static ParakeetEncoder load(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return load(gguf, ModelLoader.loadTensors(channel, gguf, arena), arena);
        }
    }

    public static ParakeetEncoder load(
            GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors, Arena arena) {
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(tensors, "tensors");
        String architecture = gguf.getValue(String.class, "general.architecture");
        if (!"parakeet".equals(architecture))
            throw new IllegalArgumentException(
                    "expected general.architecture=parakeet but was '" + architecture + "'");
        Config config = Config.from(gguf);
        if (config.magPower() != 1f && config.magPower() != 2f)
            throw new IllegalArgumentException("unsupported mag_power " + config.magPower());
        int channels = config.subsamplingChannels();
        int dim = config.dModel(), ffDim = config.ffDim();
        int flattened = channels * subsampled(subsampled(subsampled(config.nMels())));
        Block[] blocks = new Block[config.layers()];
        for (int i = 0; i < blocks.length; i++) {
            String prefix = "encoder.layers." + i + ".";
            float[] bnWeight = Tensors.floats(tensors, prefix + "conv.batch_norm.weight", dim);
            float[] bnBias = Tensors.floats(tensors, prefix + "conv.batch_norm.bias", dim);
            float[] mean = Tensors.floats(tensors, prefix + "conv.batch_norm.running_mean", dim);
            float[] variance = Tensors.floats(tensors, prefix + "conv.batch_norm.running_var", dim);
            // The depthwise conv's bias (absent on v2/v3) precedes the norm, so it folds into
            // the same per-channel affine: (x + b)*scale + shift = x*scale + (shift + b*scale).
            float[] dwBias =
                    tensors.containsKey(prefix + "conv.depthwise_conv.bias")
                            ? Tensors.floats(tensors, prefix + "conv.depthwise_conv.bias", dim)
                            : null;
            float[] bnScale = new float[dim];
            float[] bnShift = new float[dim];
            for (int c = 0; c < dim; c++) {
                bnScale[c] = (float) (bnWeight[c] / Math.sqrt(variance[c] + BATCH_NORM_EPS));
                bnShift[c] = bnBias[c] - mean[c] * bnScale[c];
                if (dwBias != null) bnShift[c] += dwBias[c] * bnScale[c];
            }
            blocks[i] =
                    new Block(
                            norm(tensors, prefix + "norm_feed_forward1"),
                            linear(tensors, prefix + "feed_forward1.linear1", dim, ffDim),
                            linear(tensors, prefix + "feed_forward1.linear2", ffDim, dim),
                            norm(tensors, prefix + "norm_self_att"),
                            linear(tensors, prefix + "self_attn.linear_q", dim, dim),
                            linear(tensors, prefix + "self_attn.linear_k", dim, dim),
                            linear(tensors, prefix + "self_attn.linear_v", dim, dim),
                            linear(tensors, prefix + "self_attn.linear_pos", dim, dim),
                            linear(tensors, prefix + "self_attn.linear_out", dim, dim),
                            Tensors.vector(tensors, prefix + "self_attn.pos_bias_u", dim),
                            Tensors.vector(tensors, prefix + "self_attn.pos_bias_v", dim),
                            norm(tensors, prefix + "norm_conv"),
                            linear(tensors, prefix + "conv.pointwise_conv1", dim, 2 * dim),
                            Tensors.floats(
                                    tensors,
                                    prefix + "conv.depthwise_conv.weight",
                                    dim * config.convKernel()),
                            bnScale,
                            bnShift,
                            linear(tensors, prefix + "conv.pointwise_conv2", dim, dim),
                            norm(tensors, prefix + "norm_feed_forward2"),
                            linear(tensors, prefix + "feed_forward2.linear1", dim, ffDim),
                            linear(tensors, prefix + "feed_forward2.linear2", ffDim, dim),
                            norm(tensors, prefix + "norm_out"));
        }
        return new ParakeetEncoder(
                config,
                featurizer(gguf, tensors, config),
                Tensors.floats(tensors, "encoder.pre_encode.conv.0.weight", channels * 9),
                Tensors.floats(tensors, "encoder.pre_encode.conv.0.bias", channels),
                Tensors.floats(tensors, "encoder.pre_encode.conv.2.weight", channels * 9),
                Tensors.floats(tensors, "encoder.pre_encode.conv.2.bias", channels),
                biasedLinear(tensors, "encoder.pre_encode.conv.3", channels, channels),
                Tensors.floats(tensors, "encoder.pre_encode.conv.5.weight", channels * 9),
                Tensors.floats(tensors, "encoder.pre_encode.conv.5.bias", channels),
                biasedLinear(tensors, "encoder.pre_encode.conv.6", channels, channels),
                biasedLinear(tensors, "encoder.pre_encode.out", flattened, dim),
                blocks);
    }

    /** A conformer projection: its bias is optional (the v2/v3 checkpoints have none). */
    private static Linear linear(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int in, int out) {
        return new Linear(
                Tensors.weight(tensors, name + ".weight", out),
                tensors.get(name + ".bias"),
                in,
                out);
    }

    /** A subsampling projection: every checkpoint has its bias. */
    private static Linear biasedLinear(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int in, int out) {
        return new Linear(
                Tensors.weight(tensors, name + ".weight", out),
                Tensors.require(tensors, name + ".bias"),
                in,
                out);
    }

    private static Norm norm(Map<String, MemoryView<MemorySegment>> tensors, String name) {
        return new Norm(
                Tensors.require(tensors, name + ".weight"),
                Tensors.require(tensors, name + ".bias"));
    }

    /** The NeMo featurizer: window and filterbank are lifted verbatim from the model weights. */
    private static LogMel featurizer(
            GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors, Config config) {
        int winLength = gguf.getValue(int.class, "parakeet.preprocessor.win_length");
        float[] window = Tensors.floats(tensors, "preprocessor.featurizer.window", winLength);
        float[] centered = new float[config.nFft()];
        System.arraycopy(window, 0, centered, (config.nFft() - winLength) / 2, winLength);
        float[] filterbank =
                Tensors.floats(
                        tensors,
                        "preprocessor.featurizer.fb",
                        config.nMels() * (config.nFft() / 2 + 1));
        return new LogMel(
                new LogMel.Spec(
                        config.nFft(),
                        config.hop(),
                        config.nMels(),
                        centered,
                        filterbank,
                        config.preemphasis(),
                        config.magPower(),
                        0f,
                        config.logZeroGuard()));
    }

    /**
     * Running per-bin mel statistics for streamed windows. Per-utterance {@code per_feature}
     * normalization is fragile on short excerpts - a 10 s cut of clean speech can normalize into
     * features the model decodes as silence - so consecutive windows of one stream share
     * statistics, weight-capped at about a minute so they still track slow gain changes.
     */
    public static final class MelStats {
        private double[] mean, variance;
        private double weight;

        private static final double CAP_FRAMES = 6_000; // ~60 s at the 10 ms hop

        /**
         * The combined statistics of the carried weight plus this window; persisted only for
         * committed windows so that polling a partial never changes later results.
         */
        private double[][] combined(float[] features, int nMels, int valid, boolean persist) {
            if (mean == null) {
                mean = new double[nMels];
                variance = new double[nMels];
            }
            double[] combinedMean = new double[nMels];
            double[] combinedVariance = new double[nMels];
            if (valid == 0) {
                System.arraycopy(mean, 0, combinedMean, 0, nMels);
                System.arraycopy(variance, 0, combinedVariance, 0, nMels);
                return new double[][] {combinedMean, combinedVariance};
            }
            double total = weight + valid;
            for (int m = 0; m < nMels; m++) {
                double sum = 0;
                for (int t = 0; t < valid; t++) sum += features[t * nMels + m];
                double windowMean = sum / valid;
                double squares = 0;
                for (int t = 0; t < valid; t++) {
                    double centered = features[t * nMels + m] - windowMean;
                    squares += centered * centered;
                }
                double windowVariance = valid > 1 ? squares / (valid - 1) : 0;
                double delta = windowMean - mean[m];
                combinedVariance[m] =
                        (weight * variance[m]
                                        + valid * windowVariance
                                        + weight * valid / total * delta * delta)
                                / total;
                combinedMean[m] = (weight * mean[m] + valid * windowMean) / total;
            }
            if (persist) {
                System.arraycopy(combinedMean, 0, mean, 0, nMels);
                System.arraycopy(combinedVariance, 0, variance, 0, nMels);
                weight = Math.min(total, CAP_FRAMES);
            }
            return new double[][] {combinedMean, combinedVariance};
        }
    }

    /**
     * NeMo mel front end over {@code pcm[from, from+length)}: frame-major {@code [frames, nMels]},
     * per-feature normalized, with {@code stats} (nullable) carried across windows.
     */
    private float[] mel(float[] pcm, int from, int length, MelStats stats, boolean persist) {
        int frames = melFrames(length);
        int nMels = config.nMels();
        float[] features = logMel.frames(pcm, from, length, config.nFft() / 2, frames);
        int valid = Math.min(length / config.hop(), frames);
        if (stats == null) {
            LogMel.normalizePerFeature(features, nMels, frames, valid);
            return features;
        }
        double[][] combined = stats.combined(features, nMels, valid, persist);
        for (int m = 0; m < nMels; m++) {
            double deviation = Math.sqrt(combined[1][m]) + 1e-5;
            for (int t = 0; t < valid; t++)
                features[t * nMels + m] =
                        (float) ((features[t * nMels + m] - combined[0][m]) / deviation);
            for (int t = valid; t < frames; t++) features[t * nMels + m] = 0f;
        }
        return features;
    }

    private int melFrames(int samples) {
        return 1 + samples / config.hop();
    }

    public Output forward(float[] pcm16k) {
        return forward(pcm16k, null, false, null);
    }

    /**
     * Runs the encoder with running mel statistics shared across a stream's windows; {@code
     * persist} folds this window into the carried statistics and is reserved for windows being
     * committed - previews and tail re-decodes stay side-effect-free.
     */
    public Output forward(float[] pcm16k, MelStats stats, boolean persist) {
        return forward(pcm16k, stats, persist, null);
    }

    public Output forward(float[] pcm16k, ObjIntConsumer<float[]> layerTap) {
        return forward(pcm16k, null, false, layerTap);
    }

    /**
     * Runs the encoder; {@code layerTap} (nullable) receives each conformer block's output as
     * frame-major {@code [frames, dModel]} for parity testing.
     */
    private Output forward(
            float[] pcm16k, MelStats stats, boolean persist, ObjIntConsumer<float[]> layerTap) {
        return Tensors.withScratch(
                workspace -> {
                    MemoryView<MemorySegment> x =
                            encode(pcm16k, 0, pcm16k.length, stats, persist, layerTap, workspace);
                    return new Output(
                            Views.toFloatArray(x, "encoder output"), frames(pcm16k.length));
                });
    }

    /** Encoder frames for {@code samples} of audio. */
    int frames(int samples) {
        return subsampled(subsampled(subsampled(melFrames(samples))));
    }

    /**
     * The pipeline's pass over {@code pcm[from, from+length)}: {@code [frames, dModel]} as a view
     * into {@code workspace}, valid until its next rewind. Every buffer comes from the workspace,
     * so a warm state encodes without allocating.
     */
    MemoryView<MemorySegment> encode(
            float[] pcm,
            int from,
            int length,
            MelStats stats,
            boolean persist,
            ObjIntConsumer<float[]> layerTap,
            Workspace workspace) {
        int melFrames = melFrames(length);
        float[] mel = mel(pcm, from, length, stats, persist);
        // Offline valid-length recurrence seeds at melFrames - 1: the center pad contributes one
        // trailing mel frame that carries no signal.
        int frames = frames(length);
        int valid = subsampled(subsampled(subsampled(melFrames - 1)));
        MemoryView<MemorySegment> x = preEncode(mel, melFrames, frames, valid, workspace);
        if (config.xscaling())
            Ops.multiplyInPlace(x, 0, frames * config.dModel(), (float) Math.sqrt(config.dModel()));
        if (layerTap != null) layerTap.accept(Views.toFloatArray(x, "pre-encode"), -1);
        MemoryView<MemorySegment> positions = positionTable(frames, workspace);
        Scratch work = Scratch.allocate(workspace, frames, config.dModel(), config.ffDim());
        for (int i = 0; i < blocks.length; i++) {
            Block block = blocks[i];
            halfFfn(x, block.ff1Norm(), block.ff1Up(), block.ff1Down(), frames, work);
            attention(x, block, positions, frames, valid, work);
            convolution(x, block, frames, valid, work);
            halfFfn(x, block.ff2Norm(), block.ff2Up(), block.ff2Down(), frames, work);
            block.outNorm().apply(x, x, frames);
            if (layerTap != null) layerTap.accept(Views.toFloatArray(x, "encoder layer"), i);
        }
        return x;
    }

    static int subsampled(int length) {
        return (length - 1) / 2 + 1;
    }

    // --- subsampling: NeMo dw_striding x8 (conv2d s2 + ReLU, then twice depthwise s2 ->
    // pointwise + ReLU), channel-major staging in plain arrays, flattened channel-major. The
    // arrays are the workspace's, so they arrive holding the previous window: every loop below
    // writes its whole output, and the one read-before-written span (flat's padding) is cleared.
    private MemoryView<MemorySegment> preEncode(
            float[] mel, int melFrames, int frames, int valid, Workspace workspace) {
        int channels = config.subsamplingChannels();
        int dim = config.dModel();
        int frequency = config.nMels();
        int t1 = subsampled(melFrames), f1 = subsampled(frequency);
        int t2 = subsampled(t1), f2 = subsampled(f1);
        int t3 = subsampled(t2), f3 = subsampled(f2);
        if (t3 != frames) throw new IllegalStateException("subsampling frame mismatch");

        // conv.0: full 3x3 stride-2 pad-1 (1 -> channels), bias + ReLU.
        float[] s1 = workspace.floatsAtLeast(channels * t1 * f1);
        Parallel.forLoop(
                0,
                channels,
                oc -> {
                    int tapBase = oc * 9;
                    float bias = conv0Bias[oc];
                    for (int ot = 0; ot < t1; ot++) {
                        for (int of = 0; of < f1; of++) {
                            float sum = bias;
                            for (int ky = 0; ky < 3; ky++) {
                                int it = 2 * ot - 1 + ky;
                                if (it < 0 || it >= melFrames) continue;
                                for (int kx = 0; kx < 3; kx++) {
                                    int f = 2 * of - 1 + kx;
                                    if (f < 0 || f >= frequency) continue;
                                    sum +=
                                            conv0Taps[tapBase + ky * 3 + kx]
                                                    * mel[it * frequency + f];
                                }
                            }
                            s1[(oc * t1 + ot) * f1 + of] = Math.max(sum, 0f);
                        }
                    }
                });

        float[] s2 = separable(s1, channels, t1, f1, t2, f2, dw1Taps, dw1Bias, pw1, workspace);
        float[] s3 = separable(s2, channels, t2, f2, t3, f3, dw2Taps, dw2Bias, pw2, workspace);

        // NeMo flattens (B, C, T', F') to (B, T', C*F') - the frame vector is channel-major.
        int flattened = channels * f3;
        float[] flat = workspace.floatsAtLeast(t3 * flattened);
        for (int c = 0; c < channels; c++)
            for (int t = 0; t < valid; t++)
                for (int f = 0; f < f3; f++)
                    flat[t * flattened + c * f3 + f] = s3[(c * t3 + t) * f3 + f];
        // Frames beyond the valid length are zero: the reference masks before the linear.
        Arrays.fill(flat, valid * flattened, t3 * flattened, 0f);

        MemoryView<MemorySegment> flatView = Views.allocateF32(workspace, t3, flattened);
        Views.copyFromArray(flatView, 0, flat, 0, t3 * flattened, "pre-encode flat");
        MemoryView<MemorySegment> x = Views.allocateF32(workspace, frames, dim);
        preOut.apply(flatView, x, frames);
        return x;
    }

    /** One depthwise 3x3 stride-2 conv (bias, no activation) then its pointwise 1x1 and ReLU. */
    private float[] separable(
            float[] in,
            int channels,
            int timeIn,
            int frequencyIn,
            int timeOut,
            int frequencyOut,
            float[] taps,
            float[] bias,
            Linear pointwise,
            Workspace workspace) {
        int positions = timeOut * frequencyOut;
        int size = positions * channels;
        float[] positionsMajor = workspace.floatsAtLeast(size);
        Parallel.forLoop(
                0,
                channels,
                c -> {
                    int tapBase = c * 9;
                    int inBase = c * timeIn * frequencyIn;
                    for (int ot = 0; ot < timeOut; ot++) {
                        for (int of = 0; of < frequencyOut; of++) {
                            float sum = bias[c];
                            for (int ky = 0; ky < 3; ky++) {
                                int it = 2 * ot - 1 + ky;
                                if (it < 0 || it >= timeIn) continue;
                                for (int kx = 0; kx < 3; kx++) {
                                    int f = 2 * of - 1 + kx;
                                    if (f < 0 || f >= frequencyIn) continue;
                                    sum +=
                                            taps[tapBase + ky * 3 + kx]
                                                    * in[inBase + it * frequencyIn + f];
                                }
                            }
                            positionsMajor[(ot * frequencyOut + of) * channels + c] = sum;
                        }
                    }
                });
        MemoryView<MemorySegment> pwIn = Views.allocateF32(workspace, positions, channels);
        Views.copyFromArray(pwIn, 0, positionsMajor, 0, size, "depthwise conv");
        MemoryView<MemorySegment> pwOut = Views.allocateF32(workspace, positions, channels);
        pointwise.apply(pwIn, pwOut, positions);
        float[] mixed = positionsMajor; // consumed by the copy above: reuse it for the result
        Views.copyToArray(pwOut, 0, mixed, 0, size, "pointwise conv");
        float[] out = workspace.floatsAtLeast(size);
        for (int p = 0; p < positions; p++)
            for (int c = 0; c < channels; c++)
                out[c * positions + p] = Math.max(mixed[p * channels + c], 0f);
        return out;
    }

    // --- Transformer-XL relative positions ---

    /** Sinusoidal table for {@code 2*frames-1} relative positions {@code +(T-1)..-(T-1)}. */
    static float[] relativePositions(int frames, int dModel) {
        return relativePositions(frames, dModel, new float[(2 * frames - 1) * dModel]);
    }

    /** As above, into {@code table} (at least {@code (2*frames-1)*dModel} long). */
    static float[] relativePositions(int frames, int dModel, float[] table) {
        int half = dModel / 2;
        int rows = 2 * frames - 1;
        double scale = -Math.log(10_000.0) / dModel;
        for (int row = 0; row < rows; row++) {
            int position = frames - 1 - row;
            for (int i = 0; i < half; i++) {
                double angle = position * Math.exp(2 * i * scale);
                table[row * dModel + 2 * i] = (float) Math.sin(angle);
                table[row * dModel + 2 * i + 1] = (float) Math.cos(angle);
            }
        }
        return table;
    }

    /** The raw table as a view; every block projects it with its own {@code linear_pos}. */
    private MemoryView<MemorySegment> positionTable(int frames, Workspace workspace) {
        int dim = config.dModel(), rows = 2 * frames - 1;
        float[] table = relativePositions(frames, dim, workspace.floatsAtLeast(rows * dim));
        MemoryView<MemorySegment> view = Views.allocateF32(workspace, rows, dim);
        Views.copyFromArray(view, 0, table, 0, rows * dim, "relative positions");
        return view;
    }

    record Scratch(
            MemoryView<MemorySegment> norm,
            MemoryView<MemorySegment> ff,
            MemoryView<MemorySegment> queryU,
            MemoryView<MemorySegment> queryV,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> attention,
            MemoryView<MemorySegment> position,
            MemoryView<MemorySegment> pointwise,
            MemoryView<MemorySegment> glu,
            MemoryView<MemorySegment> scores, // one head's [valid][valid] probabilities
            MemoryView<MemorySegment> positionScores, // one head's [valid][2*valid-1]
            MemoryView<MemorySegment> valueT, // [dim][frames]
            MemoryView<MemorySegment> attentionT, // [dim][frames]
            float[] gated, // conv module staging, [frames][dim]
            float[] mixed) {
        static Scratch allocate(Workspace workspace, int frames, int dim, int ffDim) {
            int positionRows = 2 * frames - 1;
            return new Scratch(
                    Views.allocateF32(workspace, frames, dim),
                    Views.allocateF32(workspace, frames, ffDim),
                    Views.allocateF32(workspace, frames, dim),
                    Views.allocateF32(workspace, frames, dim),
                    Views.allocateF32(workspace, frames, dim),
                    Views.allocateF32(workspace, frames, dim),
                    Views.allocateF32(workspace, frames, dim),
                    Views.allocateF32(workspace, positionRows, dim),
                    Views.allocateF32(workspace, frames, dim * 2),
                    Views.allocateF32(workspace, frames, dim),
                    Views.allocateF32(workspace, frames, frames),
                    Views.allocateF32(workspace, frames, positionRows),
                    Views.allocateF32(workspace, dim, frames),
                    Views.allocateF32(workspace, dim, frames),
                    workspace.floatsAtLeast(frames * dim),
                    workspace.floatsAtLeast(frames * dim));
        }
    }

    private void halfFfn(
            MemoryView<MemorySegment> x,
            Norm norm,
            Linear up,
            Linear down,
            int frames,
            Scratch work) {
        int ffDim = config.ffDim();
        norm.apply(work.norm(), x, frames);
        up.apply(work.norm(), work.ff(), frames);
        Parallel.forLoop(frames, row -> Ops.siluInPlace(work.ff(), (long) row * ffDim, ffDim));
        down.apply(work.ff(), work.norm(), frames); // the up projection has consumed norm
        Ops.saxpyInPlace(x, 0, work.norm(), 0, Math.multiplyExact(frames, config.dModel()), 0.5f);
    }

    private void attention(
            MemoryView<MemorySegment> x,
            Block block,
            MemoryView<MemorySegment> positionTable,
            int frames,
            int valid,
            Scratch work) {
        int dim = config.dModel();
        block.attnNorm().apply(work.norm(), x, frames);
        block.query().apply(work.norm(), work.queryU(), frames);
        block.key().apply(work.norm(), work.key(), frames);
        block.value().apply(work.norm(), work.value(), frames);
        block.position().apply(positionTable, work.position(), 2 * frames - 1);
        // Transformer-XL's learned query biases: u for the content term, v for the position term
        Convert.copyF32(work.queryU(), 0, work.queryV(), 0, (long) frames * dim);
        Ops.addRowBiasInPlace(work.queryU(), 0, block.posBiasU(), 0, frames, dim);
        Ops.addRowBiasInPlace(work.queryV(), 0, block.posBiasV(), 0, frames, dim);
        relativeAttention(work, frames, valid, dim, config.heads());
        block.output().apply(work.attention(), work.norm(), frames);
        Ops.addInPlace(x, 0, work.norm(), 0, Math.multiplyExact(frames, dim));
    }

    /**
     * Transformer-XL relative attention, {@code queryU/queryV/key/value/position -> attention}, as
     * three jam GEMMs per head: {@code (queryU·keyᵀ + shift(queryV·positionᵀ)) * scale}, softmax,
     * then the value mix. Only the valid block is computed: padded key columns are masked, and
     * padded query rows are zeroed right after anyway, so the reference's unmasked padded-row
     * softmax never reaches the output.
     *
     * <p>The value mix runs as {@code attentionᵀ = valueᵀ·scoresᵀ} (m = valid queries): with m =
     * headDim instead, jam splits 128 rows over the pool and runs at a third of the speed.
     */
    static void relativeAttention(Scratch work, int frames, int valid, int dim, int heads) {
        int headDim = dim / heads;
        int positionRows = 2 * valid - 1; // relative offsets +(valid-1)..-(valid-1)
        long positionStart = (long) (frames - valid) * dim; // offset +(valid-1) in the table
        float scale = (float) (1.0 / Math.sqrt(headDim));
        Ops.transposeCopy(work.value(), frames, dim, work.valueT());
        for (int head = 0; head < heads; head++) { // sequential: mm refuses nested regions
            long headBase = (long) head * headDim;
            MatMul.mm(
                    work.key(),
                    headBase,
                    dim,
                    work.queryU(),
                    headBase,
                    dim,
                    work.scores(),
                    0,
                    valid,
                    valid,
                    valid,
                    headDim);
            MatMul.mm(
                    work.position(),
                    positionStart + headBase,
                    dim,
                    work.queryV(),
                    headBase,
                    dim,
                    work.positionScores(),
                    0,
                    positionRows,
                    positionRows,
                    valid,
                    headDim);
            Parallel.forLoop(
                    valid,
                    query -> {
                        long row = (long) query * valid;
                        // score[q][k] takes relative offset q-k, at column valid-1-q+k
                        Ops.addInPlace(
                                work.scores(),
                                row,
                                work.positionScores(),
                                (long) query * positionRows + valid - 1 - query,
                                valid);
                        Ops.multiplyInPlace(work.scores(), row, valid, scale);
                        Ops.softmaxInPlace(work.scores(), row, valid);
                    });
            MatMul.mm(
                    work.scores(),
                    0,
                    valid,
                    work.valueT(),
                    headBase * frames,
                    frames,
                    work.attentionT(),
                    headBase * frames,
                    frames,
                    valid,
                    headDim,
                    valid);
        }
        Ops.transposeCopy(work.attentionT(), dim, frames, work.attention());
        Ops.fillInPlace(work.attention(), (long) valid * dim, (frames - valid) * dim, 0f);
    }

    private void convolution(
            MemoryView<MemorySegment> x, Block block, int frames, int valid, Scratch work) {
        int dim = config.dModel(), kernel = config.convKernel(), pad = (kernel - 1) / 2;
        block.convNorm().apply(work.norm(), x, frames);
        block.pointwise1().apply(work.norm(), work.pointwise(), frames);
        Parallel.forLoop(
                frames,
                row ->
                        Activations.glu(
                                work.glu(),
                                (long) row * dim,
                                work.pointwise(),
                                (long) row * dim * 2,
                                dim));
        Ops.fillInPlace(work.glu(), (long) valid * dim, (frames - valid) * dim, 0f);

        float[] gated = work.gated(), mixed = work.mixed();
        Views.copyToArray(work.glu(), 0, gated, 0, frames * dim, "conv glu");
        float[] taps = block.depthwise();
        float[] bnScale = block.bnScale(), bnShift = block.bnShift();
        Parallel.forLoop(
                0,
                config.dModel(),
                c -> {
                    int tapBase = c * kernel;
                    for (int t = 0; t < frames; t++) {
                        float sum = 0f;
                        for (int k = 0; k < kernel; k++) {
                            int at = t + k - pad;
                            if (at < 0 || at >= frames) continue;
                            sum += taps[tapBase + k] * gated[at * dim + c];
                        }
                        float normalized = sum * bnScale[c] + bnShift[c];
                        mixed[t * dim + c] = (float) (normalized / (1.0 + Math.exp(-normalized)));
                    }
                });
        Views.copyFromArray(work.norm(), 0, mixed, 0, frames * dim, "conv mixed");
        block.pointwise2().apply(work.norm(), work.glu(), frames);
        Ops.addInPlace(x, 0, work.glu(), 0, Math.multiplyExact(frames, dim));
    }
}
