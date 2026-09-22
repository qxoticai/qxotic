package com.qxotic.jinfer.models.parakeet;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.Workspace;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.LogMel;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.function.ObjIntConsumer;

/**
 * NVIDIA Parakeet FastConformer encoder (NeMo {@code ConformerEncoder}): NeMo mel front end, 8x
 * depthwise-separable conv subsampling, then macaron Conformer blocks with Transformer-XL
 * relative-position attention. Ported from parakeet.cpp and checked against its fixtures.
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

    /** A projection; {@code bias} is null on bias-free checkpoints such as tdt-0.6b-v3. */
    record Linear(
            MemoryView<MemorySegment> weight, MemoryView<MemorySegment> bias, int in, int out) {
        /** {@code output[rows][out] = input[rows][in] · weightᵀ + bias}. */
        void apply(MemoryView<MemorySegment> input, MemoryView<MemorySegment> output, int rows) {
            apply(input, output, 0, rows);
        }

        /** As above, into {@code output} from row {@code outputRow} on. */
        void apply(
                MemoryView<MemorySegment> input,
                MemoryView<MemorySegment> output,
                int outputRow,
                int rows) {
            long offset = (long) outputRow * out;
            MatMul.mm(weight, 0, in, input, 0, in, output, offset, out, out, rows, in);
            if (bias != null) Ops.addRowBiasInPlace(output, offset, bias, 0, rows, out);
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

    static ParakeetEncoder load(GGUF gguf, Tensors tensors) {
        Config config = Config.from(gguf);
        if (config.magPower() != 1f && config.magPower() != 2f)
            throw new IllegalArgumentException("unsupported mag_power " + config.magPower());
        int channels = config.subsamplingChannels();
        int dim = config.dModel(), ffDim = config.ffDim();
        int flattened = channels * subsampled(subsampled(subsampled(config.nMels())));
        Block[] blocks = new Block[config.layers()];
        for (int i = 0; i < blocks.length; i++) {
            String prefix = "encoder.layers." + i + ".";
            float[] bnWeight = tensors.floats(prefix + "conv.batch_norm.weight", dim);
            float[] bnBias = tensors.floats(prefix + "conv.batch_norm.bias", dim);
            float[] mean = tensors.floats(prefix + "conv.batch_norm.running_mean", dim);
            float[] variance = tensors.floats(prefix + "conv.batch_norm.running_var", dim);
            // The depthwise conv's bias (absent on v2/v3) precedes the norm, so it folds into
            // the same per-channel affine: (x + b)*scale + shift = x*scale + (shift + b*scale).
            String dwBiasName = prefix + "conv.depthwise_conv.bias";
            float[] dwBias =
                    tensors.optional(dwBiasName) == null ? null : tensors.floats(dwBiasName, dim);
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
                            tensors.vector(prefix + "self_attn.pos_bias_u", dim),
                            tensors.vector(prefix + "self_attn.pos_bias_v", dim),
                            norm(tensors, prefix + "norm_conv"),
                            linear(tensors, prefix + "conv.pointwise_conv1", dim, 2 * dim),
                            tensors.floats(
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
                tensors.floats("encoder.pre_encode.conv.0.weight", channels * 9),
                tensors.floats("encoder.pre_encode.conv.0.bias", channels),
                tensors.floats("encoder.pre_encode.conv.2.weight", channels * 9),
                tensors.floats("encoder.pre_encode.conv.2.bias", channels),
                biasedLinear(tensors, "encoder.pre_encode.conv.3", channels, channels),
                tensors.floats("encoder.pre_encode.conv.5.weight", channels * 9),
                tensors.floats("encoder.pre_encode.conv.5.bias", channels),
                biasedLinear(tensors, "encoder.pre_encode.conv.6", channels, channels),
                biasedLinear(tensors, "encoder.pre_encode.out", flattened, dim),
                blocks);
    }

    /** A conformer projection, whose bias is optional. */
    private static Linear linear(Tensors tensors, String name, int in, int out) {
        return new Linear(
                tensors.weight(name + ".weight", out), tensors.optional(name + ".bias"), in, out);
    }

    /** A subsampling projection, whose bias every checkpoint has. */
    private static Linear biasedLinear(Tensors tensors, String name, int in, int out) {
        return new Linear(
                tensors.weight(name + ".weight", out), tensors.require(name + ".bias"), in, out);
    }

    private static Norm norm(Tensors tensors, String name) {
        return new Norm(tensors.require(name + ".weight"), tensors.require(name + ".bias"));
    }

    /** The NeMo featurizer, with the window and filterbank taken from the weights. */
    private static LogMel featurizer(GGUF gguf, Tensors tensors, Config config) {
        int winLength = gguf.getValue(int.class, "parakeet.preprocessor.win_length");
        float[] window = tensors.floats("preprocessor.featurizer.window", winLength);
        float[] centered = new float[config.nFft()];
        System.arraycopy(window, 0, centered, (config.nFft() - winLength) / 2, winLength);
        float[] filterbank =
                tensors.floats(
                        "preprocessor.featurizer.fb", config.nMels() * (config.nFft() / 2 + 1));
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
     * Log-mel features of {@code pcm[from, from + length)}, normalized per feature: {@code [frames,
     * nMels]}.
     */
    float[] mel(float[] pcm, int from, int length) {
        int frames = melFrames(length);
        int nMels = config.nMels();
        float[] features = logMel.frames(pcm, from, length, config.nFft() / 2, frames);
        LogMel.normalizePerFeature(
                features, nMels, frames, Math.min(length / config.hop(), frames));
        return features;
    }

    int melFrames(int samples) {
        return 1 + samples / config.hop();
    }

    /** Encoder frames for {@code samples} of audio. */
    int frames(int samples) {
        return subsampled(subsampled(subsampled(melFrames(samples))));
    }

    /**
     * Encodes {@code pcm[from, from + length)} to {@code [frames, dModel]}, drawing every buffer
     * from {@code workspace}. The result is valid until the workspace's next rewind.
     */
    MemoryView<MemorySegment> encode(
            float[] pcm,
            int from,
            int length,
            ObjIntConsumer<float[]> layerTap,
            Workspace workspace) {
        int melFrames = melFrames(length);
        float[] mel = mel(pcm, from, length);
        // The valid length starts from melFrames - 1: the center pad adds one trailing mel frame
        // that carries no signal.
        int frames = frames(length);
        int valid = subsampled(subsampled(subsampled(melFrames - 1)));
        MemoryView<MemorySegment> x =
                preEncode(mel, melFrames, frames, valid, SUBSAMPLING_TILE, workspace);
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

    // Subsampling, NeMo's dw_striding x8: a stride-2 conv2d and ReLU, then twice a stride-2
    // depthwise conv, a pointwise conv and ReLU. Stages are channel-major arrays from the
    // workspace, which still hold the previous window's values, so every loop writes its whole
    // output and flat's padding is cleared explicitly.

    /**
     * Encoder frames per subsampling tile. Each stride-2 3x3 stage reads input rows {@code
     * 2t-1..2t+1}, so a tile needs a 15-mel-frame halo, and its intermediates stay near 40 MB for
     * any window length (a whole 65 s window took about 500 MB).
     */
    private static final int SUBSAMPLING_TILE = 64;

    /** The subsampled frames {@code [frames, dModel]}, computed {@code tile} frames at a time. */
    MemoryView<MemorySegment> preEncode(
            float[] mel, int melFrames, int frames, int valid, int tile, Workspace workspace) {
        int channels = config.subsamplingChannels();
        int frequency = config.nMels();
        int t1 = subsampled(melFrames), f1 = subsampled(frequency);
        int t2 = subsampled(t1), f2 = subsampled(f1);
        int t3 = subsampled(t2), f3 = subsampled(f2);
        if (t3 != frames) throw new IllegalStateException("subsampling frame mismatch");
        int flattened = channels * f3;
        MemoryView<MemorySegment> x = Views.allocateF32(workspace, frames, config.dModel());
        for (int j0 = 0; j0 < frames; j0 += tile) {
            int j1 = (int) Math.min(frames, (long) j0 + tile);
            // the rows each earlier stage must produce for output rows [j0, j1), clamped
            int a2 = Math.max(0, 2 * j0 - 1), b2 = Math.min(t2, 2 * j1);
            int a1 = Math.max(0, 2 * a2 - 1), b1 = Math.min(t1, 2 * b2);
            try (var scope = Workspace.scope(workspace)) {
                float[] s1 = conv0(mel, melFrames, a1, b1, f1, workspace);
                float[] s2 =
                        separable(s1, a1, b1, t1, f1, a2, b2, f2, dw1Taps, dw1Bias, pw1, workspace);
                float[] s3 =
                        separable(s2, a2, b2, t2, f2, j0, j1, f3, dw2Taps, dw2Bias, pw2, workspace);

                // NeMo flattens (B, C, T', F') to (B, T', C*F'), so the frame vector is
                // channel-major. Frames past the valid length are zero, as the reference masks
                // them before the linear.
                int rows = j1 - j0, live = Math.max(0, Math.min(j1, valid) - j0);
                float[] flat = workspace.floatsAtLeast(rows * flattened);
                for (int c = 0; c < channels; c++)
                    for (int t = 0; t < live; t++)
                        for (int f = 0; f < f3; f++)
                            flat[t * flattened + c * f3 + f] = s3[(c * rows + t) * f3 + f];
                Arrays.fill(flat, live * flattened, rows * flattened, 0f);
                MemoryView<MemorySegment> flatView = Views.allocateF32(workspace, rows, flattened);
                Views.copyFromArray(flatView, 0, flat, 0, rows * flattened, "pre-encode flat");
                preOut.apply(flatView, x, j0, rows);
            }
        }
        return x;
    }

    /**
     * conv.0, a full 3x3 stride-2 conv from 1 to {@code channels} channels with bias and ReLU, for
     * output rows {@code [from, to)}, as {@code [channels][to-from][f1]}.
     */
    private float[] conv0(
            float[] mel, int melFrames, int from, int to, int f1, Workspace workspace) {
        int channels = config.subsamplingChannels(), frequency = config.nMels(), rows = to - from;
        float[] out = workspace.floatsAtLeast(channels * rows * f1);
        Parallel.forLoop(
                0,
                channels,
                oc -> {
                    int tapBase = oc * 9;
                    float bias = conv0Bias[oc];
                    for (int ot = from; ot < to; ot++) {
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
                            out[(oc * rows + ot - from) * f1 + of] = Math.max(sum, 0f);
                        }
                    }
                });
        return out;
    }

    /**
     * A depthwise 3x3 stride-2 conv with bias, then a pointwise conv and ReLU, for output rows
     * {@code [outFrom, outTo)}. {@code in} holds rows {@code [inFrom, inTo)} of a {@code
     * timeIn}-row input, channel-major. Time indices are global, so zero padding only ever applies
     * at the real edges, never at a tile boundary.
     */
    private float[] separable(
            float[] in,
            int inFrom,
            int inTo,
            int timeIn,
            int frequencyIn,
            int outFrom,
            int outTo,
            int frequencyOut,
            float[] taps,
            float[] bias,
            Linear pointwise,
            Workspace workspace) {
        int channels = config.subsamplingChannels();
        int inRows = inTo - inFrom;
        int positions = (outTo - outFrom) * frequencyOut;
        int size = positions * channels;
        float[] positionsMajor = workspace.floatsAtLeast(size);
        Parallel.forLoop(
                0,
                channels,
                c -> {
                    int tapBase = c * 9;
                    int inBase = c * inRows * frequencyIn;
                    for (int ot = outFrom; ot < outTo; ot++) {
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
                                                    * in[inBase + (it - inFrom) * frequencyIn + f];
                                }
                            }
                            positionsMajor[((ot - outFrom) * frequencyOut + of) * channels + c] =
                                    sum;
                        }
                    }
                });
        MemoryView<MemorySegment> pwIn = Views.allocateF32(workspace, positions, channels);
        Views.copyFromArray(pwIn, 0, positionsMajor, 0, size, "depthwise conv");
        MemoryView<MemorySegment> pwOut = Views.allocateF32(workspace, positions, channels);
        pointwise.apply(pwIn, pwOut, positions);
        float[] mixed = positionsMajor; // already copied into pwIn, so reused
        Views.copyToArray(pwOut, 0, mixed, 0, size, "pointwise conv");
        float[] out = workspace.floatsAtLeast(size);
        for (int p = 0; p < positions; p++)
            for (int c = 0; c < channels; c++)
                out[c * positions + p] = Math.max(mixed[p * channels + c], 0f);
        return out;
    }

    /**
     * Fills {@code table} with the sinusoidal encodings of the relative positions {@code frames-1}
     * down to {@code 1-frames}, {@code dModel} values each.
     */
    static float[] relativePositions(int frames, int dModel, float[] table) {
        int half = dModel / 2;
        int rows = 2 * frames - 1;
        double scale = -Math.log(10_000.0) / dModel;
        double[] frequencies = new double[half]; // per column, the same for every row
        for (int i = 0; i < half; i++) frequencies[i] = Math.exp(2 * i * scale);
        for (int row = 0; row < rows; row++) {
            int position = frames - 1 - row;
            for (int i = 0; i < half; i++) {
                double angle = position * frequencies[i];
                table[row * dModel + 2 * i] = (float) Math.sin(angle);
                table[row * dModel + 2 * i + 1] = (float) Math.cos(angle);
            }
        }
        return table;
    }

    /** The table as a view; each block projects it with its own {@code linear_pos}. */
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
     * Transformer-XL relative attention as three GEMMs per head: {@code (queryU·keyᵀ +
     * shift(queryV·positionᵀ)) * scale}, softmax, then the value mix. Only the valid block is
     * computed: padded key columns are masked, and padded query rows are zeroed afterwards, so the
     * reference's unmasked softmax over padded rows never reaches the output.
     *
     * <p>The value mix runs as {@code attentionᵀ = valueᵀ·scoresᵀ} so that m is the number of valid
     * queries: with m = headDim, jam splits 128 rows over the pool and runs 3x slower.
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
