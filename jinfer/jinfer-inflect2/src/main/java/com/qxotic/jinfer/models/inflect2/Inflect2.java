// Inflect2 - VITS-family text-to-waveform model (Nano 3.97M, Micro 9.36M; F16/Q8_0/Q4_0 GGUF).
//
//   Inflect2 model = Inflect2.load(Path.of("model.gguf"), arena);
//   Media.Audio audio = model.synthesize(model.newState(), tokens, 1.0f, 0.667f, seed);
//
// The pipeline, one step each: embed the phoneme tokens and run a relative-attention transformer
// (encoder); project to a per-token latent mean/log-scale and a log-duration; repeat each token's
// latent for its duration and add noise (prior); invert the coupling layers (flow); upsample to a
// waveform with a HiFi-GAN decoder.
//
// A loaded model is immutable and its weights are read-only, so one instance serves any number of
// concurrent synthesize() calls, each with its own State.
package com.qxotic.jinfer.models.inflect2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.LeakWatch;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

/**
 * VITS-family text-to-waveform model; most callers want {@link InflectTTS}, which adds text
 * normalization and phonemization on top. Synthesize directly only when you already have phoneme
 * tokens.
 */
public final class Inflect2 {

    /** HiFi-GAN leaky-ReLU slope, and torch's default slope for the activation before conv_post. */
    private static final float LEAKY = 0.1f, FINAL_LEAKY = 0.01f;

    /**
     * Frame ceiling per call - a runaway log-duration must fail, not exhaust memory. A DoS BOUND,
     * not a modelling constant: it caps one chunk at ~43 s of audio and every buffer sized off it,
     * and it is what stops adversarial text (or a tiny speed) from turning one request into a
     * multi-gigabyte allocation. Raising it raises the worst case a single request can cost.
     */
    private static final int MAX_FRAMES = 4000;

    // Kernel sizes the GGUF metadata does not carry - they are part of the architecture, exactly as
    // in the reference model definition (the rest come from the config: the encoder FFN's, the
    // resblocks', the upsamplers').
    private static final int POINTWISE = 1; // projections: attention q/k/v/o, res/skip, proj
    private static final int VOCODER_KERNEL = 7; // dec.conv_pre and dec.conv_post
    private static final int WAVENET_KERNEL = 5; // the flow's WaveNet gates
    private static final int DURATION_KERNEL = 3; // the duration predictor's two convolutions

    /** Model dimensions read from GGUF metadata; the layer shapes come from the weights. */
    public record Configuration(
            int symbolCount,
            int interChannels,
            int hiddenChannels,
            int filterChannels,
            int nHeads,
            int nLayers,
            int kernelSize,
            int sampleRate,
            int upsampleInitialChannel,
            int[] resblockKernelSizes,
            int[] resblockDilationSizes,
            int[] upsampleRates,
            int[] upsampleKernelSizes) {

        public Configuration {
            resblockKernelSizes = resblockKernelSizes.clone();
            resblockDilationSizes = resblockDilationSizes.clone();
            upsampleRates = upsampleRates.clone();
            upsampleKernelSizes = upsampleKernelSizes.clone();
        }

        @Override
        public int[] resblockKernelSizes() {
            return resblockKernelSizes.clone();
        }

        @Override
        public int[] resblockDilationSizes() {
            return resblockDilationSizes.clone();
        }

        @Override
        public int[] upsampleRates() {
            return upsampleRates.clone();
        }

        @Override
        public int[] upsampleKernelSizes() {
            return upsampleKernelSizes.clone();
        }

        /** The phoneme symbol table this model consumes - its token space. */
        public int vocabularySize() {
            return symbolCount;
        }

        /** The frame ceiling: a runaway log-duration must fail, not exhaust memory. */
        public int contextLength() {
            return MAX_FRAMES;
        }
    }

    // ── weights ───────────────────────────────────────────────────────────
    // Resolved once at load, so the forward pass names layers instead of building tensor keys and
    // carrying dimensions. Each layer's widths come from the file where the file is unambiguous:
    // the output width always, the attention window from the relative embedding's shape, the
    // duration predictor's width from its first convolution. Kernels cannot: a dense file keeps the
    // PyTorch shape [kernel, inChannels, outChannels], but a quantized one flattens it to
    // [kernel*inChannels, outChannels] and pads that row to a block boundary, so the split is lost.
    // They are stated below and by the config, as the reference model definition states them.
    //
    // The loader resolves dtype once: every element-accessed tensor (embedding, relative
    // embeddings, norms, biases) is dequantized to F32 once, and a conv weight that is neither
    // FP32 nor Q8_0 is dequantized to F32 too, so a forward only ever meets FP32 or Q8_0.
    // quantizedLayout remembers the FILE's encoding for the one place the layouts differ:
    // a transposed conv's tap ordering (see dequantizeTransposed).

    /** One 1-D convolution. {@code bias} is null where folded weight norm left none. */
    public record Conv(
            MemoryView<MemorySegment> weight,
            MemoryView<MemorySegment> bias,
            int kernel,
            int inChannels,
            int outChannels,
            boolean quantizedLayout) {

        /** Taps per output channel. */
        int taps() {
            return kernel * inChannels;
        }

        /** Elements per stored row - more than {@link #taps} when rows are padded to a block. */
        int rowStride() {
            return Math.toIntExact(weight.logicalSize() / outChannels);
        }
    }

    /** LayerNorm parameters over {@code channels} contiguous channels. */
    public record Norm(
            MemoryView<MemorySegment> gamma, MemoryView<MemorySegment> beta, int channels) {}

    /** One encoder block: relative-position self-attention, then a convolutional feed-forward. */
    public record EncoderLayer(
            Conv query,
            Conv key,
            Conv value,
            Conv output,
            MemoryView<MemorySegment> relativeKeys,
            MemoryView<MemorySegment> relativeValues,
            int window,
            Norm attentionNorm,
            Conv expand,
            Conv contract,
            Norm ffnNorm) {}

    /** The deterministic duration predictor. */
    public record Durations(
            Conv first, Norm firstNorm, Conv second, Norm secondNorm, Conv project) {}

    /** One coupling layer: a WaveNet over half the channels predicts a shift for the other half. */
    public record Coupling(Conv pre, Conv[] gates, Conv[] residualSkip, Conv post) {}

    /** One resblock: dilated convolution pairs, each added back into the block's running value. */
    public record ResBlock(Conv[] filter, Conv[] project, int[] dilations) {}

    /** The HiFi-GAN decoder: upsample, and at each rate a bank of resblocks that gets averaged. */
    public record Decoder(
            Conv pre, Conv[] upsample, int[] rates, ResBlock[][] resblocks, Conv post) {}

    public record Weights(
            MemoryView<MemorySegment> embedding,
            int embeddingStride,
            EncoderLayer[] encoder,
            Conv project,
            Durations durations,
            Coupling[] flow,
            Decoder decoder) {}

    private final Configuration cfg;
    private final Weights w;

    /** What the file held, for reporting - the weights themselves are resolved into {@link #w}. */
    private final int tensorCount;

    private final long parameterCount;

    private Inflect2(Configuration cfg, Weights weights, int tensorCount, long parameterCount) {
        this.cfg = cfg;
        this.w = weights;
        this.tensorCount = tensorCount;
        this.parameterCount = parameterCount;
    }

    // ── loading ───────────────────────────────────────────────────────────

    /**
     * Weights map into {@code arena}, and whoever provides it owns its lifetime - the same contract
     * as {@link com.qxotic.jinfer.Model}: {@code ofAuto} is GC-managed, {@code global} lasts the
     * process, a scoped arena is deterministic and must outlive every model sharing the weights.
     * Closing one while a synthesis is running is a crash, not an exception: the kernels read raw
     * addresses.
     */
    public static Inflect2 load(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return load(channel, gguf, arena);
        }
    }

    /**
     * As {@link #load(Path, Arena)} but reusing an already-parsed {@code gguf} - the arch-dispatch
     * entry, where the header has been read to decide which port to call.
     */
    public static Inflect2 load(FileChannel channel, GGUF gguf, Arena arena) throws IOException {
        return load(channel, gguf, 0, arena);
    }

    /**
     * As {@link #load(FileChannel, GGUF, Arena)}, with the GGUF beginning at {@code baseOffset} in
     * {@code channel}. This supports models embedded in a larger file without coupling the model to
     * any particular container format.
     */
    public static Inflect2 load(FileChannel channel, GGUF gguf, long baseOffset, Arena arena)
            throws IOException {
        return load(gguf, ModelLoader.loadTensors(channel, gguf, baseOffset, arena), arena);
    }

    private static Inflect2 load(
            GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors, Arena arena) {
        Configuration config = readConfig(gguf);
        long parameters = 0;
        for (MemoryView<MemorySegment> tensor : tensors.values())
            parameters += tensor.logicalSize();
        return new Inflect2(
                config,
                loadWeights(tensors, config, MemoryAllocators.ofArena(arena)),
                tensors.size(),
                parameters);
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors,
            Configuration config,
            MemoryArena<MemorySegment> allocator) {
        int hidden = config.hiddenChannels(), latent = config.interChannels();
        MemoryView<MemorySegment> embedding = ModelLoader.require(tensors, "enc_p.emb.weight");

        EncoderLayer[] encoder = new EncoderLayer[config.nLayers()];
        for (int i = 0; i < encoder.length; i++) {
            String attention = "enc_p.encoder.attn_layers." + i + ".";
            String ffn = "enc_p.encoder.ffn_layers." + i + ".";
            MemoryView<MemorySegment> relativeKeys =
                    ModelLoader.require(tensors, attention + "emb_rel_k");
            MemoryView<MemorySegment> relativeValues =
                    ModelLoader.require(tensors, attention + "emb_rel_v");
            int relativePositions = Math.toIntExact(relativeKeys.shape().size(0));
            int headChannels = hidden / config.nHeads();
            require(
                    relativePositions > 0
                            && (relativePositions & 1) == 1
                            && relativeKeys.logicalSize() == (long) relativePositions * headChannels
                            && relativeValues.logicalSize() == relativeKeys.logicalSize(),
                    attention + "relative embeddings have the wrong shape");
            encoder[i] =
                    new EncoderLayer(
                            conv(
                                    tensors,
                                    allocator,
                                    attention + "conv_q",
                                    POINTWISE,
                                    hidden,
                                    hidden),
                            conv(
                                    tensors,
                                    allocator,
                                    attention + "conv_k",
                                    POINTWISE,
                                    hidden,
                                    hidden),
                            conv(
                                    tensors,
                                    allocator,
                                    attention + "conv_v",
                                    POINTWISE,
                                    hidden,
                                    hidden),
                            conv(
                                    tensors,
                                    allocator,
                                    attention + "conv_o",
                                    POINTWISE,
                                    hidden,
                                    hidden),
                            dequantToF32(allocator, relativeKeys, attention + "emb_rel_k"),
                            dequantToF32(allocator, relativeValues, attention + "emb_rel_v"),
                            // [2*window+1, headChannels]: keys this far either side get an
                            // embedding (the view's dims are the GGUF's, reversed)
                            (relativePositions - 1) / 2,
                            norm(tensors, allocator, "enc_p.encoder.norm_layers_1." + i, hidden),
                            conv(
                                    tensors,
                                    allocator,
                                    ffn + "conv_1",
                                    config.kernelSize(),
                                    hidden,
                                    config.filterChannels()),
                            conv(
                                    tensors,
                                    allocator,
                                    ffn + "conv_2",
                                    config.kernelSize(),
                                    config.filterChannels(),
                                    hidden),
                            norm(tensors, allocator, "enc_p.encoder.norm_layers_2." + i, hidden));
        }

        // The file interleaves the couplings with the flips applied between them, hence the 2*i.
        int couplings = 0;
        while (tensors.containsKey("flow.flows." + (2 * couplings) + ".pre.weight")) couplings++;
        if (couplings == 0) throw new IllegalArgumentException("no flow coupling layers in model");
        Coupling[] flow = new Coupling[couplings];
        for (int i = 0; i < couplings; i++) {
            String root = "flow.flows." + (2 * i) + ".";
            Conv pre = conv(tensors, allocator, root + "pre", POINTWISE, latent / 2);
            int wide = pre.outChannels();
            int layers = 0;
            while (tensors.containsKey(root + "enc.in_layers." + layers + ".weight")) layers++;
            require(layers > 0, root + "has no WaveNet layers");
            Conv[] gates = new Conv[layers], residualSkip = new Conv[layers];
            for (int layer = 0; layer < layers; layer++) {
                gates[layer] =
                        conv(
                                tensors,
                                allocator,
                                root + "enc.in_layers." + layer,
                                WAVENET_KERNEL,
                                wide,
                                2 * wide);
                residualSkip[layer] =
                        conv(
                                tensors,
                                allocator,
                                root + "enc.res_skip_layers." + layer,
                                POINTWISE,
                                wide,
                                layer == layers - 1 ? wide : 2 * wide);
            }
            flow[i] =
                    new Coupling(
                            pre,
                            gates,
                            residualSkip,
                            conv(tensors, allocator, root + "post", POINTWISE, wide, latent / 2));
        }

        int[] rates = config.upsampleRates(), dilations = config.resblockDilationSizes();
        int[] upsampleKernels = config.upsampleKernelSizes();
        int[] blockKernels = config.resblockKernelSizes();
        int blocks = blockKernels.length, perBlock = dilations.length / blocks;
        Conv pre =
                conv(
                        tensors,
                        allocator,
                        "dec.conv_pre",
                        VOCODER_KERNEL,
                        latent,
                        config.upsampleInitialChannel());
        Conv[] upsample = new Conv[rates.length];
        ResBlock[][] resblocks = new ResBlock[rates.length][blocks];
        int channels = pre.outChannels();
        for (int stage = 0; stage < rates.length; stage++) {
            upsample[stage] =
                    transposedConv(
                            tensors,
                            allocator,
                            "dec.ups." + stage,
                            upsampleKernels[stage],
                            channels,
                            channels / 2);
            channels = upsample[stage].outChannels();
            for (int block = 0; block < blocks; block++) {
                String root = "dec.resblocks." + (stage * blocks + block) + ".";
                Conv[] filter = new Conv[perBlock], project = new Conv[perBlock];
                for (int d = 0; d < perBlock; d++) {
                    filter[d] =
                            conv(
                                    tensors,
                                    allocator,
                                    root + "convs1." + d,
                                    blockKernels[block],
                                    channels,
                                    channels);
                    project[d] =
                            conv(
                                    tensors,
                                    allocator,
                                    root + "convs2." + d,
                                    blockKernels[block],
                                    channels,
                                    channels);
                }
                resblocks[stage][block] =
                        new ResBlock(
                                filter,
                                project,
                                Arrays.copyOfRange(
                                        dilations, block * perBlock, (block + 1) * perBlock));
            }
        }

        require(
                embedding.shape().rank() == 2
                        && embedding.shape().size(0) == config.symbolCount()
                        && embedding.logicalSize() % config.symbolCount() == 0
                        && embedding.logicalSize() / config.symbolCount() >= hidden,
                "enc_p.emb.weight has the wrong shape");
        MemoryView<MemorySegment> embeddingTable =
                dequantToF32(allocator, embedding, "enc_p.emb.weight");
        Conv durationFirst = conv(tensors, allocator, "dp.conv_1", DURATION_KERNEL, hidden);
        int durationWidth = durationFirst.outChannels();
        return new Weights(
                embeddingTable,
                // Row stride, not the hidden width: a quantized row is padded up to a block, and
                // the F32 copy keeps that padding (the dequantization is a flat copy).
                Math.toIntExact(embeddingTable.logicalSize() / config.symbolCount()),
                encoder,
                conv(tensors, allocator, "enc_p.proj", POINTWISE, hidden, 2 * latent),
                new Durations(
                        durationFirst,
                        norm(tensors, allocator, "dp.norm_1", durationWidth),
                        conv(
                                tensors,
                                allocator,
                                "dp.conv_2",
                                DURATION_KERNEL,
                                durationWidth,
                                durationWidth),
                        norm(tensors, allocator, "dp.norm_2", durationWidth),
                        conv(tensors, allocator, "dp.proj", POINTWISE, durationWidth, 1)),
                flow,
                new Decoder(
                        pre,
                        upsample,
                        rates,
                        resblocks,
                        conv(tensors, allocator, "dec.conv_post", VOCODER_KERNEL, channels, 1)));
    }

    /**
     * One convolution, as a model definition would name it: its kernel and input width, with the
     * output width read from the file.
     *
     * <p>Dense files keep the PyTorch shape {@code [kernel, inChannels, outChannels]} (trailing 1s
     * dropped), while quantized ones flatten it to {@code [kernel*inChannels, outChannels]} and pad
     * that row up to a block boundary - so the output width is the only dimension both agree on.
     */
    private static Conv conv(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryArena<MemorySegment> allocator,
            String name,
            int kernel,
            int inChannels) {
        MemoryView<MemorySegment> weight = ModelLoader.require(tensors, name + ".weight");
        return conv(tensors, allocator, weight, name, kernel, inChannels, outChannels(weight));
    }

    private static Conv conv(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryArena<MemorySegment> allocator,
            String name,
            int kernel,
            int inChannels,
            int expectedOutChannels) {
        Conv layer = conv(tensors, allocator, name, kernel, inChannels);
        require(
                layer.outChannels() == expectedOutChannels,
                name + " outputs " + layer.outChannels() + ", expected " + expectedOutChannels);
        return layer;
    }

    /**
     * An upsampling transposed convolution. Dense files shape it {@code [kernel, outChannels,
     * inChannels]} - output channels in the middle, the opposite of a forward convolution - while
     * quantized ones flatten to {@code [kernel*inChannels, outChannels]} like everything else, so
     * either way the output width is the second dimension (the next-to-last of the view's reversed
     * dims).
     */
    private static Conv transposedConv(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryArena<MemorySegment> allocator,
            String name,
            int kernel,
            int inChannels,
            int expectedOutChannels) {
        MemoryView<MemorySegment> weight = ModelLoader.require(tensors, name + ".weight");
        require(weight.shape().rank() >= 2, name + ".weight has the wrong rank");
        int actualOutChannels = Math.toIntExact(weight.shape().size(weight.shape().rank() - 2));
        require(
                actualOutChannels == expectedOutChannels,
                name + " outputs " + actualOutChannels + ", expected " + expectedOutChannels);
        return conv(tensors, allocator, weight, name, kernel, inChannels, actualOutChannels);
    }

    private static Conv conv(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryArena<MemorySegment> allocator,
            MemoryView<MemorySegment> weight,
            String name,
            int kernel,
            int inChannels,
            int outChannels) {
        require(kernel > 0 && inChannels > 0 && outChannels > 0, name + " has invalid dimensions");
        long taps = Math.multiplyExact((long) kernel, inChannels);
        require(taps <= Integer.MAX_VALUE, name + " is too wide");
        require(
                weight.logicalSize() % outChannels == 0,
                name + ".weight cannot be split into output rows");
        long rowStride = weight.logicalSize() / outChannels;
        require(rowStride >= taps && rowStride <= Integer.MAX_VALUE, name + ".weight is too short");
        // The weight keeps FP32 and Q8_0 as stored and dequantizes anything else to F32: a
        // forward then only ever meets the two dtypes the kernels route natively.
        DataType dtype = weight.dataType();
        return new Conv(
                dtype == DataType.FP32 || dtype == DataType.Q8_0
                        ? weight
                        : dequantToF32(allocator, weight, name + ".weight"),
                bias(tensors, allocator, name + ".bias", outChannels),
                kernel,
                inChannels,
                outChannels,
                dtype.elementsPerBlock() > 1);
    }

    /** The conv bias, dequantized to F32 (it is read one scalar at a time), or null when absent. */
    private static MemoryView<MemorySegment> bias(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryArena<MemorySegment> allocator,
            String name,
            int channels) {
        return ModelLoader.find(tensors, name)
                .map(
                        view -> {
                            require(view.logicalSize() == channels, name + " has the wrong length");
                            return dequantToF32(allocator, view, name);
                        })
                .orElse(null);
    }

    private static int outChannels(MemoryView<MemorySegment> weight) {
        int rank = weight.shape().rank();
        // The view's dims are the GGUF's reversed; the output width is the only dimension a dense
        // [kernel, in, out] and a quantized [kernel*in, out] encoding agree on.
        boolean quantized = weight.dataType().elementsPerBlock() > 1;
        return rank > (quantized ? 1 : 2) ? Math.toIntExact(weight.shape().size(0)) : 1;
    }

    private static Norm norm(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryArena<MemorySegment> allocator,
            String name,
            int channels) {
        MemoryView<MemorySegment> gamma = ModelLoader.require(tensors, name + ".gamma");
        MemoryView<MemorySegment> beta = ModelLoader.require(tensors, name + ".beta");
        require(
                gamma.logicalSize() == channels && beta.logicalSize() == channels,
                name + " has the wrong width");
        return new Norm(
                dequantToF32(allocator, gamma, name + ".gamma"),
                dequantToF32(allocator, beta, name + ".beta"),
                channels);
    }

    /**
     * A flat F32 copy of a weight whose elements are read as scalars (embedding rows, relative
     * embeddings, norm parameters, biases) or whose dtype the kernels do not route natively. The
     * copy keeps the stored layout - padded rows stay padded, so strides computed off it match the
     * old port's exactly.
     */
    private static MemoryView<MemorySegment> dequantToF32(
            MemoryArena<MemorySegment> allocator, MemoryView<MemorySegment> view, String name) {
        if (view.dataType() == DataType.FP32) return view;
        int size = Math.toIntExact(view.logicalSize());
        MemoryView<MemorySegment> copy = Views.allocateF32(allocator, size);
        Convert.copyToF32(view, 0, copy, 0, size);
        return copy;
    }

    static Configuration readConfig(GGUF gguf) {
        require(
                "inflect-v2".equals(gguf.getString("general.architecture")),
                "unsupported architecture '" + gguf.getString("general.architecture") + "'");
        int symbols = gguf.getValue(int.class, "inflect.v2.symbol_count");
        int latent = gguf.getValue(int.class, "inflect.v2.inter_channels");
        int hidden = gguf.getValue(int.class, "inflect.v2.hidden_channels");
        int filter = gguf.getValue(int.class, "inflect.v2.filter_channels");
        int heads = gguf.getValue(int.class, "inflect.v2.n_heads");
        int layers = gguf.getValue(int.class, "inflect.v2.n_layers");
        int kernel = gguf.getValue(int.class, "inflect.v2.kernel_size");
        int sampleRate = gguf.getValue(int.class, "inflect.v2.sample_rate");
        int initialChannels = gguf.getValue(int.class, "inflect.v2.upsample_initial_channel");
        int[] blockKernels = gguf.getValue(int[].class, "inflect.v2.resblock_kernel_sizes");
        int[] dilations = gguf.getValue(int[].class, "inflect.v2.resblock_dilation_sizes");
        int[] rates = gguf.getValue(int[].class, "inflect.v2.upsample_rates");
        int[] upsampleKernels = gguf.getValue(int[].class, "inflect.v2.upsample_kernel_sizes");

        require(symbols == Symbols.count(), "symbol table size does not match the model");
        require(
                latent > 0
                        && (latent & 1) == 0
                        && hidden > 0
                        && filter > 0
                        && heads > 0
                        && hidden % heads == 0
                        && layers > 0
                        && kernel > 0
                        && (kernel & 1) == 1
                        && sampleRate > 0
                        && initialChannels > 0,
                "invalid core dimensions");
        require(
                blockKernels.length > 0
                        && dilations.length > 0
                        && dilations.length % blockKernels.length == 0,
                "invalid resblock layout");
        for (int value : blockKernels)
            require(value > 0 && (value & 1) == 1, "invalid resblock kernel");
        for (int value : dilations) require(value > 0, "invalid resblock dilation");
        require(
                rates.length > 0 && rates.length == upsampleKernels.length,
                "invalid upsample layout");
        int channels = initialChannels;
        long maxSamples = MAX_FRAMES;
        for (int i = 0; i < rates.length; i++) {
            int rate = rates[i], upsampleKernel = upsampleKernels[i];
            require(
                    rate > 0
                            && upsampleKernel >= rate
                            && (upsampleKernel - rate) % 2 == 0
                            && (channels & 1) == 0,
                    "invalid upsample stage " + i);
            channels /= 2;
            require(
                    maxSamples <= Integer.MAX_VALUE / rate,
                    "upsample layout exceeds the waveform limit");
            maxSamples *= rate;
        }
        require(
                gguf.getValueOrDefault(boolean.class, "inflect.v2.add_blank", true),
                "the frontend requires blank-interspersed symbols");
        require(
                "leaky_relu".equals(gguf.getStringOrDefault("inflect.v2.activation", "leaky_relu")),
                "unsupported decoder activation");

        return new Configuration(
                symbols,
                latent,
                hidden,
                filter,
                heads,
                layers,
                kernel,
                sampleRate,
                initialChannels,
                blockKernels,
                dilations,
                rates,
                upsampleKernels);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Inflect2: " + message);
    }

    // ── model ─────────────────────────────────────────────────────────────

    public Configuration configuration() {
        return cfg;
    }

    public Weights weights() {
        return w;
    }

    public int sampleRate() {
        return cfg.sampleRate();
    }

    public int tensorCount() {
        return tensorCount;
    }

    public long parameterCount() {
        return parameterCount;
    }

    /** A state that owns its scratch: an internal shared arena freed by {@code close()}. */
    public State newState() {
        MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
        try {
            return new State(arena, arena);
        } catch (RuntimeException | Error e) {
            Arenas.close(arena);
            throw e;
        }
    }

    /**
     * A state whose scratch comes from {@code arena}, BORROWED: the caller owns and frees it, and
     * {@code state.close()} never touches it. Close YOUR arena only after the last synthesis using
     * this state returns - the kernels read raw addresses, so a live read from a closed arena is a
     * crash, not an exception.
     */
    public State newState(MemoryArena<MemorySegment> arena) {
        if (arena == null) throw new IllegalArgumentException("null arena");
        return new State(arena, null);
    }

    /**
     * Synthesize a waveform from blank-interspersed phoneme tokens (see {@link Symbols}).
     *
     * @param lengthScale stretches every predicted duration - 1/speed, so 1.25 speaks slower
     * @param variation scale of the latent noise (0 = deterministic, 0.667 is the reference
     *     default)
     */
    public Media.Audio synthesize(
            State state, int[] tokens, float lengthScale, float variation, long seed) {
        // Holds the state for this synthesis: a concurrent one fails fast, and a close waits for
        // this to return rather than freeing the arena under the kernels. Reentrant, so a caller
        // (InflectTTS.speak) may hold it across a whole multi-chunk utterance.
        Media.Audio result =
                state.exclusively(() -> synthesize0(state, tokens, lengthScale, variation, seed));
        Reference.reachabilityFence(this); // kernels read weights via raw bases; pin `this`
        return result;
    }

    private Media.Audio synthesize0(
            State state, int[] tokens, float lengthScale, float variation, long seed) {
        if (tokens.length == 0) throw new IllegalArgumentException("tokens must not be empty");
        for (int token : tokens)
            if (token < 0 || token >= cfg.symbolCount())
                throw new IllegalArgumentException(
                        "token " + token + " outside [0," + cfg.symbolCount() + ")");
        if (!(lengthScale > 0) || !Float.isFinite(lengthScale))
            throw new IllegalArgumentException(
                    "lengthScale must be finite and > 0: " + lengthScale);
        if (!(variation >= 0 && variation <= 1) || !Float.isFinite(variation))
            throw new IllegalArgumentException(
                    "variation must be finite and in [0, 1]: " + variation);

        state.rewind();
        int tokenCount = tokens.length;

        MemoryView<MemorySegment> encoded = encode(state, tokens);
        // Per-token prior: [mean | logScale] interleaved over 2*interChannels.
        MemoryView<MemorySegment> stats = conv(state, encoded, w.project(), 1, tokenCount);
        int[] repeats = state.takeInts(tokenCount);
        int frames = predictDurations(state, repeats, encoded, tokenCount, lengthScale);

        MemoryView<MemorySegment> prior =
                samplePrior(state, stats, repeats, frames, variation, seed);
        MemoryView<MemorySegment> flowed = flow(state, prior, frames);
        // The vocoder is the bandwidth-bound part and fires thousands of small parallel regions;
        MemoryView<MemorySegment> waveform = decode(state, flowed, frames);

        // The one allocation of the pass: the waveform escapes to the caller as a plain array.
        float[] pcm = new float[waveformSamples(frames)];
        Views.copyToArray(waveform, 0, pcm, 0, pcm.length, "waveform");
        return new Media.Audio(pcm, cfg.sampleRate(), 1);
    }

    // ── encoder ───────────────────────────────────────────────────────────

    /**
     * Embed the tokens and run the transformer. Time-major throughout: {@code x[token][channel]}.
     */
    private MemoryView<MemorySegment> encode(State state, int[] tokens) {
        int hidden = cfg.hiddenChannels(), tokenCount = tokens.length;
        float scale = (float) Math.sqrt(hidden);
        MemoryView<MemorySegment> x = state.take(hidden * tokenCount);
        Views.checkAlive(w.embedding(), "embedding"); // fail-fast on freed weights
        for (int token = 0; token < tokenCount; token++) {
            Convert.copyF32(
                    w.embedding(),
                    (long) tokens[token] * w.embeddingStride(),
                    x,
                    (long) token * hidden,
                    hidden);
        }
        Ops.multiplyInPlace(x, 0, hidden * tokenCount, scale);
        for (EncoderLayer layer : w.encoder()) x = encoderLayer(state, x, layer, tokenCount);
        return x;
    }

    private MemoryView<MemorySegment> encoderLayer(
            State state, MemoryView<MemorySegment> x, EncoderLayer layer, int tokenCount) {
        MemoryView<MemorySegment> attended = attention(state, x, layer, tokenCount);
        x = addNorm(state, x, attended, layer.attentionNorm(), tokenCount);
        MemoryView<MemorySegment> wide = conv(state, x, layer.expand(), 1, tokenCount);
        relu(wide, layer.expand().outChannels() * tokenCount);
        return addNorm(
                state,
                x,
                conv(state, wide, layer.contract(), 1, tokenCount),
                layer.ffnNorm(),
                tokenCount);
    }

    /**
     * Multi-head self-attention with learned relative positions: a key within the layer's window of
     * the query contributes an extra term to both the score and the value. {@code scores} is a heap
     * row: it is written and re-read one query at a time, never crossing a kernel boundary.
     */
    private MemoryView<MemorySegment> attention(
            State state, MemoryView<MemorySegment> x, EncoderLayer layer, int tokenCount) {
        int hidden = layer.query().outChannels(), heads = cfg.nHeads(), headDim = hidden / heads;
        int window = layer.window();
        MemoryView<MemorySegment> q = conv(state, x, layer.query(), 1, tokenCount);
        MemoryView<MemorySegment> k = conv(state, x, layer.key(), 1, tokenCount);
        MemoryView<MemorySegment> v = conv(state, x, layer.value(), 1, tokenCount);
        // `attended` accumulates across heads, so it starts zeroed.
        MemoryView<MemorySegment> attended = state.takeZeroed(hidden * tokenCount);
        float[] scores = state.takeWeights(tokenCount);
        float scale = 1f / (float) Math.sqrt(headDim);

        for (int head = 0; head < heads; head++) {
            int channel = head * headDim;
            for (int query = 0; query < tokenCount; query++) {
                long queryRow = (long) query * hidden + channel;
                float max = Float.NEGATIVE_INFINITY;
                for (int key = 0; key < tokenCount; key++) {
                    float score = Ops.dot(q, queryRow, k, (long) key * hidden + channel, headDim);
                    int distance = key - query;
                    if (Math.abs(distance) <= window)
                        score +=
                                Ops.dot(
                                        q,
                                        queryRow,
                                        layer.relativeKeys(),
                                        (long) (distance + window) * headDim,
                                        headDim);
                    score *= scale;
                    scores[key] = score;
                    max = Math.max(max, score);
                }
                float total = 0;
                for (int key = 0; key < tokenCount; key++) {
                    float weight = (float) Math.exp(scores[key] - max);
                    scores[key] = weight;
                    total += weight;
                }
                for (int key = 0; key < tokenCount; key++) {
                    float weight = scores[key] / total;
                    Ops.saxpyInPlace(
                            attended, queryRow, v, (long) key * hidden + channel, headDim, weight);
                    int distance = key - query;
                    if (Math.abs(distance) <= window)
                        Ops.saxpyInPlace(
                                attended,
                                queryRow,
                                layer.relativeValues(),
                                (long) (distance + window) * headDim,
                                headDim,
                                weight);
                }
            }
        }
        return conv(state, attended, layer.output(), 1, tokenCount);
    }

    // ── durations and prior ───────────────────────────────────────────────

    /**
     * Predict how many frames each token lasts, filling {@code repeats} and returning the total. A
     * degenerate prediction still yields one frame rather than an empty waveform.
     */
    private int predictDurations(
            State state,
            int[] repeats,
            MemoryView<MemorySegment> encoded,
            int tokenCount,
            float lengthScale) {
        Durations dp = w.durations();
        int width = dp.first().outChannels();
        MemoryView<MemorySegment> h = conv(state, encoded, dp.first(), 1, tokenCount);
        relu(h, width * tokenCount);
        h = norm(state, h, dp.firstNorm(), tokenCount);
        h = conv(state, h, dp.second(), 1, tokenCount);
        relu(h, width * tokenCount);
        h = norm(state, h, dp.secondNorm(), tokenCount);
        MemoryView<MemorySegment> logDuration = conv(state, h, dp.project(), 1, tokenCount);
        float[] logFrames = state.takeWeights(tokenCount);
        Views.copyToArray(logDuration, 0, logFrames, 0, tokenCount, "logDuration");

        long total = 0;
        for (int token = 0; token < tokenCount; token++) {
            double frames = Math.ceil(Math.exp(logFrames[token]) * lengthScale);
            if (!Double.isFinite(frames) || frames > MAX_FRAMES)
                throw new IllegalArgumentException(
                        "token " + token + " needs " + frames + " frames");
            repeats[token] = (int) Math.max(0, frames);
            total += repeats[token];
        }
        if (total > MAX_FRAMES)
            throw new IllegalArgumentException(
                    "chunk needs " + total + " frames, over the " + MAX_FRAMES + " ceiling");
        return (int) Math.max(total, 1);
    }

    /**
     * Repeat each token's latent for its duration and add noise. The Gaussian is drawn
     * channel-major because that order is what the seed means - a frame-major walk would give the
     * same distribution but different audio. The last token also covers any frames left over. The
     * draw runs on the heap and lands in the view with one bulk copy: per-element checked writes
     * would add a check per sample for nothing.
     */
    private MemoryView<MemorySegment> samplePrior(
            State state,
            MemoryView<MemorySegment> stats,
            int[] repeats,
            int frames,
            float variation,
            long seed) {
        int latent = cfg.interChannels(), tokenCount = repeats.length;
        Random random = state.random;
        random.setSeed(seed); // same sequence as a fresh Random(seed)
        float[] statsFlat = state.takeWeights(tokenCount * 2 * latent);
        Views.copyToArray(stats, 0, statsFlat, 0, tokenCount * 2 * latent, "stats");
        float[] samples = state.takeWeights(latent * frames);
        for (int channel = 0; channel < latent; channel++) {
            int frame = 0;
            for (int token = 0; token < tokenCount && frame < frames; token++) {
                int row = token * 2 * latent;
                float mean = statsFlat[row + channel];
                float deviation = (float) Math.exp(statsFlat[row + latent + channel]);
                int last =
                        token == tokenCount - 1 ? frames : Math.min(frames, frame + repeats[token]);
                for (; frame < last; frame++)
                    samples[frame * latent + channel] =
                            mean + (float) random.nextGaussian() * deviation * variation;
            }
        }
        MemoryView<MemorySegment> prior = state.take(latent * frames);
        Views.copyFromArray(prior, 0, samples, 0, latent * frames, "prior");
        return prior;
    }

    // ── flow ──────────────────────────────────────────────────────────────

    /** Invert the coupling stack: last coupling first, each preceded by a channel flip. */
    private MemoryView<MemorySegment> flow(State state, MemoryView<MemorySegment> z, int frames) {
        int channels = cfg.interChannels();
        for (int i = w.flow().length - 1; i >= 0; i--) {
            Ops.reverseChannelsInPlace(z, channels, frames);
            MemoryView<MemorySegment> next =
                    state.take(channels * frames); // outlives the scope below
            try (var scope = state.scope()) {
                coupling(state, z, next, w.flow()[i], frames);
            }
            z = next;
        }
        return z;
    }

    /**
     * Reversed affine coupling: a WaveNet over the first half of the channels predicts a shift,
     * which is subtracted from the second half; the first half passes through untouched.
     */
    private void coupling(
            State state,
            MemoryView<MemorySegment> z,
            MemoryView<MemorySegment> out,
            Coupling layer,
            int frames) {
        int channels = cfg.interChannels(), half = channels / 2;
        int hidden = layer.pre().outChannels();

        MemoryView<MemorySegment> first = state.take(half * frames);
        for (int frame = 0; frame < frames; frame++)
            Convert.copyF32(z, (long) frame * channels, first, (long) frame * half, half);

        MemoryView<MemorySegment> h = conv(state, first, layer.pre(), 1, frames);
        MemoryView<MemorySegment> skip =
                state.takeZeroed(hidden * frames); // accumulates over the layers
        for (int i = 0; i < layer.gates().length; i++) {
            MemoryView<MemorySegment> gates = conv(state, h, layer.gates()[i], 1, frames);
            // Each step's channels are contiguous, and its two halves are one hidden apart.
            MemoryView<MemorySegment> activated = state.take(hidden * frames);
            for (int frame = 0; frame < frames; frame++)
                Activations.tanhSigmoidGate(
                        activated,
                        (long) frame * hidden,
                        gates,
                        (long) frame * 2 * hidden,
                        gates,
                        (long) frame * 2 * hidden + hidden,
                        hidden);

            Conv projection = layer.residualSkip()[i];
            MemoryView<MemorySegment> projected = conv(state, activated, projection, 1, frames);
            int width = projection.outChannels();
            // Every layer but the last splits its projection into a residual and a skip half.
            if (width == 2 * hidden)
                for (int frame = 0; frame < frames; frame++) {
                    Ops.addInPlace(
                            h, (long) frame * hidden, projected, (long) frame * width, hidden);
                    Ops.addInPlace(
                            skip,
                            (long) frame * hidden,
                            projected,
                            (long) frame * width + hidden,
                            hidden);
                }
            else Ops.addInPlace(skip, 0, projected, 0, hidden * frames);
        }

        MemoryView<MemorySegment> shift = conv(state, skip, layer.post(), 1, frames);
        Convert.copyF32(z, 0, out, 0, channels * frames);
        for (int frame = 0; frame < frames; frame++)
            Ops.saxpyInPlace(
                    out, (long) frame * channels + half, shift, (long) frame * half, half, -1f);
    }

    // ── decoder ───────────────────────────────────────────────────────────

    /**
     * HiFi-GAN vocoder. It runs <b>channel-major</b> - {@code rows[channel][time]}, one contiguous
     * row per channel - unlike the encoder's time-major layout. The shapes are why: the decoder is
     * 12 to 96 channels wide and up to ~110k samples long, so time is the only axis worth
     * vectorizing, and with it contiguous a convolution is an FMA sweep per tap rather than an
     * im2col matrix K times the size of its input.
     */
    private MemoryView<MemorySegment> decode(State state, MemoryView<MemorySegment> z, int frames) {
        Decoder decoder = w.decoder();
        MemoryView<MemorySegment> rows = state.take(cfg.interChannels() * frames);
        Ops.transposeCopy(z, frames, cfg.interChannels(), rows);

        int channels = decoder.pre().outChannels(), time = frames;
        MemoryView<MemorySegment> x = state.take(channels * time);
        convRows(state, rows, x, decoder.pre(), 1, time);

        for (int stage = 0; stage < decoder.upsample().length; stage++) {
            Conv up = decoder.upsample()[stage];
            int stride = decoder.rates()[stage];
            int upTime = upsampledLength(time, up.kernel(), stride);
            int size = up.outChannels() * upTime;

            leaky(x, channels * time, LEAKY);
            // Taken before the scope: the stage's output outlives the resblocks' scratch.
            MemoryView<MemorySegment> upsampled = state.take(size);
            upsampleRows(state, x, upsampled, up, stride, time, upTime);

            // Every resblock reads the stage's output; their results are averaged back into it.
            ResBlock[] blocks = decoder.resblocks()[stage];
            try (var scope = state.scope()) {
                MemoryView<MemorySegment> sum = state.takeZeroed(size);
                for (ResBlock block : blocks) resblock(state, upsampled, sum, block, upTime);
                Convert.copyF32(sum, 0, upsampled, 0, size);
                Ops.divideInPlace(upsampled, 0, size, blocks.length);
            }

            x = upsampled;
            channels = up.outChannels();
            time = upTime;
        }

        leaky(x, channels * time, FINAL_LEAKY);
        // A single output channel, so channel-major already IS the waveform.
        MemoryView<MemorySegment> waveform = state.take(time);
        convRows(state, x, waveform, decoder.post(), 1, time);
        // one tanh per audio sample: scalar Math.tanh costs ~15ns each in the native image;
        // the fused FastMath pass is ~0.4ns (contract in TanhAccuracyTest - abs error ~6e-8,
        // below 24-bit audio's LSB)
        Ops.tanhInPlace(waveform, 0, time);
        return waveform;
    }

    /** One resblock, its result added into {@code sum}. {@code x} is read, not modified. */
    private void resblock(
            State state,
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> sum,
            ResBlock block,
            int time) {
        int size = block.filter()[0].outChannels() * time;
        try (var scope = state.scope()) {
            MemoryView<MemorySegment> value = state.take(size);
            MemoryView<MemorySegment> activated = state.take(size);
            MemoryView<MemorySegment> filtered = state.take(size);
            Convert.copyF32(x, 0, value, 0, size);
            for (int d = 0; d < block.dilations().length; d++) {
                Convert.copyF32(value, 0, activated, 0, size);
                leaky(activated, size, LEAKY);
                convRows(state, activated, filtered, block.filter()[d], block.dilations()[d], time);
                leaky(filtered, size, LEAKY);
                convRows(state, filtered, activated, block.project()[d], 1, time);
                Ops.addInPlace(value, 0, activated, 0, size);
            }
            Ops.addInPlace(sum, 0, value, 0, size);
        }
    }

    /** Samples the decoder produces from {@code frames} - the upsampling recurrence. */
    private int waveformSamples(int frames) {
        Decoder decoder = w.decoder();
        int time = frames;
        for (int stage = 0; stage < decoder.upsample().length; stage++)
            time =
                    upsampledLength(
                            time, decoder.upsample()[stage].kernel(), decoder.rates()[stage]);
        return time;
    }

    private static int upsampledLength(int time, int kernel, int stride) {
        return (time - 1) * stride + kernel - 2 * ((kernel - stride) / 2);
    }

    // ── convolutions ──────────────────────────────────────────────────────

    /**
     * Time-major convolution, {@code [time][channel]} in and out: gather each output step's window
     * into a row (im2col) and let the gemm do the multiply. Right for the encoder's shapes - few
     * steps, many channels.
     */
    private MemoryView<MemorySegment> conv(
            State state, MemoryView<MemorySegment> in, Conv layer, int dilation, int time) {
        int outChannels = layer.outChannels(), rowStride = layer.rowStride();

        // A 1x1 convolution's input already IS the matrix the gemm wants, one row per step - so
        // most of the encoder and the whole flow skip the gather. The exception is a quantized
        // weight whose rows are padded past their taps: then the rows need the padding zeros.
        MemoryView<MemorySegment> matrix =
                layer.kernel() == 1 && rowStride == layer.inChannels()
                        ? in
                        : gather(state, in, layer, dilation, time);

        MemoryView<MemorySegment> product = state.product(outChannels * time);
        MatMul.gemm(
                layer.weight(),
                matrix,
                rowStride,
                product,
                outChannels,
                outChannels,
                time,
                rowStride);

        MemoryView<MemorySegment> out = state.take(outChannels * time);
        Convert.copyF32(product, 0, out, 0, outChannels * time);
        if (layer.bias() != null) Ops.addRowBiasInPlace(out, 0, layer.bias(), 0, time, outChannels);
        return out;
    }

    /**
     * im2col: each output step's window laid out as one row, in the gemm's native staging. Zeroed
     * first, because the gather writes only the taps that fall inside the sequence and relies on
     * zeros for the padding either end - and a padded weight row is wider than it fills. The walk
     * runs on the heap (one bulk read of {@code in}, one bulk write into {@code columns}): the
     * element order is a permutation, so checked per-element access would buy nothing.
     */
    private MemoryView<MemorySegment> gather(
            State state, MemoryView<MemorySegment> in, Conv layer, int dilation, int time) {
        int kernel = layer.kernel(), inChannels = layer.inChannels();
        int rowStride = layer.rowStride(), pad = ((kernel - 1) * dilation) / 2;

        float[] input = state.takeWeights(time * inChannels);
        Views.copyToArray(in, 0, input, 0, time * inChannels, "in");
        float[] rows = state.takeWeights(time * rowStride);
        Arrays.fill(rows, 0, time * rowStride, 0f);
        for (int t = 0; t < time; t++)
            for (int k = 0; k < kernel; k++) {
                int source = t + k * dilation - pad;
                if (source < 0 || source >= time) continue;
                for (int c = 0; c < inChannels; c++)
                    rows[t * rowStride + k + c * kernel] = input[source * inChannels + c];
            }
        MemoryView<MemorySegment> columns = state.columns(time * rowStride);
        Views.copyFromArray(columns, 0, rows, 0, time * rowStride, "columns");
        return columns;
    }

    /**
     * Channel-major convolution: the layer's taps dequantized once, then {@link
     * Convolutions#conv1dRows}, which owns the tiling, the fan-out and the vector accumulation.
     */
    private void convRows(
            State state,
            MemoryView<MemorySegment> in,
            MemoryView<MemorySegment> out,
            Conv layer,
            int dilation,
            int time) {
        try (var scope = state.scope()) {
            Convolutions.conv1dRows(
                    in,
                    layer.inChannels(),
                    out,
                    layer.outChannels(),
                    time,
                    layer.kernel(),
                    dilation,
                    dequantize(state, layer),
                    layer.bias());
        }
    }

    /**
     * Channel-major transposed convolution - the decoder's upsampling step.
     *
     * <p>Upsampling by {@code stride} is {@code stride} independent convolutions, one per output
     * phase: among outputs {@code op = j*stride + phase} only taps with {@code k ≡ phase + pad (mod
     * stride)} contribute, and {@code j} then walks the input contiguously. Each phase is built in
     * its channel's slice of scratch and scattered into the row once, which keeps the sweeps
     * vectorized where a strided write would not be.
     */
    private void upsampleRows(
            State state,
            MemoryView<MemorySegment> in,
            MemoryView<MemorySegment> out,
            Conv layer,
            int stride,
            int time,
            int outTime) {
        int kernel = layer.kernel(), inChannels = layer.inChannels();
        int outChannels = layer.outChannels(), taps = layer.taps();
        int pad = (kernel - stride) / 2;
        int phaseLength = (outTime + stride - 1) / stride;

        try (var scope = state.scope()) {
            float[] weights = dequantizeTransposed(state, layer);
            // Hoisted out of the parallel region: a checked scalar read per channel would touch
            // the weights arena from a worker thread (a confined arena fails loudly there).
            float[] biases = layer.bias() == null ? null : state.takeWeights(outChannels);
            if (biases != null) Views.copyToArray(layer.bias(), 0, biases, 0, outChannels, "bias");
            // One phase buffer per output channel: built concurrently, scattered as they finish.
            MemoryView<MemorySegment> phases = state.take(outChannels * phaseLength);

            Parallel.forLoop(
                    0,
                    outChannels,
                    channel -> {
                        long phase = (long) channel * phaseLength;
                        long outRow = (long) channel * outTime;
                        for (int p = 0; p < stride; p++) {
                            int count = (outTime - p + stride - 1) / stride;
                            if (count <= 0) continue;
                            Ops.fillInPlace(
                                    phases, phase, count, biases == null ? 0f : biases[channel]);
                            for (int k = Math.floorMod(p + pad, stride); k < kernel; k += stride) {
                                int shift = (p + pad - k) / stride; // exact, by the choice of k
                                int start = Math.max(0, -shift);
                                int end = Math.min(count, time - shift);
                                if (end <= start) continue;
                                for (int ic = 0; ic < inChannels; ic++)
                                    Ops.saxpyInPlace(
                                            phases,
                                            phase + start,
                                            in,
                                            (long) ic * time + start + shift,
                                            end - start,
                                            weights[channel * taps + k * inChannels + ic]);
                            }
                            Ops.copyStrided(out, outRow + p, stride, phases, phase, count);
                        }
                    });
        }
    }

    /** One row of taps per output channel, as stored. */
    private float[] dequantize(State state, Conv layer) {
        int taps = layer.taps(), rowStride = layer.rowStride();
        int size = layer.outChannels() * taps;
        MemoryView<MemorySegment> staging = state.taps(size);
        for (int oc = 0; oc < layer.outChannels(); oc++)
            Convert.copyToF32(
                    layer.weight(), (long) oc * rowStride, staging, (long) oc * taps, taps);
        float[] weights = state.takeWeights(size);
        Views.copyToArray(staging, 0, weights, 0, size, "taps");
        return weights;
    }

    /**
     * The same for a transposed convolution. Its dense encoding interleaves output channels, so a
     * row is gathered element by element; quantized files are already repacked per output channel.
     */
    private float[] dequantizeTransposed(State state, Conv layer) {
        if (layer.quantizedLayout()) return dequantize(state, layer);
        int kernel = layer.kernel(), inChannels = layer.inChannels();
        int outChannels = layer.outChannels(), taps = layer.taps();
        float[] stored = state.takeWeights(outChannels * taps);
        Views.copyToArray(layer.weight(), 0, stored, 0, outChannels * taps, "weight");
        float[] weights = state.takeWeights(outChannels * taps);
        for (int oc = 0; oc < outChannels; oc++)
            for (int k = 0; k < kernel; k++)
                for (int ic = 0; ic < inChannels; ic++)
                    weights[oc * taps + k * inChannels + ic] =
                            stored[k + oc * kernel + ic * kernel * outChannels];
        return weights;
    }

    // ── small operations ──────────────────────────────────────────────────

    private MemoryView<MemorySegment> addNorm(
            State state,
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> y,
            Norm layer,
            int time) {
        int size = layer.channels() * time;
        MemoryView<MemorySegment> sum = state.take(size);
        Convert.copyF32(x, 0, sum, 0, size);
        Ops.addInPlace(sum, 0, y, 0, size);
        return norm(state, sum, layer, time);
    }

    private MemoryView<MemorySegment> norm(
            State state, MemoryView<MemorySegment> x, Norm layer, int time) {
        MemoryView<MemorySegment> out = state.take(layer.channels() * time);
        Norms.layerNorm(out, x, layer.gamma(), layer.beta(), layer.channels(), time, 1e-5f);
        return out;
    }

    private static void relu(MemoryView<MemorySegment> x, int size) {
        Ops.clampInPlace(x, 0, size, 0f, Float.MAX_VALUE);
    }

    private static void leaky(MemoryView<MemorySegment> x, int size, float slope) {
        Ops.leakyReluInPlace(x, 0, size, slope);
    }

    // ── state ─────────────────────────────────────────────────────────────

    /**
     * Reusable scratch for one synthesis at a time - the forward pass allocates nothing once a
     * state has been warmed up.
     *
     * <p>Buffers are handed out in stack order and returned by closing a {@code Scope}; each grows
     * to the largest size this state has ever been asked for, so only the first call (or a longer
     * text than before) allocates.
     *
     * <p>Not thread-safe: one state per concurrent synthesis. A state is cheap to create and holds
     * on the order of the largest waveform it has produced.
     *
     * <p>Scratch comes from an arena on the "who provides it owns it" contract: ownership is
     * decided at construction and never mutated, so {@code owned} is final and there is no Cleaner
     * - which is also why this state can live in a native image heap. A dropped unclosed state
     * leaks its arena until exit; {@code -Djinfer.leakDetection} names the line that dropped it.
     */
    public static final class State extends RuntimeState {

        private final MemoryArena<MemorySegment>
                owned; // null when borrowed: closing this state must not free it
        private final MemoryArena<MemorySegment> allocator;
        private final Runnable disarm;

        State(MemoryArena<MemorySegment> arena, MemoryArena<MemorySegment> owned) {
            this.owned = owned;
            this.allocator = arena;
            // armed last: nothing above can throw, and a ctor throw must not read as a leak
            this.disarm = LeakWatch.arm(this, "Inflect2.State");
        }

        @Override
        protected void checkResourcesAlive() {
            if (!allocator.isAlive())
                throw new IllegalStateException("the speech state's arena has been closed");
        }

        @Override
        protected void releaseResources() {
            disarm.run();
            if (owned == null) return;
            Arenas.close(owned);
        }

        private MemoryView<MemorySegment> columns, product, taps;

        @SuppressWarnings("unchecked")
        private MemoryView<MemorySegment>[] buffers = new MemoryView[64];

        private float[][] weightBuffers = new float[8][];
        private int[][] intBuffers = new int[8][];
        private int top, weightTop, intTop, depth;
        private final Random random = new Random();

        /** Start of a fresh pass: every buffer becomes available again. */
        void rewind() {
            top = 0;
            weightTop = 0;
            intTop = 0;
            depth = 0;
        }

        /**
         * A scope of the buffer stack: every buffer taken inside is handed back when it closes.
         * Scopes nest and are pooled, so this costs nothing per pass. A value that must outlive the
         * scope is taken before it opens.
         */
        final class Scope implements AutoCloseable {
            private int buffers, weights;

            @Override
            public void close() {
                top = buffers;
                weightTop = weights;
                depth--;
            }
        }

        private final Scope[] scopes = new Scope[16]; // deeper than the pipeline ever nests

        {
            for (int i = 0; i < scopes.length; i++) scopes[i] = new Scope();
        }

        Scope scope() {
            Scope scope = scopes[depth++];
            scope.buffers = top;
            scope.weights = weightTop;
            return scope;
        }

        /** A view of at least {@code size} floats; contents undefined. */
        MemoryView<MemorySegment> take(int size) {
            if (top == buffers.length) buffers = Arrays.copyOf(buffers, buffers.length * 2);
            MemoryView<MemorySegment> buffer = buffers[top];
            if (buffer == null || buffer.shape().size() < size)
                buffer = buffers[top] = Views.allocateF32(allocator, size);
            top++;
            return buffer;
        }

        /** A view of at least {@code size} floats, zeroed - for accumulators. */
        MemoryView<MemorySegment> takeZeroed(int size) {
            MemoryView<MemorySegment> buffer = take(size);
            Ops.fillInPlace(buffer, 0, size, 0f);
            return buffer;
        }

        /**
         * Heap staging for one convolution's dequantized weights, and for the short heap walks
         * (im2col, prior sampling, transposed taps) that land in a view with one bulk copy.
         */
        float[] takeWeights(int size) {
            if (weightTop == weightBuffers.length)
                weightBuffers = Arrays.copyOf(weightBuffers, weightBuffers.length * 2);
            float[] buffer = weightBuffers[weightTop];
            if (buffer == null || buffer.length < size)
                buffer = weightBuffers[weightTop] = new float[size];
            weightTop++;
            return buffer;
        }

        int[] takeInts(int size) {
            if (intTop == intBuffers.length)
                intBuffers = Arrays.copyOf(intBuffers, intBuffers.length * 2);
            int[] buffer = intBuffers[intTop];
            if (buffer == null || buffer.length < size) buffer = intBuffers[intTop] = new int[size];
            intTop++;
            return buffer;
        }

        /** The gemm's im2col staging. */
        MemoryView<MemorySegment> columns(int size) {
            return columns = grow(columns, size);
        }

        /** The gemm's output. */
        MemoryView<MemorySegment> product(int size) {
            return product = grow(product, size);
        }

        /** The conv1dRows tap staging: one dequantized row per output channel. */
        MemoryView<MemorySegment> taps(int size) {
            return taps = grow(taps, size);
        }

        /** {@code buffer}, regrown to at least {@code size} floats when it falls short. */
        private MemoryView<MemorySegment> grow(MemoryView<MemorySegment> buffer, int size) {
            return buffer != null && buffer.shape().size() >= size
                    ? buffer
                    : Views.allocateF32(allocator, size);
        }
    }
}
