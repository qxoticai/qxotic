// Inflect2 — VITS-family text-to-waveform model (Nano 3.97M, Micro 9.36M; F16/Q8_0/Q4_0 GGUF).
//
//   Inflect2 model = Inflect2.load(Path.of("model.gguf"));
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
import com.qxotic.jinfer.Activations;
import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.Convolutions;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.Norms;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.kernels.GGMLTensorEntry;
import com.qxotic.jinfer.kernels.ModelLoader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

public final class Inflect2 {

    /** HiFi-GAN leaky-ReLU slope, and torch's default slope for the activation before conv_post. */
    private static final float LEAKY = 0.1f, FINAL_LEAKY = 0.01f;

    /**
     * Frame ceiling per call — a runaway log-duration must fail, not exhaust memory. A DoS BOUND,
     * not a modelling constant: it caps one chunk at ~43 s of audio and every buffer sized off it,
     * and it is what stops adversarial text (or a tiny speed) from turning one request into a
     * multi-gigabyte allocation. Raising it raises the worst case a single request can cost.
     */
    private static final int MAX_FRAMES = 4000;

    // Kernel sizes the GGUF metadata does not carry — they are part of the architecture, exactly as
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
            int[] upsampleKernelSizes)
            implements Config {

        /** The phoneme symbol table this model consumes — its token space. */
        @Override
        public int vocabularySize() {
            return symbolCount;
        }

        /** The frame ceiling: a runaway log-duration must fail, not exhaust memory. */
        @Override
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

    /** One 1-D convolution. {@code bias} is null where folded weight norm left none. */
    public record Conv(
            FloatTensor weight, FloatTensor bias, int kernel, int inChannels, int outChannels) {

        /** Taps per output channel. */
        int taps() {
            return kernel * inChannels;
        }

        /** Elements per stored row — more than {@link #taps} when rows are padded to a block. */
        int rowStride() {
            return Math.toIntExact(weight.size() / outChannels);
        }

        float bias(int outChannel) {
            return bias == null ? 0f : bias.getFloat(outChannel);
        }
    }

    /** LayerNorm parameters over {@code channels} contiguous channels. */
    public record Norm(FloatTensor gamma, FloatTensor beta, int channels) {}

    /** One encoder block: relative-position self-attention, then a convolutional feed-forward. */
    public record EncoderLayer(
            Conv query,
            Conv key,
            Conv value,
            Conv output,
            FloatTensor relativeKeys,
            FloatTensor relativeValues,
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
            FloatTensor embedding,
            int embeddingStride,
            EncoderLayer[] encoder,
            Conv project,
            Durations durations,
            Coupling[] flow,
            Decoder decoder) {}

    private final Configuration cfg;
    private final Weights w;

    /** What the file held, for reporting — the weights themselves are resolved into {@link #w}. */
    private final int tensorCount;

    private final long parameterCount;

    private Inflect2(Configuration cfg, Weights weights, int tensorCount, long parameterCount) {
        this.cfg = cfg;
        this.w = weights;
        this.tensorCount = tensorCount;
        this.parameterCount = parameterCount;
    }

    // ── loading ───────────────────────────────────────────────────────────

    public static Inflect2 load(Path path) throws IOException {
        return load(path, Arena.ofAuto());
    }

    /**
     * Weights map into {@code arena}, and whoever provides it owns its lifetime — the same contract
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
        return load(gguf, ModelLoader.loadTensors(channel, gguf, 0, arena));
    }

    /**
     * Load from a GGUF stored in a ZIP overlay appended to the running executable — the tensor data
     * is mapped straight out of the executable, with no temp file and no copy.
     */
    public static Inflect2 loadSelfArchive(String entryName) throws IOException {
        return loadSelfArchive(entryName, Arena.ofAuto());
    }

    /** As {@link #loadSelfArchive(String)}, with the weights mapped into {@code arena}. */
    public static Inflect2 loadSelfArchive(String entryName, Arena arena) throws IOException {
        try (SelfArchive archive = SelfArchive.open()) {
            SelfArchive.Entry entry = archive.entry(entryName);
            // The header is small (< 64 KB even for Inflect2's 302 tensors).
            byte[] header = archive.readAt(entry.offset(), Math.min(entry.size(), 1 << 16));
            GGUF gguf = GGUF.read(Channels.newChannel(new ByteArrayInputStream(header)));
            return load(
                    gguf, ModelLoader.loadTensors(archive.channel(), gguf, entry.offset(), arena));
        }
    }

    private static Inflect2 load(GGUF gguf, Map<String, GGMLTensorEntry> tensors) {
        Configuration config = readConfig(gguf);
        long parameters = 0;
        for (GGMLTensorEntry tensor : tensors.values())
            parameters += FloatTensor.numberOfElementsLong(tensor.shape());
        return new Inflect2(config, loadWeights(tensors, config), tensors.size(), parameters);
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
        int hidden = config.hiddenChannels(), latent = config.interChannels();
        GGMLTensorEntry embedding = require(tensors, "enc_p.emb.weight");

        EncoderLayer[] encoder = new EncoderLayer[config.nLayers()];
        for (int i = 0; i < encoder.length; i++) {
            String attention = "enc_p.encoder.attn_layers." + i + ".";
            String ffn = "enc_p.encoder.ffn_layers." + i + ".";
            GGMLTensorEntry relativeKeys = require(tensors, attention + "emb_rel_k");
            encoder[i] =
                    new EncoderLayer(
                            conv(tensors, attention + "conv_q", POINTWISE, hidden),
                            conv(tensors, attention + "conv_k", POINTWISE, hidden),
                            conv(tensors, attention + "conv_v", POINTWISE, hidden),
                            conv(tensors, attention + "conv_o", POINTWISE, hidden),
                            ModelLoader.loadQuantized(relativeKeys),
                            ModelLoader.loadQuantized(require(tensors, attention + "emb_rel_v")),
                            // [headChannels, 2*window+1]: keys this far either side get an
                            // embedding
                            (relativeKeys.shape()[1] - 1) / 2,
                            norm(tensors, "enc_p.encoder.norm_layers_1." + i),
                            conv(tensors, ffn + "conv_1", config.kernelSize(), hidden),
                            conv(
                                    tensors,
                                    ffn + "conv_2",
                                    config.kernelSize(),
                                    config.filterChannels()),
                            norm(tensors, "enc_p.encoder.norm_layers_2." + i));
        }

        // The file interleaves the couplings with the flips applied between them, hence the 2*i.
        int couplings = 0;
        while (tensors.containsKey("flow.flows." + (2 * couplings) + ".pre.weight")) couplings++;
        if (couplings == 0) throw new IllegalArgumentException("no flow coupling layers in model");
        Coupling[] flow = new Coupling[couplings];
        for (int i = 0; i < couplings; i++) {
            String root = "flow.flows." + (2 * i) + ".";
            Conv pre = conv(tensors, root + "pre", POINTWISE, latent / 2);
            int wide = pre.outChannels();
            int layers = 0;
            while (tensors.containsKey(root + "enc.in_layers." + layers + ".weight")) layers++;
            Conv[] gates = new Conv[layers], residualSkip = new Conv[layers];
            for (int layer = 0; layer < layers; layer++) {
                gates[layer] = conv(tensors, root + "enc.in_layers." + layer, WAVENET_KERNEL, wide);
                residualSkip[layer] =
                        conv(tensors, root + "enc.res_skip_layers." + layer, POINTWISE, wide);
            }
            flow[i] =
                    new Coupling(
                            pre,
                            gates,
                            residualSkip,
                            conv(tensors, root + "post", POINTWISE, wide));
        }

        int[] rates = config.upsampleRates(), dilations = config.resblockDilationSizes();
        int[] upsampleKernels = config.upsampleKernelSizes();
        int[] blockKernels = config.resblockKernelSizes();
        int blocks = blockKernels.length, perBlock = dilations.length / blocks;
        Conv pre = conv(tensors, "dec.conv_pre", VOCODER_KERNEL, latent);
        Conv[] upsample = new Conv[rates.length];
        ResBlock[][] resblocks = new ResBlock[rates.length][blocks];
        int channels = pre.outChannels();
        for (int stage = 0; stage < rates.length; stage++) {
            upsample[stage] =
                    transposedConv(tensors, "dec.ups." + stage, upsampleKernels[stage], channels);
            channels = upsample[stage].outChannels();
            for (int block = 0; block < blocks; block++) {
                String root = "dec.resblocks." + (stage * blocks + block) + ".";
                Conv[] filter = new Conv[perBlock], project = new Conv[perBlock];
                for (int d = 0; d < perBlock; d++) {
                    filter[d] = conv(tensors, root + "convs1." + d, blockKernels[block], channels);
                    project[d] = conv(tensors, root + "convs2." + d, blockKernels[block], channels);
                }
                resblocks[stage][block] =
                        new ResBlock(
                                filter,
                                project,
                                Arrays.copyOfRange(
                                        dilations, block * perBlock, (block + 1) * perBlock));
            }
        }

        FloatTensor embeddingTable = ModelLoader.loadQuantized(embedding);
        return new Weights(
                embeddingTable,
                // Row stride, not the hidden width: a quantized row is padded up to a block.
                Math.toIntExact(embeddingTable.size() / config.symbolCount()),
                encoder,
                conv(tensors, "enc_p.proj", POINTWISE, hidden),
                new Durations(
                        conv(tensors, "dp.conv_1", DURATION_KERNEL, hidden),
                        norm(tensors, "dp.norm_1"),
                        conv(
                                tensors,
                                "dp.conv_2",
                                DURATION_KERNEL,
                                outChannels(require(tensors, "dp.conv_1.weight"))),
                        norm(tensors, "dp.norm_2"),
                        conv(
                                tensors,
                                "dp.proj",
                                POINTWISE,
                                outChannels(require(tensors, "dp.conv_2.weight")))),
                flow,
                new Decoder(
                        pre,
                        upsample,
                        rates,
                        resblocks,
                        conv(tensors, "dec.conv_post", VOCODER_KERNEL, channels)));
    }

    /**
     * One convolution, as a model definition would name it: its kernel and input width, with the
     * output width read from the file.
     *
     * <p>Dense files keep the PyTorch shape {@code [kernel, inChannels, outChannels]} (trailing 1s
     * dropped), while quantized ones flatten it to {@code [kernel*inChannels, outChannels]} and pad
     * that row up to a block boundary — so the output width is the only dimension both agree on.
     */
    private static Conv conv(
            Map<String, GGMLTensorEntry> tensors, String name, int kernel, int inChannels) {
        GGMLTensorEntry entry = require(tensors, name + ".weight");
        return new Conv(
                ModelLoader.loadQuantized(entry),
                ModelLoader.quantOrNull(tensors, name + ".bias"),
                kernel,
                inChannels,
                outChannels(entry));
    }

    private static int outChannels(GGMLTensorEntry entry) {
        int[] shape = entry.shape();
        return entry.ggmlType().isQuantized()
                ? (shape.length > 1 ? shape[1] : 1)
                : (shape.length > 2 ? shape[2] : 1);
    }

    /**
     * An upsampling transposed convolution. Dense files shape it {@code [kernel, outChannels,
     * inChannels]} — output channels in the middle, the opposite of a forward convolution — while
     * quantized ones flatten to {@code [kernel*inChannels, outChannels]} like everything else, so
     * either way the output width is the second dimension.
     */
    private static Conv transposedConv(
            Map<String, GGMLTensorEntry> tensors, String name, int kernel, int inChannels) {
        GGMLTensorEntry entry = require(tensors, name + ".weight");
        return new Conv(
                ModelLoader.loadQuantized(entry),
                ModelLoader.quantOrNull(tensors, name + ".bias"),
                kernel,
                inChannels,
                entry.shape()[1]);
    }

    private static Norm norm(Map<String, GGMLTensorEntry> tensors, String name) {
        GGMLTensorEntry gamma = require(tensors, name + ".gamma");
        return new Norm(
                ModelLoader.loadQuantized(gamma),
                ModelLoader.loadQuantized(require(tensors, name + ".beta")),
                gamma.shape()[0]);
    }

    private static GGMLTensorEntry require(Map<String, GGMLTensorEntry> tensors, String name) {
        GGMLTensorEntry tensor = tensors.get(name);
        if (tensor == null) throw new IllegalArgumentException("missing tensor: " + name);
        return tensor;
    }

    private static Configuration readConfig(GGUF gguf) {
        return new Configuration(
                gguf.getValue(int.class, "inflect.v2.symbol_count"),
                gguf.getValue(int.class, "inflect.v2.inter_channels"),
                gguf.getValue(int.class, "inflect.v2.hidden_channels"),
                gguf.getValue(int.class, "inflect.v2.filter_channels"),
                gguf.getValue(int.class, "inflect.v2.n_heads"),
                gguf.getValue(int.class, "inflect.v2.n_layers"),
                gguf.getValue(int.class, "inflect.v2.kernel_size"),
                gguf.getValue(int.class, "inflect.v2.sample_rate"),
                gguf.getValue(int.class, "inflect.v2.upsample_initial_channel"),
                gguf.getValue(int[].class, "inflect.v2.resblock_kernel_sizes"),
                gguf.getValue(int[].class, "inflect.v2.resblock_dilation_sizes"),
                gguf.getValue(int[].class, "inflect.v2.upsample_rates"),
                gguf.getValue(int[].class, "inflect.v2.upsample_kernel_sizes"));
    }

    // ── model ─────────────────────────────────────────────────────────────

    public Configuration config() {
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

    /**
     * A state over {@code arena}; {@code adopt} makes {@code state.close()} free that arena. See
     * {@link com.qxotic.jinfer.SpeechModel#newState(Arena, boolean)} for the ownership contract.
     */
    public State newState(Arena arena, boolean adopt) {
        if (arena == null) throw new IllegalArgumentException("null arena");
        return new State(arena, adopt ? arena : null);
    }

    /** A state that owns its scratch: an internal {@code ofShared} freed by {@code close()}. */
    public State newState() {
        Arena arena = Arena.ofShared();
        try {
            return newState(arena, true);
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    /**
     * A state whose scratch comes from {@code arena}, BORROWED: the caller owns and frees it, and
     * {@code state.close()} never touches it. Close YOUR arena only after the last synthesis using
     * this state returns — the kernels read raw addresses, so a live read from a closed arena is a
     * crash, not an exception.
     */
    public State newState(Arena arena) {
        return newState(arena, false);
    }

    /**
     * Synthesize a waveform from blank-interspersed phoneme tokens (see {@link Symbols}).
     *
     * @param lengthScale stretches every predicted duration — 1/speed, so 1.25 speaks slower
     * @param variation scale of the latent noise (0 = deterministic, 0.667 is the reference
     *     default)
     */
    public Media.Audio synthesize(
            State state, int[] tokens, float lengthScale, float variation, long seed) {
        // Claims the state for this synthesis: a concurrent one fails fast, and a close waits for
        // this to return rather than freeing the arena under the kernels. Reentrant, so a caller
        // (InflectTTS.speak) may hold it across a whole multi-chunk utterance.
        state.enter();
        try {
            return synthesize0(state, tokens, lengthScale, variation, seed);
        } finally {
            state.exit();
        }
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
        if (!(variation >= 0) || !Float.isFinite(variation))
            throw new IllegalArgumentException("variation must be finite and >= 0: " + variation);

        state.rewind();
        int tokenCount = tokens.length;

        F32FloatTensor encoded = encode(state, tokens);
        // Per-token prior: [mean | logScale] interleaved over 2*interChannels.
        F32FloatTensor stats = conv(state, encoded, w.project(), 1, tokenCount);
        int[] repeats = state.takeInts(tokenCount);
        int frames = predictDurations(state, repeats, encoded, tokenCount, lengthScale);

        F32FloatTensor prior = samplePrior(state, stats, repeats, frames, variation, seed);
        F32FloatTensor flowed = flow(state, prior, frames);
        // The vocoder is the bandwidth-bound part and fires thousands of small parallel regions;
        // onDecodePool runs them on the spin pool, which dispatches without a task tree.
        F32FloatTensor waveform = Parallel.onDecodePool(() -> decode(state, flowed, frames));

        // The one allocation of the pass: the waveform escapes to the caller as a plain array.
        float[] pcm = new float[waveformSamples(frames)];
        waveform.copyRawTo(0, MemorySegment.ofArray(pcm), 0, pcm.length);
        return new Media.Audio(pcm, cfg.sampleRate(), 1);
    }

    // ── encoder ───────────────────────────────────────────────────────────

    /**
     * Embed the tokens and run the transformer. Time-major throughout: {@code x[token][channel]}.
     */
    private F32FloatTensor encode(State state, int[] tokens) {
        int hidden = cfg.hiddenChannels(), tokenCount = tokens.length;
        float scale = (float) Math.sqrt(hidden);
        F32FloatTensor x = state.take(hidden * tokenCount);
        for (int token = 0; token < tokenCount; token++) {
            long row = (long) tokens[token] * w.embeddingStride();
            for (int c = 0; c < hidden; c++)
                x.setFloat((long) token * hidden + c, w.embedding().getFloat(row + c) * scale);
        }
        for (EncoderLayer layer : w.encoder()) x = encoderLayer(state, x, layer, tokenCount);
        return x;
    }

    private F32FloatTensor encoderLayer(
            State state, F32FloatTensor x, EncoderLayer layer, int tokenCount) {
        F32FloatTensor attended = attention(state, x, layer, tokenCount);
        x = addNorm(state, x, attended, layer.attentionNorm(), tokenCount);
        F32FloatTensor wide = conv(state, x, layer.expand(), 1, tokenCount);
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
     * the query contributes an extra term to both the score and the value.
     */
    private F32FloatTensor attention(
            State state, F32FloatTensor x, EncoderLayer layer, int tokenCount) {
        int hidden = layer.query().outChannels(), heads = cfg.nHeads(), headDim = hidden / heads;
        int window = layer.window();
        F32FloatTensor q = conv(state, x, layer.query(), 1, tokenCount);
        F32FloatTensor k = conv(state, x, layer.key(), 1, tokenCount);
        F32FloatTensor v = conv(state, x, layer.value(), 1, tokenCount);
        // `attended` accumulates across heads, so it starts zeroed.
        F32FloatTensor attended = state.takeZeroed(hidden * tokenCount);
        F32FloatTensor scores = state.take(tokenCount);
        float scale = 1f / (float) Math.sqrt(headDim);

        for (int head = 0; head < heads; head++) {
            int channel = head * headDim;
            for (int query = 0; query < tokenCount; query++) {
                long queryRow = (long) query * hidden + channel;
                float max = Float.NEGATIVE_INFINITY;
                for (int key = 0; key < tokenCount; key++) {
                    float score = q.dot(queryRow, k, (long) key * hidden + channel, headDim);
                    int distance = key - query;
                    if (Math.abs(distance) <= window)
                        score +=
                                q.dot(
                                        queryRow,
                                        layer.relativeKeys(),
                                        (long) (distance + window) * headDim,
                                        headDim);
                    score *= scale;
                    scores.setFloat(key, score);
                    max = Math.max(max, score);
                }
                float total = 0;
                for (int key = 0; key < tokenCount; key++) {
                    float weight = (float) Math.exp(scores.getFloat(key) - max);
                    scores.setFloat(key, weight);
                    total += weight;
                }
                for (int key = 0; key < tokenCount; key++) {
                    float weight = scores.getFloat(key) / total;
                    attended.saxpyInPlace(
                            queryRow, v, (long) key * hidden + channel, headDim, weight);
                    int distance = key - query;
                    if (Math.abs(distance) <= window)
                        attended.saxpyInPlace(
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
            State state, int[] repeats, F32FloatTensor encoded, int tokenCount, float lengthScale) {
        Durations dp = w.durations();
        int width = dp.first().outChannels();
        F32FloatTensor h = conv(state, encoded, dp.first(), 1, tokenCount);
        relu(h, width * tokenCount);
        h = norm(state, h, dp.firstNorm(), tokenCount);
        h = conv(state, h, dp.second(), 1, tokenCount);
        relu(h, width * tokenCount);
        h = norm(state, h, dp.secondNorm(), tokenCount);
        F32FloatTensor logDuration = conv(state, h, dp.project(), 1, tokenCount);

        long total = 0;
        for (int token = 0; token < tokenCount; token++) {
            double frames = Math.ceil(Math.exp(logDuration.getFloat(token)) * lengthScale);
            repeats[token] = (int) Math.max(0, Math.min(frames, MAX_FRAMES));
            total += repeats[token];
        }
        if (total > MAX_FRAMES)
            throw new IllegalArgumentException(
                    "chunk needs " + total + " frames, over the " + MAX_FRAMES + " ceiling");
        return (int) Math.max(total, 1);
    }

    /**
     * Repeat each token's latent for its duration and add noise. The Gaussian is drawn
     * channel-major because that order is what the seed means — a frame-major walk would give the
     * same distribution but different audio. The last token also covers any frames left over.
     */
    private F32FloatTensor samplePrior(
            State state,
            F32FloatTensor stats,
            int[] repeats,
            int frames,
            float variation,
            long seed) {
        int latent = cfg.interChannels(), tokenCount = repeats.length;
        Random random = state.random;
        random.setSeed(seed); // same sequence as a fresh Random(seed)
        F32FloatTensor prior = state.take(latent * frames);
        for (int channel = 0; channel < latent; channel++) {
            int frame = 0;
            for (int token = 0; token < tokenCount && frame < frames; token++) {
                long row = (long) token * 2 * latent;
                float mean = stats.getFloat(row + channel);
                float deviation = (float) Math.exp(stats.getFloat(row + latent + channel));
                int last =
                        token == tokenCount - 1 ? frames : Math.min(frames, frame + repeats[token]);
                for (; frame < last; frame++)
                    prior.setFloat(
                            (long) frame * latent + channel,
                            mean + (float) random.nextGaussian() * deviation * variation);
            }
        }
        return prior;
    }

    // ── flow ──────────────────────────────────────────────────────────────

    /** Invert the coupling stack: last coupling first, each preceded by a channel flip. */
    private F32FloatTensor flow(State state, F32FloatTensor z, int frames) {
        int channels = cfg.interChannels();
        for (int i = w.flow().length - 1; i >= 0; i--) {
            flip(z, channels, frames);
            F32FloatTensor next = state.take(channels * frames); // outlives the scope below
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
            State state, F32FloatTensor z, F32FloatTensor out, Coupling layer, int frames) {
        int channels = cfg.interChannels(), half = channels / 2;
        int hidden = layer.pre().outChannels();

        F32FloatTensor first = state.take(half * frames);
        for (int frame = 0; frame < frames; frame++)
            z.copyTo((long) frame * channels, first, (long) frame * half, half);

        F32FloatTensor h = conv(state, first, layer.pre(), 1, frames);
        F32FloatTensor skip = state.takeZeroed(hidden * frames); // accumulates over the layers
        for (int i = 0; i < layer.gates().length; i++) {
            F32FloatTensor gates = conv(state, h, layer.gates()[i], 1, frames);
            // Each step's channels are contiguous, and its two halves are one hidden apart.
            F32FloatTensor activated = state.take(hidden * frames);
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
            F32FloatTensor projected = conv(state, activated, projection, 1, frames);
            int width = projection.outChannels();
            // Every layer but the last splits its projection into a residual and a skip half.
            if (width == 2 * hidden)
                for (int frame = 0; frame < frames; frame++) {
                    h.addInPlace((long) frame * hidden, projected, (long) frame * width, hidden);
                    skip.addInPlace(
                            (long) frame * hidden,
                            projected,
                            (long) frame * width + hidden,
                            hidden);
                }
            else skip.addInPlace(0, projected, 0, hidden * frames);
        }

        F32FloatTensor shift = conv(state, skip, layer.post(), 1, frames);
        z.copyTo(0, out, 0, channels * frames);
        for (int frame = 0; frame < frames; frame++)
            out.saxpyInPlace((long) frame * channels + half, shift, (long) frame * half, half, -1f);
    }

    // ── decoder ───────────────────────────────────────────────────────────

    /**
     * HiFi-GAN vocoder. It runs <b>channel-major</b> — {@code rows[channel][time]}, one contiguous
     * row per channel — unlike the encoder's time-major layout. The shapes are why: the decoder is
     * 12 to 96 channels wide and up to ~110k samples long, so time is the only axis worth
     * vectorizing, and with it contiguous a convolution is an FMA sweep per tap rather than an
     * im2col matrix K times the size of its input.
     */
    private F32FloatTensor decode(State state, F32FloatTensor z, int frames) {
        Decoder decoder = w.decoder();
        F32FloatTensor rows = state.take(cfg.interChannels() * frames);
        toChannelMajor(z, rows, frames, cfg.interChannels());

        int channels = decoder.pre().outChannels(), time = frames;
        F32FloatTensor x = state.take(channels * time);
        convRows(state, rows, x, decoder.pre(), 1, time);

        for (int stage = 0; stage < decoder.upsample().length; stage++) {
            Conv up = decoder.upsample()[stage];
            int stride = decoder.rates()[stage];
            int upTime = upsampledLength(time, up.kernel(), stride);
            int size = up.outChannels() * upTime;

            leaky(x, channels * time, LEAKY);
            // Taken before the scope: the stage's output outlives the resblocks' scratch.
            F32FloatTensor upsampled = state.take(size);
            upsampleRows(state, x, upsampled, up, stride, time, upTime);

            // Every resblock reads the stage's output; their results are averaged back into it.
            ResBlock[] blocks = decoder.resblocks()[stage];
            try (var scope = state.scope()) {
                F32FloatTensor sum = state.takeZeroed(size);
                for (ResBlock block : blocks) resblock(state, upsampled, sum, block, upTime);
                sum.copyTo(0, upsampled, 0, size);
                upsampled.divideInPlace(0, size, blocks.length);
            }

            x = upsampled;
            channels = up.outChannels();
            time = upTime;
        }

        leaky(x, channels * time, FINAL_LEAKY);
        // A single output channel, so channel-major already IS the waveform.
        F32FloatTensor waveform = state.take(time);
        convRows(state, x, waveform, decoder.post(), 1, time);
        // one tanh per audio sample: scalar Math.tanh costs ~15ns each in the native image;
        // the fused FastMath pass is ~0.4ns (contract in TanhAccuracyTest - abs error ~6e-8,
        // below 24-bit audio's LSB)
        com.qxotic.jinfer.FastMath.tanhInPlace(waveform, 0, time);
        return waveform;
    }

    /** One resblock, its result added into {@code sum}. {@code x} is read, not modified. */
    private void resblock(
            State state, F32FloatTensor x, F32FloatTensor sum, ResBlock block, int time) {
        int size = block.filter()[0].outChannels() * time;
        try (var scope = state.scope()) {
            F32FloatTensor value = state.take(size);
            F32FloatTensor activated = state.take(size);
            F32FloatTensor filtered = state.take(size);
            x.copyTo(0, value, 0, size);
            for (int d = 0; d < block.dilations().length; d++) {
                value.copyTo(0, activated, 0, size);
                leaky(activated, size, LEAKY);
                convRows(state, activated, filtered, block.filter()[d], block.dilations()[d], time);
                leaky(filtered, size, LEAKY);
                convRows(state, filtered, activated, block.project()[d], 1, time);
                value.addInPlace(0, activated, 0, size);
            }
            sum.addInPlace(0, value, 0, size);
        }
    }

    /** Samples the decoder produces from {@code frames} — the upsampling recurrence. */
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
     * into a row (im2col) and let the tensor backend do the matrix multiply. Right for the
     * encoder's shapes — few steps, many channels.
     */
    private F32FloatTensor conv(
            State state, F32FloatTensor in, Conv layer, int dilation, int time) {
        int outChannels = layer.outChannels(), rowStride = layer.rowStride();

        // A 1x1 convolution's input already IS the matrix the gemm wants, one row per step — so
        // most of the encoder and the whole flow skip the gather. The exception is a quantized
        // weight whose rows are padded past their taps: then the rows need the padding zeros.
        F32FloatTensor matrix =
                layer.kernel() == 1 && rowStride == layer.inChannels()
                        ? in
                        : gather(state, in, layer, dilation, time);

        F32FloatTensor product = state.product(outChannels * time);
        layer.weight()
                .gemm(matrix, rowStride, product, outChannels, time, outChannels, rowStride, 0);

        F32FloatTensor out = state.take(outChannels * time);
        product.copyTo(0, out, 0, outChannels * time);
        if (layer.bias() != null)
            for (int t = 0; t < time; t++)
                out.addInPlace((long) t * outChannels, layer.bias(), 0, outChannels);
        return out;
    }

    /**
     * im2col: each output step's window laid out as one row, in the gemm's native staging. Zeroed
     * first, because the gather writes only the taps that fall inside the sequence and relies on
     * zeros for the padding either end — and a padded weight row is wider than it fills.
     */
    private F32FloatTensor gather(
            State state, F32FloatTensor in, Conv layer, int dilation, int time) {
        int kernel = layer.kernel(), inChannels = layer.inChannels();
        int rowStride = layer.rowStride(), pad = ((kernel - 1) * dilation) / 2;

        F32FloatTensor columns = state.columns(time * rowStride);
        columns.fillInPlace(0, time * rowStride, 0f);
        for (int t = 0; t < time; t++)
            for (int k = 0; k < kernel; k++) {
                int source = t + k * dilation - pad;
                if (source < 0 || source >= time) continue;
                for (int c = 0; c < inChannels; c++)
                    columns.setFloat(
                            (long) t * rowStride + k + c * kernel,
                            in.getFloat((long) source * inChannels + c));
            }
        return columns;
    }

    /**
     * Channel-major convolution: the layer's taps dequantized once, then {@link
     * Convolutions#conv1dRows}, which owns the tiling, the fan-out and the vector accumulation.
     */
    private void convRows(
            State state,
            F32FloatTensor in,
            F32FloatTensor out,
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
     * Channel-major transposed convolution — the decoder's upsampling step.
     *
     * <p>Upsampling by {@code stride} is {@code stride} independent convolutions, one per output
     * phase: among outputs {@code op = j*stride + phase} only taps with {@code k ≡ phase + pad (mod
     * stride)} contribute, and {@code j} then walks the input contiguously. Each phase is built in
     * its channel's slice of scratch and scattered into the row once, which keeps the sweeps
     * vectorized where a strided write would not be.
     */
    private void upsampleRows(
            State state,
            F32FloatTensor in,
            F32FloatTensor out,
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
            // One phase buffer per output channel: built concurrently, scattered as they finish.
            F32FloatTensor phases = state.take(outChannels * phaseLength);

            Parallel.parallelFor(
                    0,
                    outChannels,
                    channel -> {
                        long phase = (long) channel * phaseLength;
                        long outRow = (long) channel * outTime;
                        for (int p = 0; p < stride; p++) {
                            int count = (outTime - p + stride - 1) / stride;
                            if (count <= 0) continue;
                            phases.fillInPlace(phase, count, layer.bias(channel));
                            for (int k = Math.floorMod(p + pad, stride); k < kernel; k += stride) {
                                int shift = (p + pad - k) / stride; // exact, by the choice of k
                                int start = Math.max(0, -shift);
                                int end = Math.min(count, time - shift);
                                if (end <= start) continue;
                                for (int ic = 0; ic < inChannels; ic++)
                                    phases.saxpyInPlace(
                                            phase + start,
                                            in,
                                            (long) ic * time + start + shift,
                                            end - start,
                                            weights[channel * taps + k * inChannels + ic]);
                            }
                            for (int j = 0, op = p; j < count; j++, op += stride)
                                out.setFloat(outRow + op, phases.getFloat(phase + j));
                        }
                    });
        }
    }

    /** One row of taps per output channel, as stored. */
    private float[] dequantize(State state, Conv layer) {
        int taps = layer.taps(), rowStride = layer.rowStride();
        float[] weights = state.takeWeights(layer.outChannels() * taps);
        for (int oc = 0; oc < layer.outChannels(); oc++)
            layer.weight().copyRow((long) oc * rowStride, weights, oc * taps, taps);
        return weights;
    }

    /**
     * The same for a transposed convolution. Its dense encoding interleaves output channels, so a
     * row is gathered element by element; quantized files are already repacked per output channel.
     */
    private float[] dequantizeTransposed(State state, Conv layer) {
        if (layer.weight().type().isQuantized()) return dequantize(state, layer);
        int kernel = layer.kernel(), inChannels = layer.inChannels();
        int outChannels = layer.outChannels(), taps = layer.taps();
        float[] weights = state.takeWeights(outChannels * taps);
        for (int oc = 0; oc < outChannels; oc++)
            for (int k = 0; k < kernel; k++)
                for (int ic = 0; ic < inChannels; ic++)
                    weights[oc * taps + k * inChannels + ic] =
                            layer.weight()
                                    .getFloat(
                                            k
                                                    + (long) oc * kernel
                                                    + (long) ic * kernel * outChannels);
        return weights;
    }

    // ── small operations ──────────────────────────────────────────────────

    private F32FloatTensor addNorm(
            State state, F32FloatTensor x, F32FloatTensor y, Norm layer, int time) {
        int size = layer.channels() * time;
        F32FloatTensor sum = state.take(size);
        x.copyTo(0, sum, 0, size);
        sum.addInPlace(0, y, 0, size);
        return norm(state, sum, layer, time);
    }

    private F32FloatTensor norm(State state, F32FloatTensor x, Norm layer, int time) {
        F32FloatTensor out = state.take(layer.channels() * time);
        Norms.layerNorm(out, x, layer.gamma(), layer.beta(), layer.channels(), time, 1e-5f);
        return out;
    }

    private static void relu(F32FloatTensor x, int size) {
        x.clampInPlace(0, size, 0f, Float.MAX_VALUE);
    }

    private static void leaky(F32FloatTensor x, int size, float slope) {
        x.leakyReluInPlace(0, size, slope);
    }

    /** Reverse the channel order of every frame — the flow's Flip layer. */
    private static void flip(F32FloatTensor x, int channels, int time) {
        for (int t = 0; t < time; t++)
            for (int c = 0; c < channels / 2; c++) {
                long low = (long) t * channels + c, high = (long) t * channels + (channels - 1 - c);
                float swap = x.getFloat(low);
                x.setFloat(low, x.getFloat(high));
                x.setFloat(high, swap);
            }
    }

    /** {@code [time][channel]} to {@code [channel][time]} — the decoder's entry layout. */
    private static void toChannelMajor(
            F32FloatTensor in, F32FloatTensor out, int time, int channels) {
        for (int c = 0; c < channels; c++)
            for (int t = 0; t < time; t++)
                out.setFloat((long) c * time + t, in.getFloat((long) t * channels + c));
    }

    // ── state ─────────────────────────────────────────────────────────────

    /**
     * Reusable scratch for one synthesis at a time — the forward pass allocates nothing once a
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
     * — which is also why this state can live in a native image heap. A dropped unclosed state
     * leaks its arena until exit; {@code -Djinfer.leakDetection} names the line that dropped it.
     */
    public static final class State implements com.qxotic.jinfer.SpeechState {

        private final Arena arena;
        private final Arena owned; // null when borrowed: closing this state must not free it
        private final Runnable disarm;
        // One lock, three laws - the same contract BaseState carries for generative states:
        // concurrent synthesis fails fast, close BLOCKS to quiescence, entry after close throws.
        private final java.util.concurrent.locks.ReentrantLock lock =
                new java.util.concurrent.locks.ReentrantLock();
        private final java.util.concurrent.atomic.AtomicBoolean closed =
                new java.util.concurrent.atomic.AtomicBoolean();

        State(Arena arena, Arena owned) {
            this.arena = arena;
            this.owned = owned;
            // armed last: nothing above can throw, and a ctor throw must not read as a leak
            this.disarm = com.qxotic.jinfer.LeakWatch.arm(this, "Inflect2.State");
        }

        /**
         * Claims this state for a synthesis on the current thread (reentrant - one utterance may
         * hold it across many chunks). Fails fast: another thread synthesizing -> {@link
         * java.util.ConcurrentModificationException}; closed -> {@link IllegalStateException}.
         */
        void enter() {
            if (closed.get()) throw new IllegalStateException("speech state is closed");
            if (!lock.tryLock()) {
                // the holder is either another synthesis (a contract violation) or the winning
                // closer draining us; `closed` says which
                if (closed.get()) throw new IllegalStateException("speech state is closed");
                throw new java.util.ConcurrentModificationException(
                        "a speech state is a single serial pipeline (one synthesis at a time) -"
                                + " for parallel pipelines create one state each");
            }
            if (closed.get()) { // barged the non-fair lock ahead of a draining closer
                lock.unlock();
                throw new IllegalStateException("speech state is closed");
            }
        }

        /** Releases one {@link #enter} claim. */
        void exit() {
            lock.unlock();
        }

        /** True once {@link #close} has been called; entries then fail loudly. */
        public boolean isClosed() {
            return closed.get();
        }

        /**
         * Idempotent, BLOCKING close: returns only after the in-flight synthesis (if any) has
         * finished, then frees the arena iff this state owns it. Its returning is therefore the
         * caller's quiescence certificate - closing DURING a synthesis is safe, because it waits
         * rather than freeing memory the kernels are reading. After close every entry fails with
         * {@link IllegalStateException}. Racing closers return immediately (the CAS winner waits);
         * closing from within this state's own synthesis throws instead of self-freeing.
         */
        @Override
        public void close() {
            if (lock.isHeldByCurrentThread()) {
                throw new IllegalStateException(
                        "cannot close a speech state from within its own synthesis");
            }
            if (!closed.compareAndSet(false, true)) return; // Arena.close is one-shot
            lock.lock(); // BLOCKS until the in-flight synthesis returns
            try {
                disarm.run();
                if (owned == null) return;
                try {
                    owned.close();
                } catch (UnsupportedOperationException nonCloseable) {
                    // an adopted ofAuto/global manages itself: owning it means nothing to free
                }
            } finally {
                lock.unlock();
            }
        }

        private F32FloatTensor columns, product;
        private F32FloatTensor[] buffers = new F32FloatTensor[64];
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

        /** A tensor of at least {@code size} floats; contents undefined. */
        F32FloatTensor take(int size) {
            if (top == buffers.length) buffers = Arrays.copyOf(buffers, buffers.length * 2);
            F32FloatTensor buffer = buffers[top];
            if (buffer == null || buffer.size() < size)
                buffer = buffers[top] = F32FloatTensor.allocate(arena, size);
            top++;
            return buffer;
        }

        /** A tensor of at least {@code size} floats, zeroed — for accumulators. */
        F32FloatTensor takeZeroed(int size) {
            F32FloatTensor buffer = take(size);
            buffer.fillInPlace(0, size, 0f);
            return buffer;
        }

        /**
         * Heap staging for one convolution's dequantized weights. A {@code float[]} on purpose:
         * {@link FloatTensor#copyRow} dequantizes into one, and the taps are then read as scalars.
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

        /** The gemm's im2col staging, and the gemm's output. */
        F32FloatTensor columns(int size) {
            if (columns == null || columns.size() < size)
                columns = F32FloatTensor.allocate(arena, size);
            return columns;
        }

        F32FloatTensor product(int size) {
            if (product == null || product.size() < size)
                product = F32FloatTensor.allocate(arena, size);
            return product;
        }
    }
}
