package com.qxotic.jinfer.models.lfm3;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jinfer.kernels.Trace;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.llm.SpeechOutput;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.IntConsumer;
import java.util.random.RandomGenerator;

/**
 * Liquid LFM3: the LFM2 hybrid of gated short convolutions and GQA, where attention alternates
 * sliding-window layers (a ring KV cache, plain RoPE) with full layers (partial RoPE through {@code
 * rope_freqs}), every attention output is gated per head by a sigmoid, and both sublayers carry
 * sandwich post-norms. LFM3-VL and LFM3-Omni attach vision and audio towers from their mmproj.
 *
 * <pre>
 *   h  = embed(tokens)
 *   for each layer:
 *     x  = rmsnorm(h)
 *     y  = conv(x)                                   short-conv layers
 *          wo(sigmoid(wg x) * attend(rope(qn(wq x)), rope(kn(wk x)), wv x))   attention layers
 *     h += postnorm(y)
 *     h += postnorm(down(silu(gate(rmsnorm(h))) * up(rmsnorm(h))))
 *   logits = embed^T rmsnorm(h)
 * </pre>
 */
public final class Lfm3
        implements LanguageModel<Lfm3.Configuration, Lfm3.Weights, Lfm3.State>,
                Multimodal,
                SpeechOutput<Lfm3.State> {

    /** The short-conv in_proj's row split: B | C | x, each {@code dim} wide. */
    private static final int SHORTCONV_PARTS = 3;

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;
    private final Lfm3Vision vision;
    private final Lfm3Audio audio;
    private final Lfm3Speech speech;

    private Lfm3(
            Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            Lfm3Vision vision,
            Lfm3Audio audio,
            Lfm3Speech speech) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
        this.vision = vision;
        this.audio = audio;
        this.speech = speech;
    }

    @Override
    public Configuration configuration() {
        return configuration;
    }

    @Override
    public Weights weights() {
        return weights;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    @SuppressWarnings("unchecked")
    public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
        if (modality == Media.Image.class && vision != null)
            return Optional.of((MediaProjector<R>) vision);
        if (modality == Media.Audio.class && audio != null)
            return Optional.of((MediaProjector<R>) audio);
        return Optional.empty();
    }

    /** LFM3-Omni's speech head, or null. */
    Lfm3Speech speech() {
        return speech;
    }

    @Override
    public boolean speaks() {
        return speech != null;
    }

    @Override
    public int speechRate() {
        return Lfm3Speech.SAMPLE_RATE;
    }

    @Override
    public int opener(boolean transcript) {
        return SpecialTokens.require(tokenizer, transcript ? "<|mixed_start|>" : "<|audio_start|>");
    }

    @Override
    public int framePlaceholder() {
        return SpecialTokens.require(tokenizer, "<|audio_out|>");
    }

    @Override
    public SpeechOutput.Speaker speaker(State state, boolean transcript, RandomGenerator random) {
        if (speech == null)
            throw new UnsupportedOperationException("speech needs LFM3-Omni's mmproj");
        return speech.speaker(
                this, state, transcript, random, SpecialTokens.require(tokenizer, "<|text_end|>"));
    }

    /** The final-normed hidden state of the last ingested row: the speech head's input. */
    void hiddenState(State state, MemoryView<MemorySegment> out) {
        state.exclusively(
                () -> {
                    int dim = configuration.embeddingLength;
                    long row = (long) (state.lastBatchSize() - 1) * dim;
                    Norms.rmsnormGgml(
                            out,
                            0,
                            state.residual,
                            row,
                            weights.outputNorm,
                            dim,
                            configuration.rmsNormEps);
                });
        Reference.reachabilityFence(this);
    }

    @Override
    public Optional<CheckpointCodec<State>> checkpointCodec() {
        return Optional.of(new Lfm3CheckpointCodec(configuration));
    }

    @Override
    public State newState(
            int contextCapacity, int batchCapacity, MemoryArena<MemorySegment> arena) {
        return new State(configuration, contextCapacity, batchCapacity, arena, false);
    }

    @Override
    public State newState(int contextCapacity, int batchCapacity) {
        MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
        try {
            return new State(configuration, contextCapacity, batchCapacity, arena, true);
        } catch (RuntimeException | Error failure) {
            Arenas.close(arena);
            throw failure;
        }
    }

    @Override
    public void ingest(State state, Batch batch) {
        ingest(state, batch, null);
    }

    /** As {@link #ingest(State, Batch)}, calling {@code afterBlock} with each finished layer. */
    void ingest(State state, Batch batch, IntConsumer afterBlock) {
        state.exclusively(() -> forward(state, batch, afterBlock));
        Reference.reachabilityFence(this);
    }

    private void forward(State state, Batch batch, IntConsumer afterBlock) {
        int rows = batch.count();
        if (rows <= 0) throw new IllegalArgumentException("batch must not be empty");
        if (rows > state.batchCapacity())
            throw new IllegalArgumentException(
                    "batch " + rows + " exceeds batchCapacity " + state.batchCapacity());
        int start = state.position();
        if (start + rows > state.contextCapacity())
            throw new IllegalArgumentException(
                    "ingest of "
                            + rows
                            + " at position "
                            + start
                            + " exceeds contextCapacity "
                            + state.contextCapacity());
        int dim = configuration.embeddingLength;
        switch (batch.input()) {
            case Batch.Input.Tokens tokens -> {
                int[] ids = tokens.ids();
                for (int id : ids)
                    if (id < 0 || id >= configuration.vocabularySize)
                        throw new IllegalArgumentException(
                                "token id "
                                        + id
                                        + " outside [0,"
                                        + configuration.vocabularySize
                                        + ")");
                Views.checkAlive(weights.tokenEmbeddings, "tokenEmbeddings");
                Convert.gatherToF32(weights.tokenEmbeddings, ids, 0, rows, state.residual, 0, dim);
            }
            case Batch.Input.Embeddings embeddings -> {
                if (vision == null && audio == null && speech == null)
                    throw new UnsupportedOperationException("no media encoder loaded");
                if (embeddings.bidirectional())
                    throw new UnsupportedOperationException("LFM3 attends to media causally");
                if (embeddings.rows().shape().flatAt(1) != dim)
                    throw new IllegalArgumentException(
                            "embedding width "
                                    + embeddings.rows().shape().flatAt(1)
                                    + " != model width "
                                    + dim);
                Convert.copyF32(
                        Views.castToSegmentBacked(embeddings.rows(), "embedding rows"),
                        0,
                        state.residual,
                        0,
                        (long) rows * dim);
            }
            case Batch.Input.Sequences ignored ->
                    throw new UnsupportedOperationException("LFM3 does not pack sequences");
        }
        int lanes = configuration.headSize / 2;
        RoPE.fill(state.ropeCosFull, state.ropeSinFull, start, rows, lanes, weights.ropeFull);
        RoPE.fill(state.ropeCosSwa, state.ropeSinSwa, start, rows, lanes, weights.ropeSwa);
        for (int layer = 0; layer < configuration.numberOfLayers; layer++) {
            block(state, layer, start, rows);
            if (afterBlock != null) afterBlock.accept(layer);
        }
        state.advance(batch);
    }

    /** One pre-norm block in place on the residual: the mixer, then the FFN. */
    private void block(State state, int layer, int start, int rows) {
        int dim = configuration.embeddingLength;
        float eps = configuration.rmsNormEps;
        LayerWeights w = weights.layers[layer];
        Norms.rmsnormRowsGgml(state.normed, state.residual, w.attnNorm, rows, dim, eps);
        if (w.shortConv != null) shortConv(state, layer, w.shortConv, rows);
        else attention(state, layer, w.attention, start, rows);
        if (w.postAttnNorm != null)
            Norms.rmsnormRowsGgml(state.branch, state.branch, w.postAttnNorm, rows, dim, eps);
        Ops.addRows(state.residual, state.branch, rows, dim);

        Norms.rmsnormRowsGgml(state.normed, state.residual, w.ffnNorm, rows, dim, eps);
        Towers.ffn(
                w.ffnGate,
                w.ffnUp,
                w.ffnDown,
                state.normed,
                state.hidden,
                state.hidden2,
                state.branch,
                rows,
                configuration.feedForwardLength);
        if (w.postFfnNorm != null)
            Norms.rmsnormRowsGgml(state.branch, state.branch, w.postFfnNorm, rows, dim, eps);
        Ops.addRows(state.residual, state.branch, rows, dim);
        if (Trace.ENABLED) Trace.sum("l_out-" + layer, state.residual, rows * dim);
    }

    /** in_proj into B | C | x, the causal FIR over B*x gated by C, then out_proj. */
    private void shortConv(State state, int layer, ShortConvWeights w, int rows) {
        int dim = configuration.embeddingLength;
        MatMul.gemm(w.inProj, state.normed, state.shortConvIn, rows);
        Convolutions.shortConvScan(
                w.kernel,
                state.shortConvState[layer],
                state.shortConvIn,
                state.shortConvOut,
                rows,
                dim,
                configuration.shortConvLCache,
                SHORTCONV_PARTS);
        MatMul.gemm(w.outProj, state.shortConvOut, state.branch, rows);
    }

    private void attention(State state, int layer, AttentionWeights w, int start, int rows) {
        Configuration c = configuration;
        int headSize = c.headSize, heads = c.numberOfHeads, kvHeads = c.kvHeads[layer];
        int queryDim = c.queryDim(), kvDim = c.kvDim(layer);
        boolean swa = c.isSwa[layer];
        MemoryView<MemorySegment> cos = swa ? state.ropeCosSwa : state.ropeCosFull;
        MemoryView<MemorySegment> sin = swa ? state.ropeSinSwa : state.ropeSinFull;

        MatMul.gemm(w.query, state.normed, state.query, rows);
        MatMul.gemm(w.key, state.normed, state.batchK, rows);
        MatMul.gemm(w.value, state.normed, state.batchV, rows);
        MatMul.gemm(w.gate, state.normed, state.gate, rows);
        Parallel.forLoop(
                rows,
                row -> {
                    Towers.headNormRope(
                            state.query,
                            (long) row * queryDim,
                            heads,
                            headSize,
                            w.queryNorm,
                            c.rmsNormEps,
                            row,
                            cos,
                            sin);
                    Towers.headNormRope(
                            state.batchK,
                            (long) row * kvDim,
                            kvHeads,
                            headSize,
                            w.keyNorm,
                            c.rmsNormEps,
                            row,
                            cos,
                            sin);
                });

        int window = swa ? c.slidingWindow : 0, ringMask = swa ? c.slidingWindow - 1 : 0;
        float scale = 1f / (float) Math.sqrt(headSize);
        int kvMul = heads / kvHeads;
        MemoryView<MemorySegment> keys = state.keyCache[layer], values = state.valueCache[layer];
        if (rows == 1) {
            FlashAttention.flashDecode(
                    state.query,
                    state.attnOut,
                    keys,
                    values,
                    state.batchK,
                    state.batchV,
                    heads,
                    start,
                    swa ? Math.max(0, start - window + 1) : 0,
                    headSize,
                    kvDim,
                    kvMul,
                    scale,
                    ringMask,
                    null,
                    state.decodeScratch);
        } else {
            FlashAttention.slidingWindowPrefill(
                    state.query,
                    state.attnOut,
                    keys,
                    values,
                    state.batchK,
                    state.batchV,
                    heads,
                    start,
                    rows,
                    headSize,
                    kvDim,
                    queryDim,
                    kvDim,
                    kvMul,
                    scale,
                    window,
                    ringMask,
                    null);
        }
        commitKv(state, layer, start, rows);

        long gateStride = state.gate.stride().flatAt(0);
        Parallel.forLoop(
                rows,
                row -> {
                    for (int head = 0; head < heads; head++) {
                        float g = Views.getFloat(state.gate, row * gateStride + head, "gate");
                        Ops.multiplyInPlace(
                                state.attnOut,
                                (long) row * queryDim + (long) head * headSize,
                                headSize,
                                Activations.sigmoid(g));
                    }
                });
        MatMul.gemm(w.output, state.attnOut, state.branch, rows);
    }

    /** This chunk's K/V into the layer's cache: linear for full layers, the ring for SWA. */
    private void commitKv(State state, int layer, int start, int rows) {
        int kvDim = configuration.kvDim(layer);
        if (!configuration.isSwa[layer]) {
            long at = (long) start * kvDim;
            Convert.f32ToF16Rows(state.batchK, state.keyCache[layer], at, rows, kvDim);
            Convert.f32ToF16Rows(state.batchV, state.valueCache[layer], at, rows, kvDim);
            return;
        }
        // only the last window of a long chunk survives the ring; earlier rows would be overwritten
        int ringMask = configuration.slidingWindow - 1;
        for (int row = Math.max(0, rows - configuration.slidingWindow); row < rows; row++) {
            long at = (long) ((start + row) & ringMask) * kvDim;
            Convert.f32ToF16(state.batchK, (long) row * kvDim, state.keyCache[layer], at, kvDim);
            Convert.f32ToF16(state.batchV, (long) row * kvDim, state.valueCache[layer], at, kvDim);
        }
    }

    @Override
    public MemoryView<?> logits(State state, int output) {
        MemoryView<?> result = state.exclusively(() -> projectLogits(state, output));
        Reference.reachabilityFence(this);
        return result;
    }

    private MemoryView<?> projectLogits(State state, int output) {
        if (output < 0 || output >= state.outputCount())
            throw new IllegalArgumentException(
                    "output " + output + " outside [0," + state.outputCount() + ")");
        int dim = configuration.embeddingLength;
        int row = state.lastBatchSize() - state.outputCount() + output;
        Norms.rmsnormGgml(
                state.normed,
                0,
                state.residual,
                (long) row * dim,
                weights.outputNorm,
                dim,
                configuration.rmsNormEps);
        MatMul.gemv(weights.output, state.normed, state.logits);
        return state.logits;
    }

    // === Configuration ===

    /**
     * @param kvHeads per layer; 0 marks a short-conv layer
     * @param isSwa per layer; a sliding-window attention layer attends the last {@code
     *     slidingWindow} positions through a ring cache
     */
    public record Configuration(
            int embeddingLength,
            int feedForwardLength,
            int numberOfLayers,
            int numberOfHeads,
            int[] kvHeads,
            boolean[] isSwa,
            int headSize,
            int vocabularySize,
            int maxContextLength,
            float rmsNormEps,
            float ropeTheta,
            float ropeThetaSwa,
            int slidingWindow,
            int shortConvLCache)
            implements ContextConfiguration {

        public Configuration {
            kvHeads = kvHeads.clone();
            isSwa = isSwa.clone();
        }

        @Override
        public int[] kvHeads() {
            return kvHeads.clone();
        }

        @Override
        public boolean[] isSwa() {
            return isSwa.clone();
        }

        public boolean isSwa(int layer) {
            return isSwa[layer];
        }

        public boolean isShortConv(int layer) {
            return kvHeads[layer] == 0;
        }

        public int queryDim() {
            return numberOfHeads * headSize;
        }

        public int kvDim(int layer) {
            return kvHeads[layer] * headSize;
        }

        /** Cache rows a layer keeps: the whole context, or the ring of a sliding-window layer. */
        int cachePositions(int layer, int contextCapacity) {
            return isSwa[layer] ? Math.min(contextCapacity, slidingWindow) : contextCapacity;
        }
    }

    // === Weights ===

    public record AttentionWeights(
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> queryNorm,
            MemoryView<MemorySegment> keyNorm) {}

    /** {@code kernel}: per-channel taps {@code c * lCache + k}, as the GGUF lays them out. */
    public record ShortConvWeights(
            MemoryView<MemorySegment> kernel,
            MemoryView<MemorySegment> inProj,
            MemoryView<MemorySegment> outProj) {}

    /** Exactly one of {@code attention} and {@code shortConv}; post-norms are both or neither. */
    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> postAttnNorm,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> postFfnNorm,
            AttentionWeights attention,
            ShortConvWeights shortConv,
            MemoryView<MemorySegment> ffnGate,
            MemoryView<MemorySegment> ffnUp,
            MemoryView<MemorySegment> ffnDown) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbeddings,
            LayerWeights[] layers,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> output,
            RoPE.Schedule ropeFull,
            RoPE.Schedule ropeSwa) {}

    // === State ===

    public static final class State extends ContextState {
        /**
         * The residual stream; normed is every projection's input, branch every sublayer's output.
         */
        final MemoryView<MemorySegment> residual, normed, branch, logits;

        final MemoryView<MemorySegment> query, gate, attnOut, batchK, batchV;
        final MemoryView<MemorySegment> hidden, hidden2;
        final MemoryView<MemorySegment> shortConvIn, shortConvOut;
        final MemoryView<MemorySegment> ropeCosFull, ropeSinFull, ropeCosSwa, ropeSinSwa;
        final FlashAttention.DecodeScratch decodeScratch =
                new FlashAttention.DecodeScratch(memoryArena());

        /** Per layer, null where it does not apply: F16 KV caches, short-conv history. */
        final MemoryView<MemorySegment>[] keyCache, valueCache, shortConvState;

        @SuppressWarnings("unchecked")
        State(
                Configuration c,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity > c.maxContextLength)
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds the model's maxContextLength "
                                + c.maxContextLength);
            MemoryArena<MemorySegment> m = memoryArena();
            int rows = batchCapacity, dim = c.embeddingLength, lanes = c.headSize / 2;
            int maxKvDim = 0;
            for (int layer = 0; layer < c.numberOfLayers; layer++)
                maxKvDim = Math.max(maxKvDim, c.kvDim(layer));
            residual = Views.allocateF32(m, rows, dim);
            normed = Views.allocateF32(m, rows, dim);
            branch = Views.allocateF32(m, rows, dim);
            logits = Views.allocateF32(m, 1, c.vocabularySize);
            query = Views.allocateF32(m, rows, c.queryDim());
            gate = Views.allocateF32(m, rows, c.numberOfHeads);
            attnOut = Views.allocateF32(m, rows, c.queryDim());
            batchK = Views.allocateF32(m, rows, maxKvDim);
            batchV = Views.allocateF32(m, rows, maxKvDim);
            hidden = Views.allocateF32(m, rows, c.feedForwardLength);
            hidden2 = Views.allocateF32(m, rows, c.feedForwardLength);
            shortConvIn = Views.allocateF32(m, rows, SHORTCONV_PARTS * dim);
            shortConvOut = Views.allocateF32(m, rows, dim);
            ropeCosFull = Views.allocateF32(m, rows, lanes);
            ropeSinFull = Views.allocateF32(m, rows, lanes);
            ropeCosSwa = Views.allocateF32(m, rows, lanes);
            ropeSinSwa = Views.allocateF32(m, rows, lanes);
            keyCache = new MemoryView[c.numberOfLayers];
            valueCache = new MemoryView[c.numberOfLayers];
            shortConvState = new MemoryView[c.numberOfLayers];
            for (int layer = 0; layer < c.numberOfLayers; layer++) {
                if (c.isShortConv(layer)) {
                    shortConvState[layer] =
                            Views.allocateF32(m, (long) (c.shortConvLCache - 1) * dim);
                } else {
                    int positions = c.cachePositions(layer, contextCapacity);
                    keyCache[layer] = Views.allocateF16(m, positions, c.kvDim(layer));
                    valueCache[layer] = Views.allocateF16(m, positions, c.kvDim(layer));
                }
            }
        }

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }

        /**
         * Stale KV rows past the cursor are masked and harmless; the short-conv history is not, it
         * would carry the previous conversation into the next.
         */
        @Override
        protected void clearHistory() {
            for (MemoryView<MemorySegment> conv : shortConvState)
                if (conv != null) Ops.fillInPlace(conv, 0, Math.toIntExact(conv.logicalSize()), 0f);
        }
    }

    // === Loading ===

    public static Lfm3 loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return loadModel(channel, ModelLoader.readGguf(channel, path.toString()), arena, null);
        }
    }

    /** The backbone and its LFM3-VL / LFM3-Omni mmproj (vision, audio or both). */
    public static Lfm3 loadModel(Path textPath, Path mmprojPath, Arena arena) throws IOException {
        return loadModel(textPath, arena).withMedia(mmprojPath, arena);
    }

    /**
     * As {@link #loadModel(Path, Arena)} over a parsed header; a null {@code tokenizer} builds the
     * GGUF's own.
     */
    public static Lfm3 loadModel(FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration config = readConfiguration(gguf, tokenizer.vocabulary().size());
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new Lfm3(config, tokenizer, loadWeights(tensors, config), null, null, null);
    }

    /** A model sharing this backbone's weights with the towers of {@code mmprojPath} attached. */
    public Lfm3 withMedia(Path mmprojPath, Arena arena) throws IOException {
        Objects.requireNonNull(mmprojPath, "mmprojPath");
        Objects.requireNonNull(arena, "arena");
        try (FileChannel channel = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, mmprojPath.toString());
            Map<String, MemoryView<MemorySegment>> tensors =
                    ModelLoader.loadTensors(channel, gguf, arena);
            int dim = configuration.embeddingLength;
            Lfm3Vision v =
                    gguf.getValueOrDefault(boolean.class, "clip.has_vision_encoder", false)
                            ? new Lfm3Vision(gguf, tensors, dim, arena)
                            : null;
            Lfm3Audio a =
                    gguf.getValueOrDefault(boolean.class, "clip.has_audio_encoder", false)
                            ? new Lfm3Audio(gguf, tensors, dim)
                            : null;
            Lfm3Speech sp =
                    gguf.getValueOrDefault(boolean.class, "clip.has_gen_audio_encoder", false)
                            ? new Lfm3Speech(gguf, tensors, dim)
                            : null;
            if (v == null && a == null)
                throw new IllegalArgumentException(
                        "'"
                                + mmprojPath.getFileName()
                                + "' has neither a vision nor an audio tower");
            return new Lfm3(configuration, tokenizer, weights, v, a, sp);
        }
    }

    static Configuration readConfiguration(GGUF gguf, int vocabularySize) {
        String arch = gguf.getString("general.architecture");
        require("lfm3".equals(arch), "unsupported architecture '" + arch + "'");
        int layers = gguf.getValue(int.class, "lfm3.block_count");
        int dim = gguf.getValue(int.class, "lfm3.embedding_length");
        int heads = gguf.getValue(int.class, "lfm3.attention.head_count");
        int headSize = gguf.getValue(int.class, "lfm3.attention.key_length");
        require(
                gguf.getValueOrDefault(int.class, "lfm3.attention.value_length", headSize)
                        == headSize,
                "key and value head sizes differ");
        require(
                gguf.getValueOrDefault(int.class, "lfm3.rope.dimension_count", headSize)
                        == headSize,
                "RoPE must span the head (partial RoPE comes from rope_freqs)");
        int[] kvHeads = gguf.getValue(int[].class, "lfm3.attention.head_count_kv");
        boolean[] isSwa = gguf.getValue(boolean[].class, "lfm3.attention.sliding_window_pattern");
        require(kvHeads.length == layers && isSwa.length == layers, "per-layer metadata length");
        int sharedKvHeads = 0;
        for (int layer = 0; layer < layers; layer++) {
            require(
                    kvHeads[layer] == 0 || heads % kvHeads[layer] == 0,
                    "KV heads do not divide query heads at layer " + layer);
            // one batch K/V buffer serves every attention layer
            if (kvHeads[layer] > 0) {
                require(
                        sharedKvHeads == 0 || kvHeads[layer] == sharedKvHeads,
                        "attention layers differ in KV heads at layer " + layer);
                sharedKvHeads = kvHeads[layer];
            }
            require(!isSwa[layer] || kvHeads[layer] > 0, "short-conv layer " + layer + " is SWA");
        }
        int window = gguf.getValue(int.class, "lfm3.attention.sliding_window");
        require(window > 0 && Integer.bitCount(window) == 1, "sliding window must be a power of 2");
        Configuration config =
                new Configuration(
                        dim,
                        gguf.getValue(int.class, "lfm3.feed_forward_length"),
                        layers,
                        heads,
                        kvHeads,
                        isSwa,
                        headSize,
                        vocabularySize,
                        gguf.getValue(int.class, "lfm3.context_length"),
                        gguf.getValue(float.class, "lfm3.attention.layer_norm_rms_epsilon"),
                        gguf.getValue(float.class, "lfm3.rope.freq_base"),
                        gguf.getValue(float.class, "lfm3.rope.freq_base_swa"),
                        window,
                        gguf.getValue(int.class, "lfm3.shortconv.l_cache"));
        require(
                dim > 0 && heads > 0 && headSize > 0 && headSize % 2 == 0 && layers > 0,
                "invalid core dimensions");
        require(config.shortConvLCache > 1, "short-conv cache must exceed one tap");
        require(
                gguf.getValueOrDefault(int.class, "lfm3.vocab_size", vocabularySize)
                        == vocabularySize,
                "tokenizer vocabulary does not match the model");
        return config;
    }

    static Weights loadWeights(Map<String, MemoryView<MemorySegment>> tensors, Configuration c) {
        int dim = c.embeddingLength, ff = c.feedForwardLength, headSize = c.headSize;
        MemoryView<MemorySegment> embeddings =
                weight(tensors, "token_embd.weight", c.vocabularySize, dim);
        MemoryView<MemorySegment> output =
                ModelLoader.find(tensors, "output.weight").orElse(embeddings);
        requireWeight(output, "output.weight", c.vocabularySize, dim);
        LayerWeights[] layers = new LayerWeights[c.numberOfLayers];
        for (int layer = 0; layer < c.numberOfLayers; layer++) {
            String p = "blk." + layer + ".";
            AttentionWeights attention = null;
            ShortConvWeights shortConv = null;
            if (c.isShortConv(layer)) {
                shortConv =
                        new ShortConvWeights(
                                f32(tensors, p + "shortconv.conv.weight", dim, c.shortConvLCache),
                                weight(tensors, p + "shortconv.in_proj.weight", 3 * dim, dim),
                                weight(tensors, p + "shortconv.out_proj.weight", dim, dim));
            } else {
                int kvDim = c.kvDim(layer);
                attention =
                        new AttentionWeights(
                                weight(tensors, p + "attn_q.weight", c.queryDim(), dim),
                                weight(tensors, p + "attn_k.weight", kvDim, dim),
                                weight(tensors, p + "attn_v.weight", kvDim, dim),
                                weight(tensors, p + "attn_output.weight", dim, c.queryDim()),
                                weight(tensors, p + "attn_gate.weight", c.numberOfHeads, dim),
                                f32(tensors, p + "attn_q_norm.weight", headSize),
                                f32(tensors, p + "attn_k_norm.weight", headSize));
            }
            MemoryView<MemorySegment> postAttnNorm =
                    optionalF32(tensors, p + "post_attention_norm.weight", dim);
            MemoryView<MemorySegment> postFfnNorm =
                    optionalF32(tensors, p + "post_ffw_norm.weight", dim);
            require(
                    (postAttnNorm == null) == (postFfnNorm == null),
                    "layer " + layer + " needs both post-norms or neither");
            layers[layer] =
                    new LayerWeights(
                            f32(tensors, p + "attn_norm.weight", dim),
                            postAttnNorm,
                            f32(tensors, p + "ffn_norm.weight", dim),
                            postFfnNorm,
                            attention,
                            shortConv,
                            weight(tensors, p + "ffn_gate.weight", ff, dim),
                            weight(tensors, p + "ffn_up.weight", ff, dim),
                            weight(tensors, p + "ffn_down.weight", dim, ff));
        }
        // full layers rotate the leading pairs only: rope_freqs holds 1 there and 1e30 beyond
        float[] freqFactors =
                Views.toFloatArray(f32(tensors, "rope_freqs.weight", headSize / 2), "rope_freqs");
        return new Weights(
                embeddings,
                layers,
                f32(tensors, "output_norm.weight", dim),
                output,
                RoPE.withFreqFactors(headSize, c.ropeTheta, freqFactors),
                RoPE.plain(headSize, c.ropeThetaSwa));
    }

    static MemoryView<MemorySegment> weight(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int rows, int columns) {
        MemoryView<MemorySegment> value = ModelLoader.require(tensors, name);
        requireWeight(value, name, rows, columns);
        return value;
    }

    private static void requireWeight(
            MemoryView<MemorySegment> value, String name, int rows, int columns) {
        Shape actual = value.dataType().logicalShape(value.shape());
        Shape expected = Shape.flat(rows, columns);
        require(actual.equals(expected), name + " expected " + expected + " but was " + actual);
    }

    static MemoryView<MemorySegment> f32(
            Map<String, MemoryView<MemorySegment>> tensors, String name, long... shape) {
        MemoryView<MemorySegment> value = ModelLoader.requireF32(tensors, name);
        Shape expected = Shape.flat(shape);
        require(
                value.shape().equals(expected),
                name + " expected " + expected + " but was " + value.shape());
        return value;
    }

    private static MemoryView<MemorySegment> optionalF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name, long... shape) {
        return tensors.containsKey(name) ? f32(tensors, name, shape) : null;
    }

    static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("LFM3: " + message);
    }
}
