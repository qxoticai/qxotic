package com.qxotic.jinfer.models.mellum;

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
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Moe;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jinfer.kernels.Trace;
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
import java.util.Optional;

/**
 * JetBrains Mellum 2 decoder ({@code general.architecture = mellum}): a mixture-of-experts
 * transformer with grouped-query attention, per-head Q/K RMS-norm, NeoX rotary and a repeating
 * sliding-window pattern - the window layers rotate with the plain schedule, the full-attention
 * layers with YaRN. Every block routes through softmax top-k experts with renormalized weights
 * (llama.cpp {@code build_moe_ffn}, softmax gating, {@code norm_w}); there is no dense or shared
 * expert. Loaded from GGUF via {@link #loadModel(Path, Arena)}; block caching through a {@link
 * CheckpointCodec} over the FP16 KV caches (linear for full layers, a ring for window layers).
 */
public final class Mellum
        implements LanguageModel<Mellum.Configuration, Mellum.Weights, Mellum.State> {
    static final String ARCHITECTURE = "mellum";

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    Mellum(Configuration configuration, Tokenizer tokenizer, Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
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
    public Optional<CheckpointCodec<State>> checkpointCodec() {
        return Optional.of(new MellumCheckpointCodec(configuration));
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
        state.exclusively(() -> forward(state, batch));
        Reference.reachabilityFence(this);
    }

    private void forward(State state, Batch batch) {
        int rows = batch.count();
        if (rows <= 0 || rows > state.batchCapacity())
            throw new IllegalArgumentException("invalid Mellum batch size " + rows);
        int start = state.position();
        if (start + rows > state.contextCapacity())
            throw new IllegalArgumentException(
                    "ingest of "
                            + rows
                            + " at position "
                            + start
                            + " exceeds contextCapacity "
                            + state.contextCapacity());
        int[] tokens =
                switch (batch.input()) {
                    case Batch.Input.Tokens t -> t.ids();
                    case Batch.Input.Sequences ignored ->
                            throw new UnsupportedOperationException(
                                    "Mellum does not support packed sequences");
                    case Batch.Input.Embeddings ignored ->
                            throw new UnsupportedOperationException("Mellum is text-only");
                };
        for (int token : tokens)
            if (token < 0 || token >= configuration.vocabularySize)
                throw new IllegalArgumentException("token id outside the vocabulary: " + token);
        forward(state, tokens, start, rows);
        state.advance(batch);
    }

    private void forward(State state, int[] tokens, int startPos, int rows) {
        Configuration c = configuration;
        int lanes = c.ropeDimensionCount / 2;
        RoPE.fill(state.ropeCosFull, state.ropeSinFull, startPos, rows, lanes, weights.ropeFull);
        RoPE.fill(state.ropeCosSwa, state.ropeSinSwa, startPos, rows, lanes, weights.ropeSwa);
        Views.checkAlive(weights.tokenEmbedding, "tokenEmbedding");
        Convert.gatherToF32(
                weights.tokenEmbedding, tokens, 0, rows, state.residual, 0, c.embeddingLength);
        for (int layer = 0; layer < c.numberOfLayers; layer++) {
            attention(state, layer, startPos, rows);
            moe(state, layer, rows);
            if (Trace.ENABLED)
                Trace.sum("l_out-" + layer, state.residual, rows * c.embeddingLength);
        }
    }

    /**
     * Pre-norm GQA: Q/K/V projections, per-head Q/K RMS-norm, NeoX RoPE (the window layers on the
     * plain schedule, the full layers on YaRN), causal attention over the layer's own cache ({@code
     * scale = 1/sqrt(headSize)}), output projection added to the residual.
     */
    private void attention(State state, int layer, int startPos, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[layer];
        int dim = c.embeddingLength, headSize = c.headSize, kvDim = c.kvDim();
        int heads = c.numberOfHeads, kvHeads = c.numberOfKeyValueHeads, kvMul = heads / kvHeads;
        int queryDim = c.queryDim();
        boolean swa = c.isSwa[layer];
        MemoryView<MemorySegment> cos = swa ? state.ropeCosSwa : state.ropeCosFull;
        MemoryView<MemorySegment> sin = swa ? state.ropeSinSwa : state.ropeSinFull;
        int ropeLanes = c.ropeDimensionCount / 2;

        Norms.rmsnormRowsGgml(state.normed, state.residual, w.attnNorm, rows, dim, c.rmsNormEps);
        MatMul.gemm(w.query, state.normed, state.query, rows);
        MatMul.gemm(w.key, state.normed, state.batchK, rows);
        MatMul.gemm(w.value, state.normed, state.batchV, rows);
        Parallel.forLoop(
                rows,
                row -> {
                    for (int head = 0; head < heads; head++) {
                        long offset = (long) row * queryDim + (long) head * headSize;
                        Norms.rmsnormGgml(
                                state.query,
                                offset,
                                state.query,
                                offset,
                                w.queryNorm,
                                headSize,
                                c.rmsNormEps);
                        RoPE.applyNeox(state.query, offset, row, cos, sin, ropeLanes);
                    }
                    for (int head = 0; head < kvHeads; head++) {
                        long offset = (long) row * kvDim + (long) head * headSize;
                        Norms.rmsnormGgml(
                                state.batchK,
                                offset,
                                state.batchK,
                                offset,
                                w.keyNorm,
                                headSize,
                                c.rmsNormEps);
                        RoPE.applyNeox(state.batchK, offset, row, cos, sin, ropeLanes);
                    }
                });

        float scale = 1f / (float) Math.sqrt(headSize);
        if (rows > 1)
            FlashAttention.slidingWindowPrefill(
                    state.query,
                    state.attentionOut,
                    state.keyCache[layer],
                    state.valueCache[layer],
                    state.batchK,
                    state.batchV,
                    heads,
                    startPos,
                    rows,
                    headSize,
                    kvDim,
                    queryDim,
                    kvDim,
                    kvMul,
                    scale,
                    swa ? c.slidingWindow : 0,
                    swa ? c.slidingWindow - 1 : 0,
                    null);
        else
            FlashAttention.flashDecode(
                    state.query,
                    state.attentionOut,
                    state.keyCache[layer],
                    state.valueCache[layer],
                    state.batchK,
                    state.batchV,
                    heads,
                    startPos,
                    c.attentionStart(layer, startPos),
                    headSize,
                    kvDim,
                    kvMul,
                    scale,
                    swa ? c.slidingWindow - 1 : 0,
                    null,
                    state.decodeScratch);
        MatMul.gemm(w.output, state.attentionOut, state.branch, rows);
        Ops.addInPlace(state.residual, 0, state.branch, 0, rows * dim);
        commitKv(state, layer, startPos, rows);
    }

    /**
     * Writes the chunk's K/V into layer {@code layer}'s cache - inside the layer, because {@code
     * batchK}/{@code batchV} are reused by every layer. Window layers write through their ring
     * slot.
     */
    private void commitKv(State state, int layer, int startPos, int rows) {
        Configuration c = configuration;
        int kvDim = c.kvDim();
        for (int row = 0; row < rows; row++) {
            long position = c.kvCacheIndex(layer, startPos + row);
            Convert.f32ToF16(
                    state.batchK,
                    (long) row * kvDim,
                    state.keyCache[layer],
                    position * kvDim,
                    kvDim);
            Convert.f32ToF16(
                    state.batchV,
                    (long) row * kvDim,
                    state.valueCache[layer],
                    position * kvDim,
                    kvDim);
        }
    }

    /**
     * Pre-norm MoE block: router logits, softmax, top-k, the k weights renormalized to sum to one,
     * then each selected expert's SwiGLU FFN scatter-added into the residual.
     */
    private void moe(State state, int layer, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[layer];
        int dim = c.embeddingLength, expertFf = c.expertFeedForwardLength;
        Norms.rmsnormRowsGgml(state.normed, state.residual, w.ffnNorm, rows, dim, c.rmsNormEps);
        MatMul.gemm(w.router, state.normed, state.moeRouter, rows);
        Moe.softmaxSelectTopK(
                state.moeRouter,
                rows,
                c.expertCount,
                c.expertUsedCount,
                state.moeRowTopE,
                state.moeRowTopP,
                state.moeExpertCounts);
        Moe.dispatch(
                state.moeRouting,
                rows,
                dim,
                state.normed,
                state.moeGather,
                state.moeDown,
                state.branch,
                null,
                (expert, count, gather, output) -> {
                    MatMul.gemm(w.expertGate[expert], gather, state.moeHidden, count);
                    MatMul.gemm(w.expertUp[expert], gather, state.moeHidden2, count);
                    Activations.siluMultiply(
                            state.moeHidden, 0, state.moeHidden2, 0, count * expertFf);
                    MatMul.gemm(w.expertDown[expert], state.moeHidden, output, count);
                });
        Ops.addInPlace(state.residual, 0, state.branch, 0, rows * dim);
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
        MatMul.gemv(weights.outputWeight, state.normed, state.logits);
        return state.logits;
    }

    // === Configuration ===

    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int headSize,
            int vocabularySize,
            int maxContextLength,
            float rmsNormEps,
            int expertCount,
            int expertUsedCount,
            int expertFeedForwardLength,
            int slidingWindow,
            boolean[] isSwa,
            int ropeDimensionCount,
            double ropeTheta,
            double ropeThetaSwa,
            float ropeScalingFactor,
            int ropeOriginalContext,
            float ropeBetaFast,
            float ropeBetaSlow,
            float ropeAttentionFactor)
            implements ContextConfiguration {

        int queryDim() {
            return numberOfHeads * headSize;
        }

        int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        int kvCachePositions(int layer, int capacity) {
            return isSwa[layer] ? Math.min(capacity, slidingWindow) : capacity;
        }

        int kvCacheIndex(int layer, int position) {
            return isSwa[layer] ? position & (slidingWindow - 1) : position;
        }

        int attentionStart(int layer, int position) {
            return isSwa[layer] ? Math.max(0, position - slidingWindow + 1) : 0;
        }
    }

    // === Weights ===

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> queryNorm,
            MemoryView<MemorySegment> keyNorm,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> router,
            MemoryView<MemorySegment>[] expertGate,
            MemoryView<MemorySegment>[] expertUp,
            MemoryView<MemorySegment>[] expertDown) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbedding,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> outputWeight,
            LayerWeights[] layers,
            RoPE.Schedule ropeFull,
            RoPE.Schedule ropeSwa) {}

    // === State ===

    public static final class State extends ContextState {
        final MemoryView<MemorySegment> residual, normed, branch, logits;
        final MemoryView<MemorySegment> query, attentionOut, batchK, batchV;
        final MemoryView<MemorySegment> ropeCosFull, ropeSinFull, ropeCosSwa, ropeSinSwa;
        final MemoryView<MemorySegment> moeRouter, moeGather, moeDown, moeHidden, moeHidden2;
        final MemoryView<MemorySegment>[] keyCache, valueCache;
        final FlashAttention.DecodeScratch decodeScratch;
        final int[] moeExpertCounts, moeRowTopE;
        final float[] moeRowTopP;
        final Moe.Routing moeRouting;

        @SuppressWarnings("unchecked")
        State(
                Configuration c,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity <= 0 || contextCapacity > c.maxContextLength)
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " outside (0, "
                                + c.maxContextLength
                                + "], the model's maxContextLength");
            int rows = batchCapacity(), dim = c.embeddingLength, kvDim = c.kvDim();
            int lanes = c.ropeDimensionCount / 2;
            residual = Views.allocateF32(memoryArena(), rows, dim);
            normed = Views.allocateF32(memoryArena(), rows, dim);
            branch = Views.allocateF32(memoryArena(), rows, dim);
            logits = Views.allocateF32(memoryArena(), 1, c.vocabularySize);
            query = Views.allocateF32(memoryArena(), rows, c.queryDim());
            attentionOut = Views.allocateF32(memoryArena(), rows, c.queryDim());
            batchK = Views.allocateF32(memoryArena(), rows, kvDim);
            batchV = Views.allocateF32(memoryArena(), rows, kvDim);
            ropeCosFull = Views.allocateF32(memoryArena(), rows, lanes);
            ropeSinFull = Views.allocateF32(memoryArena(), rows, lanes);
            ropeCosSwa = Views.allocateF32(memoryArena(), rows, lanes);
            ropeSinSwa = Views.allocateF32(memoryArena(), rows, lanes);
            moeRouter = Views.allocateF32(memoryArena(), rows, c.expertCount);
            moeGather = Views.allocateF32(memoryArena(), rows, dim);
            moeDown = Views.allocateF32(memoryArena(), rows, dim);
            moeHidden = Views.allocateF32(memoryArena(), rows, c.expertFeedForwardLength);
            moeHidden2 = Views.allocateF32(memoryArena(), rows, c.expertFeedForwardLength);
            decodeScratch = new FlashAttention.DecodeScratch(memoryArena());
            moeExpertCounts = new int[c.expertCount];
            moeRowTopE = new int[rows * c.expertUsedCount];
            moeRowTopP = new float[rows * c.expertUsedCount];
            moeRouting =
                    new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts, c.expertUsedCount);
            keyCache = new MemoryView[c.numberOfLayers];
            valueCache = new MemoryView[c.numberOfLayers];
            for (int layer = 0; layer < c.numberOfLayers; layer++) {
                int positions = c.kvCachePositions(layer, contextCapacity);
                keyCache[layer] = Views.allocateF16(memoryArena(), positions, kvDim);
                valueCache[layer] = Views.allocateF16(memoryArena(), positions, kvDim);
            }
        }

        /** Pure attention carries nothing but KV; rows beyond the cursor are masked, not read. */
        @Override
        protected void clearHistory() {}

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }
    }

    // === Loading ===

    /**
     * Loads a Mellum GGUF, memory-mapping its weights into {@code arena} (which owns their
     * lifetime). The tokenizer is built from the GGUF metadata.
     *
     * @throws IllegalArgumentException if the file is not a {@code mellum} architecture model
     */
    public static Mellum loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(channel, gguf, arena);
        }
    }

    /** As {@link #loadModel(Path, Arena)} over an already-parsed GGUF header. */
    public static Mellum loadModel(FileChannel channel, GGUF gguf, Arena arena) throws IOException {
        return loadModel(channel, gguf, arena, null);
    }

    /**
     * As {@link #loadModel(Path, Arena)} with a caller-supplied {@code tokenizer}; {@code null}
     * builds one from the GGUF metadata.
     */
    public static Mellum loadModel(FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        String arch = gguf.getString("general.architecture");
        if (!ARCHITECTURE.equals(arch))
            throw new IllegalArgumentException("unsupported architecture: " + arch);
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration c = loadConfiguration(gguf, tokenizer.vocabulary().size());
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new Mellum(c, tokenizer, loadWeights(tensors, c));
    }

    static Configuration loadConfiguration(GGUF gguf, int vocabularySize) {
        String arch = ARCHITECTURE;
        int layers = gguf.getValue(int.class, arch + ".block_count");
        int dim = gguf.getValue(int.class, arch + ".embedding_length");
        int heads = gguf.getValue(int.class, arch + ".attention.head_count");
        int headSize =
                gguf.getValueOrDefault(int.class, arch + ".attention.key_length", dim / heads);
        int valueSize =
                gguf.getValueOrDefault(int.class, arch + ".attention.value_length", headSize);
        int context = gguf.getValue(int.class, arch + ".context_length");
        int window = gguf.getValueOrDefault(int.class, arch + ".attention.sliding_window", 0);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10_000f);
        Configuration c =
                new Configuration(
                        dim,
                        layers,
                        heads,
                        gguf.getValue(int.class, arch + ".attention.head_count_kv"),
                        headSize,
                        vocabularySize,
                        context,
                        gguf.getValueOrDefault(
                                float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-6f),
                        gguf.getValue(int.class, arch + ".expert_count"),
                        gguf.getValue(int.class, arch + ".expert_used_count"),
                        gguf.getValue(int.class, arch + ".expert_feed_forward_length"),
                        window,
                        swaLayers(gguf, arch, layers, window),
                        gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize),
                        ropeTheta,
                        gguf.getValueOrDefault(
                                float.class, arch + ".rope.freq_base_swa", ropeTheta),
                        gguf.getValueOrDefault(float.class, arch + ".rope.scaling.factor", 1f),
                        gguf.getValueOrDefault(
                                int.class, arch + ".rope.scaling.original_context_length", context),
                        gguf.getValueOrDefault(
                                float.class, arch + ".rope.scaling.yarn_beta_fast", 32f),
                        gguf.getValueOrDefault(
                                float.class, arch + ".rope.scaling.yarn_beta_slow", 1f),
                        // llama.cpp's rope_attn_factor: the bare multiplier on top of the YaRN
                        // magnitude RoPE.yarn derives itself (the converter's yarn_attn_factor
                        // key, which llama.cpp never reads, is that derived magnitude)
                        gguf.getValueOrDefault(
                                float.class, arch + ".rope.scaling.attn_factor", 1f));
        validate(c, valueSize);
        String scaling = gguf.getValueOrDefault(String.class, arch + ".rope.scaling.type", "none");
        require(
                scaling.equals("none") || scaling.equals("yarn"),
                "unsupported RoPE scaling " + scaling);
        require(
                gguf.getValueOrDefault(int.class, arch + ".vocab_size", vocabularySize)
                        == vocabularySize,
                "tokenizer vocabulary does not match the model");
        return c;
    }

    /**
     * Which layers slide: llama.cpp reads {@code attention.sliding_window_pattern} either as the
     * period {@code n} (layer {@code il} slides unless {@code il % n == 0}; {@code 0} means every
     * layer) or as one boolean per layer, and defaults to a period of 4. No window: no layer
     * slides.
     */
    private static boolean[] swaLayers(GGUF gguf, String arch, int layers, int window) {
        boolean[] swa = new boolean[layers];
        if (window <= 0) return swa;
        Object pattern =
                gguf.getValueOrDefault(Object.class, arch + ".attention.sliding_window_pattern", 4);
        if (pattern instanceof boolean[] perLayer) {
            require(
                    perLayer.length == layers,
                    "sliding_window_pattern must have one flag per layer");
            return perLayer.clone();
        }
        int period = ((Number) pattern).intValue();
        require(period >= 0, "sliding-window pattern must not be negative");
        for (int layer = 0; layer < layers; layer++)
            swa[layer] = period == 0 || layer % period != 0;
        return swa;
    }

    private static void validate(Configuration c, int valueSize) {
        require(
                c.embeddingLength > 0
                        && c.numberOfLayers > 0
                        && c.numberOfHeads > 0
                        && c.numberOfKeyValueHeads > 0
                        && c.numberOfHeads % c.numberOfKeyValueHeads == 0
                        && c.headSize > 0
                        && c.vocabularySize > 0
                        && c.maxContextLength > 0,
                "invalid core dimensions");
        require(valueSize == c.headSize, "different key/value head sizes are unsupported");
        require(
                c.ropeDimensionCount > 0
                        && (c.ropeDimensionCount & 1) == 0
                        && c.ropeDimensionCount <= c.headSize,
                "invalid RoPE dimensions");
        require(
                c.rmsNormEps > 0f
                        && Float.isFinite(c.rmsNormEps)
                        && c.ropeTheta > 0
                        && c.ropeThetaSwa > 0
                        && c.ropeScalingFactor > 0f
                        && c.ropeOriginalContext > 0
                        && c.ropeAttentionFactor > 0f,
                "invalid normalization or RoPE metadata");
        require(
                c.expertCount > 0
                        && c.expertUsedCount > 0
                        && c.expertUsedCount <= c.expertCount
                        && c.expertFeedForwardLength > 0,
                "invalid MoE metadata");
        require(c.isSwa.length == c.numberOfLayers, "one sliding-window flag per layer");
        boolean anySwa = false;
        for (boolean swa : c.isSwa) anySwa |= swa;
        if (anySwa)
            require(
                    c.slidingWindow > 0 && Integer.bitCount(c.slidingWindow) == 1,
                    "sliding window must be a power of two");
    }

    static Weights loadWeights(Map<String, MemoryView<MemorySegment>> tensors, Configuration c) {
        int dim = c.embeddingLength, queryDim = c.queryDim(), kvDim = c.kvDim();
        int experts = c.expertCount, expertFf = c.expertFeedForwardLength;
        LayerWeights[] layers = new LayerWeights[c.numberOfLayers];
        for (int layer = 0; layer < layers.length; layer++) {
            String p = "blk." + layer + ".";
            layers[layer] =
                    new LayerWeights(
                            f32(tensors, p + "attn_norm.weight", dim),
                            weight(tensors, p + "attn_q.weight", queryDim, dim),
                            weight(tensors, p + "attn_k.weight", kvDim, dim),
                            weight(tensors, p + "attn_v.weight", kvDim, dim),
                            f32(tensors, p + "attn_q_norm.weight", c.headSize),
                            f32(tensors, p + "attn_k_norm.weight", c.headSize),
                            weight(tensors, p + "attn_output.weight", dim, queryDim),
                            f32(tensors, p + "ffn_norm.weight", dim),
                            weight(tensors, p + "ffn_gate_inp.weight", experts, dim),
                            experts(tensors, p + "ffn_gate_exps.weight", experts, expertFf, dim),
                            experts(tensors, p + "ffn_up_exps.weight", experts, expertFf, dim),
                            experts(tensors, p + "ffn_down_exps.weight", experts, dim, expertFf));
        }
        MemoryView<MemorySegment> embedding =
                weight(tensors, "token_embd.weight", c.vocabularySize, dim);
        MemoryView<MemorySegment> head =
                ModelLoader.find(tensors, "output.weight").orElse(embedding);
        requireShape(head, "output.weight", c.vocabularySize, dim);
        // the full-attention layers rotate with YaRN, the window layers with the plain schedule
        // at their own base (llama.cpp: freq_scale 1, ext_factor 0, attn_factor 1 on SWA layers)
        RoPE.Schedule full =
                c.ropeScalingFactor == 1f
                        ? RoPE.plain(c.ropeDimensionCount, c.ropeTheta)
                        : RoPE.yarn(
                                c.ropeDimensionCount,
                                c.ropeTheta,
                                c.ropeScalingFactor,
                                c.ropeOriginalContext,
                                c.ropeBetaFast,
                                c.ropeBetaSlow,
                                1f,
                                c.ropeAttentionFactor);
        RoPE.Schedule swa = RoPE.plain(c.ropeDimensionCount, c.ropeThetaSwa);
        return new Weights(
                embedding, f32(tensors, "output_norm.weight", dim), head, layers, full, swa);
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment>[] experts(
            Map<String, MemoryView<MemorySegment>> tensors,
            String name,
            int experts,
            int rows,
            int columns) {
        MemoryView<MemorySegment> stacked = ModelLoader.require(tensors, name);
        Shape actual = stacked.dataType().logicalShape(stacked.shape());
        Shape expected = Shape.flat(experts, rows, columns);
        require(actual.equals(expected), name + " expected " + expected + " but was " + actual);
        return Views.sliceLeadingAxis(stacked, experts);
    }

    private static MemoryView<MemorySegment> weight(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int rows, int columns) {
        MemoryView<MemorySegment> value = ModelLoader.require(tensors, name);
        requireShape(value, name, rows, columns);
        return value;
    }

    private static void requireShape(
            MemoryView<MemorySegment> value, String name, int rows, int columns) {
        Shape actual = value.dataType().logicalShape(value.shape());
        Shape expected = Shape.flat(rows, columns);
        require(actual.equals(expected), name + " expected " + expected + " but was " + actual);
    }

    private static MemoryView<MemorySegment> f32(
            Map<String, MemoryView<MemorySegment>> tensors, String name, long... shape) {
        MemoryView<MemorySegment> value = ModelLoader.requireF32(tensors, name);
        Shape expected = Shape.flat(shape);
        require(
                value.shape().equals(expected),
                name + " expected " + expected + " but was " + value.shape());
        return value;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Mellum: " + message);
    }
}
