package com.qxotic.jinfer.models.laguna;

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
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;

/** Poolside Laguna XS 2.1 decoder. */
public final class Laguna
        implements LanguageModel<Laguna.Configuration, Laguna.Weights, Laguna.State> {
    private static final int SIGMOID_GATING = 2;

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    Laguna(Configuration configuration, Tokenizer tokenizer, Weights weights) {
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
        return Optional.of(new LagunaCheckpointCodec(configuration));
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
            throw new IllegalArgumentException("invalid Laguna batch size " + rows);
        int start = state.position();
        if (start + rows > state.contextCapacity())
            throw new IllegalArgumentException("batch exceeds the allocated context");
        int[] tokens =
                switch (batch.input()) {
                    case Batch.Input.Tokens t -> t.ids();
                    case Batch.Input.Sequences ignored ->
                            throw new UnsupportedOperationException(
                                    "Laguna does not support packed sequences");
                    case Batch.Input.Embeddings ignored ->
                            throw new UnsupportedOperationException("Laguna is text-only");
                };
        for (int token : tokens)
            if (token < 0 || token >= configuration.vocabularySize)
                throw new IllegalArgumentException("token id outside the vocabulary: " + token);
        forward(state, tokens, start, rows);
        state.advance(batch);
    }

    private void forward(State state, int[] tokens, int startPos, int rows) {
        Configuration c = configuration;
        RoPE.fill(
                state.ropeCosFull,
                state.ropeSinFull,
                startPos,
                rows,
                c.ropeDimensionCount / 2,
                weights.ropeFull);
        if (c.hasSwa())
            RoPE.fill(
                    state.ropeCosSwa,
                    state.ropeSinSwa,
                    startPos,
                    rows,
                    c.ropeDimensionCountSwa / 2,
                    weights.ropeSwa);
        Views.checkAlive(weights.tokenEmbedding, "tokenEmbedding");
        Convert.gatherToF32(
                weights.tokenEmbedding, tokens, 0, rows, state.residual, 0, c.embeddingLength);
        for (int layer = 0; layer < c.numberOfLayers; layer++)
            decoderBlock(state, layer, startPos, rows);
    }

    private void decoderBlock(State state, int layer, int startPos, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[layer];
        Norms.rmsnormRowsGgml(
                state.normed, state.residual, w.attnNorm, rows, c.embeddingLength, c.rmsNormEps);
        attention(state, layer, startPos, rows);
        Ops.addInPlace(state.residual, 0, state.branch, 0, rows * c.embeddingLength);
        Norms.rmsnormRowsGgml(
                state.normed, state.residual, w.ffnNorm, rows, c.embeddingLength, c.rmsNormEps);
        if (layer < c.denseLeadingLayers) denseFfn(state, w.dense, rows);
        else moe(state, w.moe, rows);
        Ops.addInPlace(state.residual, 0, state.branch, 0, rows * c.embeddingLength);
        if (Trace.ENABLED) Trace.sum("l_out-" + layer, state.residual, rows * c.embeddingLength);
    }

    private void attention(State state, int layer, int startPos, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[layer];
        int heads = c.headCount[layer], headSize = c.headSize, kvDim = c.kvDim();
        int kvMul = heads / c.keyValueHeadCount;
        boolean swa = c.isSwa[layer];
        MemoryView<MemorySegment> cos = swa ? state.ropeCosSwa : state.ropeCosFull;
        MemoryView<MemorySegment> sin = swa ? state.ropeSinSwa : state.ropeSinFull;
        int ropeLanes = (swa ? c.ropeDimensionCountSwa : c.ropeDimensionCount) / 2;

        MatMul.gemm(w.query, state.normed, state.query, rows);
        MatMul.gemm(w.key, state.normed, state.batchK, rows);
        MatMul.gemm(w.value, state.normed, state.batchV, rows);
        MatMul.gemm(w.attentionGate, state.normed, state.attentionGate, rows);
        long queryStride = state.query.stride().flatAt(0);
        Parallel.forLoop(
                rows,
                row -> {
                    long queryRow = row * queryStride;
                    for (int head = 0; head < heads; head++) {
                        long offset = queryRow + (long) head * headSize;
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
                    for (int head = 0; head < c.keyValueHeadCount; head++) {
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
                Math.toIntExact(queryStride),
                kvDim,
                kvMul,
                1f / (float) Math.sqrt(headSize),
                swa ? c.slidingWindow : 0,
                swa ? c.slidingWindow - 1 : 0,
                null);

        long outputStride = state.attentionOut.stride().flatAt(0);
        long gateStride = state.attentionGate.stride().flatAt(0);
        for (int row = 0; row < rows; row++)
            for (int head = 0; head < heads; head++) {
                float gate =
                        Activations.softplus(
                                Views.getFloat(
                                        state.attentionGate,
                                        row * gateStride + head,
                                        "attentionGate"));
                Ops.multiplyInPlace(
                        state.attentionOut,
                        row * outputStride + (long) head * headSize,
                        headSize,
                        gate);
            }
        MatMul.gemm(w.output, state.attentionOut, state.branch, rows);
        commitKv(state, layer, startPos, rows);
    }

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

    private void denseFfn(State state, DenseWeights w, int rows) {
        Configuration c = configuration;
        MatMul.gemm(w.gate, state.normed, state.denseHidden, rows);
        MatMul.gemm(w.up, state.normed, state.denseHidden2, rows);
        Activations.siluMultiply(
                state.denseHidden, 0, state.denseHidden2, 0, rows * c.feedForwardLength);
        MatMul.gemm(w.down, state.denseHidden, state.branch, rows);
    }

    private void moe(State state, MoeWeights w, int rows) {
        Configuration c = configuration;
        int routes = rows * c.expertUsedCount;
        MatMul.gemm(w.router, state.normed, state.moeRouter, rows);
        Ops.mapInPlace(state.moeRouter, 0, rows * c.expertCount, Activations::sigmoid);
        Convert.copyF32(state.moeRouter, 0, state.moeSelection, 0, (long) rows * c.expertCount);
        Ops.addRowBiasInPlace(state.moeSelection, 0, w.selectionBias, 0, rows, c.expertCount);
        Moe.selectTopK(
                state.moeSelection,
                state.moeRouter,
                rows,
                c.expertCount,
                c.expertUsedCount,
                state.moeRowTopE,
                state.moeRowTopP,
                state.moeExpertCounts);
        Moe.normalizeTopP(state.moeRowTopP, rows, c.expertUsedCount);
        for (int route = 0; route < routes; route++)
            state.moeRowTopP[route] *= c.expertWeightsScale;
        Moe.Routing routing = state.moeRouting;
        routing.seqLen = rows;
        Moe.dispatch(
                routing,
                c.embeddingLength,
                state.normed,
                state.moeGather,
                state.moeDown,
                state.branch,
                null,
                (expert, count, gather, output) -> {
                    MatMul.gemm(w.expertGate[expert], gather, state.moeHidden, count);
                    MatMul.gemm(w.expertUp[expert], gather, state.moeHidden2, count);
                    Activations.siluMultiply(
                            state.moeHidden,
                            0,
                            state.moeHidden2,
                            0,
                            count * c.expertFeedForwardLength);
                    MatMul.gemm(w.expertDown[expert], state.moeHidden, output, count);
                });
        MatMul.gemm(w.sharedGate, state.normed, state.sharedHidden, rows);
        MatMul.gemm(w.sharedUp, state.normed, state.sharedHidden2, rows);
        Activations.siluMultiply(
                state.sharedHidden, 0, state.sharedHidden2, 0, rows * c.sharedFeedForwardLength);
        MatMul.gemm(w.sharedDown, state.sharedHidden, state.sharedOut, rows);
        Ops.addInPlace(state.branch, 0, state.sharedOut, 0, rows * c.embeddingLength);
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
        int row = state.lastBatchSize() - state.outputCount() + output;
        Norms.rmsnormGgml(
                state.normed,
                0,
                state.residual,
                (long) row * configuration.embeddingLength,
                weights.outputNorm,
                configuration.embeddingLength,
                configuration.rmsNormEps);
        MatMul.gemv(weights.outputWeight, state.normed, state.logits);
        return state.logits;
    }

    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int[] headCount,
            int keyValueHeadCount,
            int headSize,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            int feedForwardLength,
            int denseLeadingLayers,
            int expertCount,
            int expertUsedCount,
            int expertFeedForwardLength,
            int sharedFeedForwardLength,
            float expertWeightsScale,
            int slidingWindow,
            boolean[] isSwa,
            int ropeDimensionCount,
            int ropeDimensionCountSwa,
            double ropeTheta,
            double ropeThetaSwa,
            float ropeScalingFactor,
            int ropeOriginalContext,
            float ropeBetaFast,
            float ropeBetaSlow,
            float ropeAttentionFactor)
            implements ContextConfiguration {
        int queryDim(int layer) {
            return headCount[layer] * headSize;
        }

        int maxQueryDim() {
            return Arrays.stream(headCount).max().orElseThrow() * headSize;
        }

        int maxHeadCount() {
            return Arrays.stream(headCount).max().orElseThrow();
        }

        int kvDim() {
            return keyValueHeadCount * headSize;
        }

        boolean hasSwa() {
            return slidingWindow > 0;
        }

        int kvCachePositions(int layer, int capacity) {
            return isSwa[layer] ? Math.min(capacity, slidingWindow) : capacity;
        }

        int kvCacheIndex(int layer, int position) {
            return isSwa[layer] ? position & (slidingWindow - 1) : position;
        }
    }

    public record DenseWeights(
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> up,
            MemoryView<MemorySegment> down) {}

    public record MoeWeights(
            MemoryView<MemorySegment> router,
            MemoryView<MemorySegment> selectionBias,
            MemoryView<MemorySegment>[] expertGate,
            MemoryView<MemorySegment>[] expertUp,
            MemoryView<MemorySegment>[] expertDown,
            MemoryView<MemorySegment> sharedGate,
            MemoryView<MemorySegment> sharedUp,
            MemoryView<MemorySegment> sharedDown) {}

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> queryNorm,
            MemoryView<MemorySegment> keyNorm,
            MemoryView<MemorySegment> attentionGate,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> ffnNorm,
            DenseWeights dense,
            MoeWeights moe) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbedding,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> outputWeight,
            LayerWeights[] layers,
            RoPE.Schedule ropeFull,
            RoPE.Schedule ropeSwa) {}

    public static final class State extends ContextState {
        final MemoryView<MemorySegment> residual, normed, branch, logits;
        final MemoryView<MemorySegment> query, attentionGate, attentionOut;
        final MemoryView<MemorySegment> ropeCosFull, ropeSinFull, ropeCosSwa, ropeSinSwa;
        final MemoryView<MemorySegment> denseHidden, denseHidden2;
        final MemoryView<MemorySegment> moeRouter, moeSelection, moeGather, moeDown;
        final MemoryView<MemorySegment> moeHidden, moeHidden2;
        final MemoryView<MemorySegment> sharedHidden, sharedHidden2, sharedOut;
        final MemoryView<MemorySegment> batchK, batchV;
        final MemoryView<MemorySegment>[] keyCache, valueCache;
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
            if (contextCapacity <= 0 || contextCapacity > c.contextLength)
                throw new IllegalArgumentException("invalid context capacity " + contextCapacity);
            int rows = batchCapacity(), dim = c.embeddingLength, kvDim = c.kvDim();
            residual = Views.allocateF32(memoryArena(), rows, dim);
            normed = Views.allocateF32(memoryArena(), rows, dim);
            branch = Views.allocateF32(memoryArena(), rows, dim);
            logits = Views.allocateF32(memoryArena(), 1, c.vocabularySize);
            query = Views.allocateF32(memoryArena(), rows, c.maxQueryDim());
            attentionGate = Views.allocateF32(memoryArena(), rows, c.maxHeadCount());
            attentionOut = Views.allocateF32(memoryArena(), rows, c.maxQueryDim());
            ropeCosFull = Views.allocateF32(memoryArena(), rows, c.ropeDimensionCount / 2);
            ropeSinFull = Views.allocateF32(memoryArena(), rows, c.ropeDimensionCount / 2);
            ropeCosSwa =
                    c.hasSwa()
                            ? Views.allocateF32(memoryArena(), rows, c.ropeDimensionCountSwa / 2)
                            : null;
            ropeSinSwa =
                    c.hasSwa()
                            ? Views.allocateF32(memoryArena(), rows, c.ropeDimensionCountSwa / 2)
                            : null;
            denseHidden = Views.allocateF32(memoryArena(), rows, c.feedForwardLength);
            denseHidden2 = Views.allocateF32(memoryArena(), rows, c.feedForwardLength);
            moeRouter = Views.allocateF32(memoryArena(), rows, c.expertCount);
            moeSelection = Views.allocateF32(memoryArena(), rows, c.expertCount);
            moeGather = Views.allocateF32(memoryArena(), rows, dim);
            moeDown = Views.allocateF32(memoryArena(), rows, dim);
            moeHidden = Views.allocateF32(memoryArena(), rows, c.expertFeedForwardLength);
            moeHidden2 = Views.allocateF32(memoryArena(), rows, c.expertFeedForwardLength);
            sharedHidden = Views.allocateF32(memoryArena(), rows, c.sharedFeedForwardLength);
            sharedHidden2 = Views.allocateF32(memoryArena(), rows, c.sharedFeedForwardLength);
            sharedOut = Views.allocateF32(memoryArena(), rows, dim);
            moeExpertCounts = new int[c.expertCount];
            moeRowTopE = new int[rows * c.expertUsedCount];
            moeRowTopP = new float[rows * c.expertUsedCount];
            moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts);
            moeRouting.topK = c.expertUsedCount;
            moeRouting.numExperts = c.expertCount;
            keyCache = new MemoryView[c.numberOfLayers];
            valueCache = new MemoryView[c.numberOfLayers];
            batchK = Views.allocateF32(memoryArena(), rows, kvDim);
            batchV = Views.allocateF32(memoryArena(), rows, kvDim);
            for (int layer = 0; layer < c.numberOfLayers; layer++) {
                int positions = c.kvCachePositions(layer, contextCapacity);
                keyCache[layer] = Views.allocateF16(memoryArena(), positions, kvDim);
                valueCache[layer] = Views.allocateF16(memoryArena(), positions, kvDim);
            }
        }

        @Override
        protected void clearHistory() {}

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }
    }

    public static Laguna loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(channel, gguf, arena);
        }
    }

    public static Laguna loadModel(FileChannel channel, GGUF gguf, Arena arena) throws IOException {
        return loadModel(channel, gguf, arena, null);
    }

    public static Laguna loadModel(FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        String arch = gguf.getString("general.architecture");
        if (!"laguna".equals(arch))
            throw new IllegalArgumentException("unsupported architecture: " + arch);
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration c = loadConfiguration(gguf, tokenizer.vocabulary().size(), arch);
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new Laguna(c, tokenizer, loadWeights(tensors, c));
    }

    static Configuration loadConfiguration(GGUF gguf, int vocabularySize, String arch) {
        int layers = gguf.getValue(int.class, arch + ".block_count");
        int dim = gguf.getValue(int.class, arch + ".embedding_length");
        int[] heads =
                scalarOrArray(gguf.getValue(Object.class, arch + ".attention.head_count"), layers);
        int kvHeads = gguf.getValue(int.class, arch + ".attention.head_count_kv");
        int headSize =
                gguf.getValueOrDefault(int.class, arch + ".attention.key_length", dim / heads[0]);
        int valueSize =
                gguf.getValueOrDefault(int.class, arch + ".attention.value_length", headSize);
        int window = gguf.getValueOrDefault(int.class, arch + ".attention.sliding_window", 0);
        int swaPeriod =
                gguf.getValueOrDefault(int.class, arch + ".attention.sliding_window_pattern", 4);
        boolean[] swa = new boolean[layers];
        if (window > 0) {
            require(swaPeriod >= 0, "sliding-window pattern must not be negative");
            for (int layer = 0; layer < layers; layer++)
                swa[layer] = swaPeriod == 0 || layer % swaPeriod != 0;
        }
        int ropeDim = gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);
        int ropeDimSwa =
                gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count_swa", ropeDim);
        Configuration c =
                new Configuration(
                        dim,
                        layers,
                        heads,
                        kvHeads,
                        headSize,
                        vocabularySize,
                        gguf.getValue(int.class, arch + ".context_length"),
                        gguf.getValueOrDefault(
                                float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-6f),
                        gguf.getValue(int.class, arch + ".feed_forward_length"),
                        gguf.getValue(int.class, arch + ".leading_dense_block_count"),
                        gguf.getValue(int.class, arch + ".expert_count"),
                        gguf.getValue(int.class, arch + ".expert_used_count"),
                        gguf.getValue(int.class, arch + ".expert_feed_forward_length"),
                        gguf.getValue(int.class, arch + ".expert_shared_feed_forward_length"),
                        gguf.getValueOrDefault(float.class, arch + ".expert_weights_scale", 1f),
                        window,
                        swa,
                        ropeDim,
                        ropeDimSwa,
                        gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10_000f),
                        gguf.getValueOrDefault(float.class, arch + ".rope.freq_base_swa", 10_000f),
                        gguf.getValueOrDefault(float.class, arch + ".rope.scaling.factor", 1f),
                        gguf.getValueOrDefault(
                                int.class,
                                arch + ".rope.scaling.original_context_length",
                                gguf.getValue(int.class, arch + ".context_length")),
                        gguf.getValueOrDefault(
                                float.class, arch + ".rope.scaling.yarn_beta_fast", 32f),
                        gguf.getValueOrDefault(
                                float.class, arch + ".rope.scaling.yarn_beta_slow", 1f),
                        gguf.getValueOrDefault(
                                float.class, arch + ".rope.scaling.yarn_attn_factor", 1f));
        validate(c, valueSize);
        require(
                gguf.getValueOrDefault(int.class, arch + ".expert_gating_func", SIGMOID_GATING)
                        == SIGMOID_GATING,
                "only sigmoid expert gating is supported");
        require(
                gguf.getValueOrDefault(boolean.class, arch + ".expert_weights_norm", true),
                "expert weights must be normalized");
        require(
                gguf.getValueOrDefault(int.class, arch + ".expert_shared_count", 1) == 1,
                "exactly one shared expert is supported");
        require(
                gguf.getValueOrDefault(int.class, arch + ".vocab_size", vocabularySize)
                        == vocabularySize,
                "tokenizer vocabulary does not match the model");
        String scaling = gguf.getValueOrDefault(String.class, arch + ".rope.scaling.type", "none");
        require(
                scaling.equals("none") || scaling.equals("yarn"),
                "unsupported RoPE scaling " + scaling);
        return c;
    }

    private static void validate(Configuration c, int valueSize) {
        require(
                c.embeddingLength > 0
                        && c.numberOfLayers > 0
                        && c.headCount.length == c.numberOfLayers
                        && c.isSwa.length == c.numberOfLayers
                        && c.keyValueHeadCount > 0
                        && c.headSize > 0
                        && c.vocabularySize > 0
                        && c.contextLength > 0,
                "invalid core dimensions");
        for (int heads : c.headCount)
            require(heads > 0 && heads % c.keyValueHeadCount == 0, "invalid per-layer head count");
        require(valueSize == c.headSize, "different key/value head sizes are unsupported");
        require(
                c.ropeDimensionCount > 0
                        && (c.ropeDimensionCount & 1) == 0
                        && c.ropeDimensionCount <= c.headSize
                        && c.ropeDimensionCountSwa > 0
                        && (c.ropeDimensionCountSwa & 1) == 0
                        && c.ropeDimensionCountSwa <= c.headSize,
                "invalid RoPE dimensions");
        require(
                c.rmsNormEps > 0f
                        && Float.isFinite(c.rmsNormEps)
                        && c.ropeTheta > 0
                        && c.ropeThetaSwa > 0
                        && c.ropeScalingFactor > 0f
                        && c.ropeOriginalContext > 0,
                "invalid normalization or RoPE metadata");
        require(
                c.feedForwardLength > 0
                        && c.denseLeadingLayers >= 0
                        && c.denseLeadingLayers <= c.numberOfLayers
                        && c.expertCount > 0
                        && c.expertUsedCount > 0
                        && c.expertUsedCount <= c.expertCount
                        && c.expertFeedForwardLength > 0
                        && c.sharedFeedForwardLength > 0
                        && c.expertWeightsScale > 0f
                        && Float.isFinite(c.expertWeightsScale),
                "invalid FFN or MoE metadata");
        if (c.hasSwa())
            require(
                    Integer.bitCount(c.slidingWindow) == 1,
                    "sliding window must be a power of two");
    }

    private static int[] scalarOrArray(Object value, int layers) {
        if (value instanceof int[] array) {
            require(array.length == layers, "head_count must have one value per layer");
            return array.clone();
        }
        int[] array = new int[layers];
        Arrays.fill(array, ((Number) value).intValue());
        return array;
    }

    @SuppressWarnings("unchecked")
    static Weights loadWeights(Map<String, MemoryView<MemorySegment>> tensors, Configuration c) {
        LayerWeights[] layers = new LayerWeights[c.numberOfLayers];
        for (int layer = 0; layer < layers.length; layer++) {
            String p = "blk." + layer + ".";
            DenseWeights dense =
                    layer < c.denseLeadingLayers
                            ? new DenseWeights(
                                    ModelLoader.require(tensors, p + "ffn_gate.weight"),
                                    ModelLoader.require(tensors, p + "ffn_up.weight"),
                                    ModelLoader.require(tensors, p + "ffn_down.weight"))
                            : null;
            MoeWeights moe =
                    layer >= c.denseLeadingLayers
                            ? new MoeWeights(
                                    ModelLoader.require(tensors, p + "ffn_gate_inp.weight"),
                                    ModelLoader.requireF32(tensors, p + "exp_probs_b.bias"),
                                    Views.sliceLeadingAxis(
                                            ModelLoader.require(
                                                    tensors, p + "ffn_gate_exps.weight"),
                                            c.expertCount),
                                    Views.sliceLeadingAxis(
                                            ModelLoader.require(tensors, p + "ffn_up_exps.weight"),
                                            c.expertCount),
                                    Views.sliceLeadingAxis(
                                            ModelLoader.require(
                                                    tensors, p + "ffn_down_exps.weight"),
                                            c.expertCount),
                                    ModelLoader.require(tensors, p + "ffn_gate_shexp.weight"),
                                    ModelLoader.require(tensors, p + "ffn_up_shexp.weight"),
                                    ModelLoader.require(tensors, p + "ffn_down_shexp.weight"))
                            : null;
            layers[layer] =
                    new LayerWeights(
                            ModelLoader.requireF32(tensors, p + "attn_norm.weight"),
                            ModelLoader.require(tensors, p + "attn_q.weight"),
                            ModelLoader.require(tensors, p + "attn_k.weight"),
                            ModelLoader.require(tensors, p + "attn_v.weight"),
                            ModelLoader.requireF32(tensors, p + "attn_q_norm.weight"),
                            ModelLoader.requireF32(tensors, p + "attn_k_norm.weight"),
                            perHeadGate(tensors, p + "attn_gate.weight", c, layer),
                            ModelLoader.require(tensors, p + "attn_output.weight"),
                            ModelLoader.requireF32(tensors, p + "ffn_norm.weight"),
                            dense,
                            moe);
        }
        MemoryView<MemorySegment> embedding = ModelLoader.require(tensors, "token_embd.weight");
        RoPE.Schedule full =
                RoPE.yarn(
                        c.ropeDimensionCount,
                        c.ropeTheta,
                        c.ropeScalingFactor,
                        c.ropeOriginalContext,
                        c.ropeBetaFast,
                        c.ropeBetaSlow,
                        1f,
                        yarnKernelAttentionFactor(c.ropeScalingFactor, c.ropeAttentionFactor));
        RoPE.Schedule swa = c.hasSwa() ? RoPE.plain(c.ropeDimensionCountSwa, c.ropeThetaSwa) : full;
        return new Weights(
                embedding,
                ModelLoader.requireF32(tensors, "output_norm.weight"),
                ModelLoader.find(tensors, "output.weight").orElse(embedding),
                layers,
                full,
                swa);
    }

    static float yarnKernelAttentionFactor(float scalingFactor, float attentionFactor) {
        // Laguna stores the final YaRN magnitude. RoPE.yarn mirrors ggml and applies mscale
        // internally, so pass only the bare multiplier or mscale would be applied twice.
        float mscale = scalingFactor <= 1f ? 1f : (float) (1.0 + 0.1 * Math.log(scalingFactor));
        return attentionFactor / mscale;
    }

    private static MemoryView<MemorySegment> perHeadGate(
            Map<String, MemoryView<MemorySegment>> tensors,
            String name,
            Configuration c,
            int layer) {
        MemoryView<MemorySegment> gate = ModelLoader.require(tensors, name);
        Shape actual = gate.dataType().logicalShape(gate.shape());
        Shape expected = Shape.flat(c.headCount[layer], c.embeddingLength);
        require(
                actual.equals(expected),
                "only per-head attention gates are supported; "
                        + name
                        + " expected "
                        + expected
                        + " but was "
                        + actual);
        return gate;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Laguna: " + message);
    }
}
