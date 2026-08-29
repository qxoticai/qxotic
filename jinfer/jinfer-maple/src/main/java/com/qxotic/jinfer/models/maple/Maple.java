package com.qxotic.jinfer.models.maple;

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

/** Maple sparse-transformer inference against the MemoryView boundary. */
public final class Maple implements LanguageModel<Maple.Configuration, Maple.Weights, Maple.State> {
    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    Maple(Configuration configuration, Tokenizer tokenizer, Weights weights) {
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
        return Optional.of(new MapleCheckpointCodec(configuration));
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
        if (rows <= 0) throw new IllegalArgumentException("batch must not be empty");
        if (rows > state.batchCapacity())
            throw new IllegalArgumentException(
                    "batch " + rows + " exceeds batchCapacity " + state.batchCapacity());
        int startPos = state.position();
        if (startPos + rows > state.contextCapacity())
            throw new IllegalArgumentException(
                    "ingest exceeds contextCapacity " + state.contextCapacity());
        switch (batch.input()) {
            case Batch.Input.Tokens tokens -> {
                for (int id : tokens.ids())
                    if (id < 0 || id >= configuration.vocabularySize)
                        throw new IllegalArgumentException("token id out of range: " + id);
                if (rows == 1) {
                    forward(state, tokens.ids(), startPos, rows);
                } else forward(state, tokens.ids(), startPos, rows);
            }
            case Batch.Input.Sequences ignored ->
                    throw new UnsupportedOperationException(
                            "Maple does not support packed sequences");
            case Batch.Input.Embeddings ignored ->
                    throw new UnsupportedOperationException("Maple is text-only");
        }
        state.advance(batch);
    }

    private void forward(State state, int[] tokens, int startPos, int rows) {
        RoPE.fill(
                state.ropeCos,
                state.ropeSin,
                startPos,
                rows,
                configuration.ropeDimension / 2,
                weights.rope);
        embed(state, tokens, rows);
        for (int layer = 0; layer < configuration.numberOfLayers; layer++) {
            attention(state, layer, startPos, rows);
            feedForward(state, layer, rows);
            if (Trace.ENABLED)
                Trace.sum("l_out-" + layer, state.residual, rows * configuration.embeddingLength);
        }
    }

    private void embed(State state, int[] tokens, int rows) {
        int dim = configuration.embeddingLength;
        Views.checkAlive(weights.tokenEmbeddings, "tokenEmbeddings");
        for (int row = 0; row < rows; row++)
            Convert.copyToF32(
                    weights.tokenEmbeddings,
                    (long) tokens[row] * dim,
                    state.residual,
                    (long) row * dim,
                    dim);
    }

    private void attention(State state, int layer, int startPos, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[layer];
        int dim = c.embeddingLength, head = c.headSize;
        int qDim = c.queryDim(),
                kvDim = c.kvDim(),
                kvMul = c.numberOfHeads / c.numberOfKeyValueHeads;
        boolean sliding = c.isSliding(layer);

        Norms.rmsnormRows(state.normed, state.residual, w.attnNorm, rows, dim, c.rmsNormEps);
        MatMul.gemm(w.q, state.normed, state.query, rows);
        MatMul.gemm(w.k, state.normed, state.batchK[layer], rows);
        MatMul.gemm(w.v, state.normed, state.batchV[layer], rows);

        Parallel.forLoop(
                rows,
                row -> {
                    for (int h = 0; h < c.numberOfHeads; h++) {
                        long off = (long) row * qDim + (long) h * head;
                        Norms.rmsnorm(
                                state.query, off, state.query, off, w.qNorm, head, c.rmsNormEps);
                        if (sliding)
                            RoPE.applyNeox(
                                    state.query,
                                    off,
                                    row,
                                    state.ropeCos,
                                    state.ropeSin,
                                    c.ropeDimension / 2);
                    }
                    for (int h = 0; h < c.numberOfKeyValueHeads; h++) {
                        long off = (long) row * kvDim + (long) h * head;
                        Norms.rmsnorm(
                                state.batchK[layer],
                                off,
                                state.batchK[layer],
                                off,
                                w.kNorm,
                                head,
                                c.rmsNormEps);
                        if (sliding)
                            RoPE.applyNeox(
                                    state.batchK[layer],
                                    off,
                                    row,
                                    state.ropeCos,
                                    state.ropeSin,
                                    c.ropeDimension / 2);
                    }
                });

        float scale = 1f / (float) Math.sqrt(head);
        if (rows > 1)
            FlashAttention.slidingWindowPrefill(
                    state.query,
                    state.attnOut,
                    state.keyCache[layer],
                    state.valueCache[layer],
                    state.batchK[layer],
                    state.batchV[layer],
                    c.numberOfHeads,
                    startPos,
                    rows,
                    head,
                    kvDim,
                    qDim,
                    kvDim,
                    kvMul,
                    scale,
                    sliding ? c.slidingWindow : 0,
                    sliding ? c.slidingWindow - 1 : 0,
                    null);
        else
            FlashAttention.flashDecode(
                    state.query,
                    state.attnOut,
                    state.keyCache[layer],
                    state.valueCache[layer],
                    state.batchK[layer],
                    state.batchV[layer],
                    c.numberOfHeads,
                    startPos,
                    c.attentionStart(layer, startPos),
                    head,
                    kvDim,
                    kvMul,
                    scale,
                    sliding ? c.slidingWindow - 1 : 0,
                    null,
                    state.decodeScratch);

        MatMul.gemm(w.o, state.attnOut, state.branch, rows);
        Ops.addInPlace(state.residual, 0, state.branch, 0, rows * dim);
        commitKv(state, layer, startPos, rows);
    }

    private void commitKv(State state, int layer, int startPos, int rows) {
        int kvDim = configuration.kvDim();
        for (int row = 0; row < rows; row++) {
            long slot = configuration.kvCacheIndex(layer, startPos + row);
            Convert.f32ToF16(
                    state.batchK[layer],
                    (long) row * kvDim,
                    state.keyCache[layer],
                    slot * kvDim,
                    kvDim);
            Convert.f32ToF16(
                    state.batchV[layer],
                    (long) row * kvDim,
                    state.valueCache[layer],
                    slot * kvDim,
                    kvDim);
        }
    }

    private void feedForward(State state, int layer, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[layer];
        int dim = c.embeddingLength, experts = c.expertCount;
        int topK = c.expertUsedCount, expertFf = c.expertFeedForwardLength;

        Norms.rmsnormRows(state.normed, state.residual, w.ffnNorm, rows, dim, c.rmsNormEps);
        MatMul.gemm(w.router, state.normed, state.router, rows);
        for (int row = 0; row < rows; row++)
            Ops.softmaxInPlace(state.router, (long) row * experts, experts);
        Moe.selectTopK(
                state.router,
                rows,
                experts,
                topK,
                state.topExperts,
                state.topWeights,
                state.expertCounts);
        for (int row = 0; row < rows; row++) {
            float sum = 0f;
            for (int k = 0; k < topK; k++) sum += state.topWeights[row * topK + k];
            float scale = sum == 0f ? 0f : c.expertWeightsScale / sum;
            for (int k = 0; k < topK; k++) state.topWeights[row * topK + k] *= scale;
        }

        state.routing.seqLen = rows;
        Moe.dispatch(
                state.routing,
                dim,
                state.normed,
                state.gather,
                state.expertOut,
                state.moeOut,
                null,
                (expert, count, gather, out) -> {
                    MatMul.gemm(w.gateExperts[expert], gather, state.hidden, count);
                    MatMul.gemm(w.upExperts[expert], gather, state.hidden2, count);
                    float clamp = c.swigluClamp[layer];
                    Ops.clampInPlace(
                            state.hidden, 0, count * expertFf, Float.NEGATIVE_INFINITY, clamp);
                    Ops.clampInPlace(state.hidden2, 0, count * expertFf, -clamp, clamp);
                    Activations.siluMultiply(state.hidden, 0, state.hidden2, 0, count * expertFf);
                    MatMul.gemm(w.downExperts[expert], state.hidden, out, count);
                });
        Ops.addInPlace(state.residual, 0, state.moeOut, 0, rows * dim);
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
        {
            Norms.rmsnorm(
                    state.head,
                    0,
                    state.residual,
                    (long) row * dim,
                    weights.outputNorm,
                    dim,
                    configuration.rmsNormEps);
            MatMul.gemv(weights.outputWeight, state.head, state.logits);
            return state.logits;
        }
    }

    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int headSize,
            int ropeDimension,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            double ropeTheta,
            int slidingWindow,
            boolean[] slidingMask,
            int expertCount,
            int expertUsedCount,
            int expertFeedForwardLength,
            float expertWeightsScale,
            float[] swigluClamp)
            implements ContextConfiguration {
        public Configuration {
            if (embeddingLength <= 0
                    || numberOfLayers <= 0
                    || numberOfHeads <= 0
                    || numberOfKeyValueHeads <= 0
                    || headSize <= 0
                    || vocabularySize <= 0
                    || contextLength <= 0
                    || expertCount <= 0
                    || expertFeedForwardLength <= 0)
                throw new IllegalArgumentException("model dimensions must be positive");
            if (numberOfHeads % numberOfKeyValueHeads != 0)
                throw new IllegalArgumentException("query heads must be divisible by KV heads");
            if (ropeDimension <= 0 || ropeDimension > headSize || (ropeDimension & 1) != 0)
                throw new IllegalArgumentException("invalid ropeDimension " + ropeDimension);
            if (slidingWindow <= 0 || Integer.bitCount(slidingWindow) != 1)
                throw new IllegalArgumentException("slidingWindow must be a power of two");
            if (slidingMask == null || slidingMask.length != numberOfLayers)
                throw new IllegalArgumentException("slidingMask length must equal layer count");
            if (expertUsedCount <= 0 || expertUsedCount > expertCount)
                throw new IllegalArgumentException("invalid expertUsedCount " + expertUsedCount);
            if (swigluClamp == null || swigluClamp.length != numberOfLayers)
                throw new IllegalArgumentException("swigluClamp length must equal layer count");
        }

        int queryDim() {
            return numberOfHeads * headSize;
        }

        int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        boolean isSliding(int layer) {
            return slidingMask[layer];
        }

        int kvCachePositions(int layer, int capacity) {
            return isSliding(layer) ? Math.min(capacity, slidingWindow) : capacity;
        }

        int kvCacheIndex(int layer, int position) {
            return isSliding(layer) ? position & (slidingWindow - 1) : position;
        }

        int attentionStart(int layer, int position) {
            return isSliding(layer) ? Math.max(0, position - slidingWindow + 1) : 0;
        }
    }

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> k,
            MemoryView<MemorySegment> v,
            MemoryView<MemorySegment> qNorm,
            MemoryView<MemorySegment> kNorm,
            MemoryView<MemorySegment> o,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> router,
            MemoryView<MemorySegment>[] gateExperts,
            MemoryView<MemorySegment>[] upExperts,
            MemoryView<MemorySegment>[] downExperts) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbeddings,
            LayerWeights[] layers,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> outputWeight,
            RoPE.Schedule rope) {}

    public static final class State extends ContextState {
        final MemoryView<MemorySegment> residual, normed, branch, query, attnOut, head, logits;
        final MemoryView<MemorySegment> ropeCos, ropeSin;
        final MemoryView<MemorySegment> router, gather, expertOut, moeOut, hidden, hidden2;
        final int[] expertCounts, topExperts;
        final float[] topWeights;
        final Moe.Routing routing;
        final FlashAttention.DecodeScratch decodeScratch =
                new FlashAttention.DecodeScratch(memoryArena());
        final MemoryView<MemorySegment>[] keyCache, valueCache, batchK, batchV;

        @SuppressWarnings("unchecked")
        State(
                Configuration c,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity > c.contextLength)
                throw new IllegalArgumentException("contextCapacity exceeds model context length");
            int rows = batchCapacity(), dim = c.embeddingLength;
            int qDim = c.queryDim(), kvDim = c.kvDim();
            residual = Views.allocateF32(memoryArena(), rows, dim);
            normed = Views.allocateF32(memoryArena(), rows, dim);
            branch = Views.allocateF32(memoryArena(), rows, dim);
            query = Views.allocateF32(memoryArena(), rows, qDim);
            attnOut = Views.allocateF32(memoryArena(), rows, qDim);
            head = Views.allocateF32(memoryArena(), 1, dim);
            logits = Views.allocateF32(memoryArena(), 1, c.vocabularySize);
            ropeCos = Views.allocateF32(memoryArena(), rows, c.ropeDimension / 2);
            ropeSin = Views.allocateF32(memoryArena(), rows, c.ropeDimension / 2);
            router = Views.allocateF32(memoryArena(), rows, c.expertCount);
            gather = Views.allocateF32(memoryArena(), rows, dim);
            expertOut = Views.allocateF32(memoryArena(), rows, dim);
            moeOut = Views.allocateF32(memoryArena(), rows, dim);
            hidden = Views.allocateF32(memoryArena(), rows, c.expertFeedForwardLength);
            hidden2 = Views.allocateF32(memoryArena(), rows, c.expertFeedForwardLength);
            expertCounts = new int[c.expertCount];
            topExperts = new int[rows * c.expertUsedCount];
            topWeights = new float[rows * c.expertUsedCount];
            routing = new Moe.Routing(topExperts, topWeights, expertCounts);
            routing.topK = c.expertUsedCount;
            routing.numExperts = c.expertCount;
            keyCache = new MemoryView[c.numberOfLayers];
            valueCache = new MemoryView[c.numberOfLayers];
            batchK = new MemoryView[c.numberOfLayers];
            batchV = new MemoryView[c.numberOfLayers];
            for (int layer = 0; layer < c.numberOfLayers; layer++) {
                int positions = c.kvCachePositions(layer, contextCapacity);
                keyCache[layer] = Views.allocateF16(memoryArena(), positions, kvDim);
                valueCache[layer] = Views.allocateF16(memoryArena(), positions, kvDim);
                batchK[layer] = Views.allocateF32(memoryArena(), rows, kvDim);
                batchV[layer] = Views.allocateF32(memoryArena(), rows, kvDim);
            }
        }

        @Override
        protected void clearHistory() {}

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }
    }

    public static Maple loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(channel, gguf, arena);
        }
    }

    public static Maple loadModel(FileChannel channel, GGUF gguf, Arena arena) throws IOException {
        return loadModel(channel, gguf, arena, null);
    }

    public static Maple loadModel(FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration configuration = loadConfiguration(gguf, tokenizer);
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new Maple(configuration, tokenizer, loadWeights(tensors, configuration));
    }

    static Configuration loadConfiguration(GGUF gguf, Tokenizer tokenizer) {
        String arch = "maple";
        int layers = gguf.getValue(int.class, arch + ".block_count");
        int dim = gguf.getValue(int.class, arch + ".embedding_length");
        int heads = gguf.getValue(int.class, arch + ".attention.head_count");
        int keyLength = gguf.getValue(int.class, arch + ".attention.key_length");
        int valueLength = gguf.getValue(int.class, arch + ".attention.value_length");
        if (keyLength != valueLength)
            throw new IllegalArgumentException("Maple key/value head sizes differ");
        boolean normalize =
                gguf.getValueOrDefault(boolean.class, arch + ".expert_weights_norm", false);
        int gating = gguf.getValueOrDefault(int.class, arch + ".expert_gating_func", 1);
        if (!normalize || gating != 1)
            throw new IllegalArgumentException("unsupported Maple expert routing metadata");
        return new Configuration(
                dim,
                layers,
                heads,
                gguf.getValue(int.class, arch + ".attention.head_count_kv"),
                keyLength,
                gguf.getValue(int.class, arch + ".rope.dimension_count"),
                tokenizer.vocabulary().size(),
                gguf.getValue(int.class, arch + ".context_length"),
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-6f),
                gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10_000f),
                gguf.getValue(int.class, arch + ".attention.sliding_window"),
                gguf.getValue(boolean[].class, arch + ".attention.sliding_window_pattern"),
                gguf.getValue(int.class, arch + ".expert_count"),
                gguf.getValue(int.class, arch + ".expert_used_count"),
                gguf.getValue(int.class, arch + ".expert_feed_forward_length"),
                gguf.getValueOrDefault(float.class, arch + ".expert_weights_scale", 1f),
                gguf.getValue(float[].class, arch + ".swiglu_clamp_exp"));
    }

    static Weights loadWeights(Map<String, MemoryView<MemorySegment>> tensors, Configuration c) {
        LayerWeights[] layers = new LayerWeights[c.numberOfLayers];
        for (int layer = 0; layer < layers.length; layer++) {
            String p = "blk." + layer + ".";
            layers[layer] =
                    new LayerWeights(
                            ModelLoader.requireF32(tensors, p + "attn_norm.weight"),
                            ModelLoader.require(tensors, p + "attn_q.weight"),
                            ModelLoader.require(tensors, p + "attn_k.weight"),
                            ModelLoader.require(tensors, p + "attn_v.weight"),
                            ModelLoader.requireF32(tensors, p + "attn_q_norm.weight"),
                            ModelLoader.requireF32(tensors, p + "attn_k_norm.weight"),
                            ModelLoader.require(tensors, p + "attn_output.weight"),
                            ModelLoader.requireF32(tensors, p + "ffn_norm.weight"),
                            ModelLoader.requireF32(tensors, p + "ffn_gate_inp.weight"),
                            Views.sliceLeadingAxis(
                                    ModelLoader.require(tensors, p + "ffn_gate_exps.weight")),
                            Views.sliceLeadingAxis(
                                    ModelLoader.require(tensors, p + "ffn_up_exps.weight")),
                            Views.sliceLeadingAxis(
                                    ModelLoader.require(tensors, p + "ffn_down_exps.weight")));
        }
        MemoryView<MemorySegment> embeddings = ModelLoader.require(tensors, "token_embd.weight");
        return new Weights(
                embeddings,
                layers,
                ModelLoader.requireF32(tensors, "output_norm.weight"),
                ModelLoader.find(tensors, "output.weight").orElse(embeddings),
                RoPE.plain(c.ropeDimension, c.ropeTheta));
    }
}
