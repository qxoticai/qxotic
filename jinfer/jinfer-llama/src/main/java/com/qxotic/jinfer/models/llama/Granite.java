// Dense Granite transformer with metadata-defined embedding, residual, logit, and attention
// scales. Text-only with interleaved RoPE, an F16 KV cache, and eager full-layer evaluation.
package com.qxotic.jinfer.models.llama;

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

public final class Granite
        implements LanguageModel<Granite.Configuration, Granite.Weights, Granite.State> {

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    Granite(Configuration configuration, Tokenizer tokenizer, Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
    }

    @Override
    public Configuration configuration() {
        return configuration;
    }

    @Override
    public Optional<CheckpointCodec<State>> checkpointCodec() {
        // uniform full attention: per-position K/V rows resume on their own
        return Optional.of(new GraniteCheckpointCodec(configuration));
    }

    @Override
    public Weights weights() {
        return weights;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
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
    public void ingest(State s, Batch batch) {
        s.exclusively(() -> forward(s, batch));
        Reference.reachabilityFence(this);
    }

    private void forward(State s, Batch batch) {
        int n = batch.count();
        if (n <= 0) throw new IllegalArgumentException("Granite token batch must not be empty");
        if (n > s.batchCapacity()) {
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + s.batchCapacity());
        }
        int from = s.position();
        if (from + n > s.contextCapacity()) {
            throw new IllegalArgumentException(
                    "ingest of "
                            + n
                            + " at position "
                            + from
                            + " exceeds contextCapacity "
                            + s.contextCapacity());
        }
        int[] ids =
                switch (batch.input()) {
                    case Batch.Input.Tokens t -> t.ids();
                    case Batch.Input.Sequences seq ->
                            throw new UnsupportedOperationException(
                                    "Granite is generative: packed sequences (batched embedding)"
                                            + " not supported");
                    case Batch.Input.Embeddings ignored ->
                            throw new UnsupportedOperationException(
                                    "Granite is text-only: embedding input is not supported");
                };
        for (int id : ids)
            if (id < 0 || id >= configuration.vocabularySize)
                throw new IllegalArgumentException(
                        "token id " + id + " outside [0," + configuration.vocabularySize + ")");
        if (n == 1)
            Parallel.runDecodeStep(
                    () -> {
                        forward(s, ids, from, n);
                        return null;
                    });
        else forward(s, ids, from, n);
        s.advance(batch);
    }

    @Override
    public MemoryView<?> logits(State s, int output) {
        MemoryView<?> result = s.exclusively(() -> projectLogits(s, output));
        Reference.reachabilityFence(this);
        return result;
    }

    private MemoryView<?> projectLogits(State s, int output) {
        if (output < 0 || output >= s.outputCount())
            throw new IllegalArgumentException("output " + output + " outside retained outputs");
        int dim = configuration.embeddingLength;
        int row = s.lastBatchSize() - s.outputCount() + output;
        return Parallel.runDecodeStep(
                () -> {
                    Norms.rmsnorm(
                            s.normed,
                            0,
                            s.residual,
                            (long) row * dim,
                            weights.finalNorm(),
                            dim,
                            configuration.rmsNormEps);
                    MatMul.gemv(
                            weights.wcls(), s.normed, s.logits, configuration.vocabularySize, dim);
                    float ls = configuration.logitScale;
                    if (ls != 1.0f) {
                        Ops.divideInPlace(s.logits, 0, configuration.vocabularySize, ls);
                    }
                    return s.logits;
                });
    }

    // === Forward ===

    private void forward(State state, int[] tokens, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        // ONCE for the batch: an angle never depends on the layer, so all of them read these rows
        RoPE.fill(state.ropeCos, state.ropeSin, startPos, seqLen, config.ropeHalf(), w.rope());
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;
        float embScale = config.embeddingScale, residScale = config.residualScale;

        Views.checkAlive(w.tokenEmbeddings(), "tokenEmbeddings"); // fail-fast on freed weights
        Convert.gatherToF32(w.tokenEmbeddings(), tokens, 0, seqLen, state.residual, 0, dim);
        if (embScale != 1.0f) {
            Ops.multiplyInPlace(state.residual, 0, Math.multiplyExact(seqLen, dim), embScale);
        }

        for (int l = 0; l < config.numberOfLayers; l++) {
            LayerWeights lw = w.layers()[l];
            Norms.rmsnormRows(state.normed, state.residual, lw.attnNorm(), seqLen, dim, eps);
            attention(state, l, startPos, seqLen);
            Ops.addScaled(state.residual, state.normed, seqLen * dim, residScale);
            Norms.rmsnormRows(state.normed, state.residual, lw.ffnNorm(), seqLen, dim, eps);
            feedForward(state, l, seqLen);
            Ops.addScaled(state.residual, state.normed, seqLen * dim, residScale);
            if (Trace.ENABLED) {
                Trace.sum("l_out-" + l, state.residual, seqLen * dim);
            }
        }
    }

    /**
     * Standard RoPE GQA attention with Granite's custom attention scale: Q/K/V projections, per-row
     * interleaved RoPE, K/V committed to the cache, causal flash attention (or single-token
     * decode), output projection back to {@code state.normed}. (No attention-temperature tuning -
     * that's mistral3.)
     */
    private void attention(State state, int layer, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        LayerWeights lw = w.layers()[layer];
        int dim = config.embeddingLength;
        int headSize = config.headSize;
        int heads = config.numberOfHeads;
        int kvHeads = config.numberOfKeyValueHeads;
        int kvDim = config.kvDim();
        int queryDim = config.queryDim();
        int kvMul = heads / kvHeads;
        int ropeHalf = config.ropeHalf();

        MatMul.gemm(lw.wq(), state.normed, dim, state.query, queryDim, queryDim, seqLen, dim);
        MatMul.gemm(lw.wk(), state.normed, dim, state.batchK, kvDim, kvDim, seqLen, dim);
        MatMul.gemm(lw.wv(), state.normed, dim, state.batchV, kvDim, kvDim, seqLen, dim);
        Parallel.forLoop(
                seqLen,
                s -> {
                    for (int h = 0; h < heads; h++) {
                        RoPE.applyInterleaved(
                                state.query,
                                (long) s * queryDim + h * headSize,
                                s,
                                state.ropeCos,
                                state.ropeSin,
                                ropeHalf);
                    }
                    for (int h = 0; h < kvHeads; h++) {
                        RoPE.applyInterleaved(
                                state.batchK,
                                (long) s * kvDim + h * headSize,
                                s,
                                state.ropeCos,
                                state.ropeSin,
                                ropeHalf);
                    }
                });

        MemoryView<MemorySegment> keyCache = state.keyCache[layer],
                valueCache = state.valueCache[layer];
        float attScale = config.attentionScale();
        if (seqLen == 1) {
            FlashAttention.flashDecode(
                    state.query,
                    state.attnOut,
                    keyCache,
                    valueCache,
                    state.batchK,
                    state.batchV,
                    heads,
                    startPos,
                    0,
                    headSize,
                    kvDim,
                    kvMul,
                    attScale,
                    0,
                    null,
                    state.decodeScratch);
        } else {
            FlashAttention.slidingWindowPrefill(
                    state.query,
                    state.attnOut,
                    keyCache,
                    valueCache,
                    state.batchK,
                    state.batchV,
                    heads,
                    startPos,
                    seqLen,
                    headSize,
                    kvDim,
                    queryDim,
                    kvDim,
                    kvMul,
                    attScale,
                    0,
                    0,
                    null);
        }
        // commit this chunk's K/V INSIDE the layer: the batch buffers are shared by every layer
        // (the Qwen 3 lesson: a deferred commit writes the LAST layer's values everywhere)
        commitKv(state, layer, startPos, seqLen);
        MatMul.gemm(lw.wo(), state.attnOut, queryDim, state.normed, dim, dim, seqLen, queryDim);
    }

    /**
     * Commit this chunk's F32 K/V (state.batchK/batchV) into the F16 cache at {@code [startPos,
     * startPos+seqLen)}.
     */
    private void commitKv(State state, int l, int startPos, int seqLen) {
        int kvDim = configuration.kvDim();
        int count = Math.multiplyExact(seqLen, kvDim);
        long cacheOffset = Math.multiplyExact((long) startPos, kvDim);
        Convert.f32ToF16(state.batchK, 0, state.keyCache[l], cacheOffset, count);
        Convert.f32ToF16(state.batchV, 0, state.valueCache[l], cacheOffset, count);
    }

    /** Dense SwiGLU FFN over the pre-normed rows in {@code state.normed}, written back in place. */
    private void feedForward(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        LayerWeights lw = weights.layers()[l];
        MatMul.gemm(lw.w1(), state.normed, dim, state.hidden, hiddenDim, hiddenDim, seqLen, dim);
        MatMul.gemm(lw.w3(), state.normed, dim, state.hidden2, hiddenDim, hiddenDim, seqLen, dim);
        Activations.siluMultiply(
                state.hidden, 0, state.hidden2, 0, Math.multiplyExact(seqLen, hiddenDim));
        MatMul.gemm(lw.w2(), state.hidden, hiddenDim, state.normed, dim, dim, seqLen, hiddenDim);
    }

    // === Configuration ===

    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int headSize,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            float ropeTheta,
            int ropeDimensionCount,
            int hiddenDim,
            float embeddingScale,
            float residualScale,
            float logitScale,
            float attentionScaleValue)
            implements ContextConfiguration {
        public int queryDim() {
            return numberOfHeads * headSize;
        }

        public int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        public int ropeHalf() {
            return Math.min(ropeDimensionCount, headSize) / 2;
        }

        /**
         * Granite replaces the default 1/sqrt(headSize) with a metadata-supplied attention scale.
         */
        public float attentionScale() {
            return attentionScaleValue != 0f
                    ? attentionScaleValue
                    : 1.0f / (float) Math.sqrt(headSize);
        }
    }

    // === Weights ===

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> wq,
            MemoryView<MemorySegment> wk,
            MemoryView<MemorySegment> wv,
            MemoryView<MemorySegment> wo,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> w1,
            MemoryView<MemorySegment> w2,
            MemoryView<MemorySegment> w3) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbeddings,
            LayerWeights[] layers,
            MemoryView<MemorySegment> finalNorm,
            RoPE.Schedule rope,
            MemoryView<MemorySegment> wcls) {}

    // === State ===

    public static final class State extends ContextState {

        /** The residual stream every block adds back into. */
        final MemoryView<MemorySegment> residual;

        /** Pre-norm output - the input of EVERY projection; second life as the FFN down dest. */
        final MemoryView<MemorySegment> normed;

        /** Q projection (roped in place per row). */
        final MemoryView<MemorySegment> query;

        /** Flash-attention result, all heads concatenated, pre-o_proj. */
        final MemoryView<MemorySegment> attnOut;

        /** FFN gate projection; post silu-multiply the gated hidden. */
        final MemoryView<MemorySegment> hidden;

        /** FFN up projection (the multiplicand consumed by siluMultiply). */
        final MemoryView<MemorySegment> hidden2;

        /** The LM head's output buffer. */
        final MemoryView<MemorySegment> logits;

        final MemoryView<MemorySegment> ropeCos, ropeSin;
        final FlashAttention.DecodeScratch decodeScratch;
        final MemoryView<MemorySegment> batchK, batchV; // this chunk's K/V (uniform kvDim)
        final MemoryView<MemorySegment>[] keyCache, valueCache; // per layer, F16

        /** Recycles this allocation: cursor to 0; stale KV rows beyond it are attention-masked. */
        @Override
        protected void clearHistory() {}

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }

        @SuppressWarnings("unchecked")
        State(
                Configuration config,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity <= 0 || contextCapacity > config.contextLength()) {
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " outside [1,"
                                + config.contextLength()
                                + "]");
            }
            if (batchCapacity <= 0)
                throw new IllegalArgumentException("batchCapacity " + batchCapacity);
            int c = batchCapacity;
            int dim = config.embeddingLength;
            int queryDim = config.queryDim();
            int kvDim = config.kvDim();
            int hidden = config.hiddenDim;
            this.residual = Views.allocateF32(memoryArena(), c * dim);
            this.normed = Views.allocateF32(memoryArena(), c * dim);
            this.batchK = Views.allocateF32(memoryArena(), c * kvDim);
            this.batchV = Views.allocateF32(memoryArena(), c * kvDim);
            this.query = Views.allocateF32(memoryArena(), c * queryDim);
            this.attnOut = Views.allocateF32(memoryArena(), c * queryDim);
            this.hidden = Views.allocateF32(memoryArena(), c * hidden);
            this.hidden2 = Views.allocateF32(memoryArena(), c * hidden);
            this.logits = Views.allocateF32(memoryArena(), config.vocabularySize);
            // rotary values for the batch about to be ingested: sized by BATCH, never context
            this.ropeCos = Views.allocateF32(memoryArena(), c * config.ropeHalf());
            this.ropeSin = Views.allocateF32(memoryArena(), c * config.ropeHalf());
            this.decodeScratch = new FlashAttention.DecodeScratch(memoryArena());
            this.keyCache = new MemoryView[config.numberOfLayers];
            this.valueCache = new MemoryView[config.numberOfLayers];
            for (int l = 0; l < config.numberOfLayers; l++) {
                keyCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
                valueCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
            }
        }
    }

    // === Loading ===

    public static Granite loadModel(Path ggufPath, Arena arena) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, arena);
        }
    }

    public static Granite loadModel(FileChannel fileChannel, GGUF gguf, Arena arena)
            throws IOException {
        return loadModel(fileChannel, gguf, arena, null);
    }

    /**
     * As above with a caller-supplied tokenizer; null = the GGUF's own (the differential gate feeds
     * ONE tokenizer instance to both trees).
     */
    public static Granite loadModel(
            FileChannel fileChannel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        if (tokenizer == null) {
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        }
        String arch = gguf.getString("general.architecture"); // "granite"

        int contextLength = gguf.getValue(int.class, arch + ".context_length");
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int numberOfKeyValueHeads =
                gguf.getValueOrDefault(int.class, arch + ".attention.head_count_kv", numberOfHeads);
        int headSize =
                gguf.getValueOrDefault(
                        int.class, arch + ".attention.key_length", embeddingLength / numberOfHeads);
        int hiddenDim = gguf.getValue(int.class, arch + ".feed_forward_length");
        float rmsNormEps =
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10000f);
        int ropeDimensionCount =
                gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);

        // Granite's four scalars (default 1.0 / off -> plain Llama, but Granite supplies real
        // values).
        float embeddingScale = gguf.getValueOrDefault(float.class, arch + ".embedding_scale", 1.0f);
        float residualScale = gguf.getValueOrDefault(float.class, arch + ".residual_scale", 1.0f);
        float logitScale = gguf.getValueOrDefault(float.class, arch + ".logit_scale", 1.0f);
        float attentionScale = gguf.getValueOrDefault(float.class, arch + ".attention.scale", 0f);

        Configuration config =
                new Configuration(
                        embeddingLength,
                        numberOfLayers,
                        numberOfHeads,
                        numberOfKeyValueHeads,
                        headSize,
                        tokenizer.vocabulary().size(),
                        contextLength,
                        rmsNormEps,
                        ropeTheta,
                        ropeDimensionCount,
                        hiddenDim,
                        embeddingScale,
                        residualScale,
                        logitScale,
                        attentionScale);

        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(fileChannel, gguf, arena);
        RoPE.Schedule rope = buildRope(config, tensors);
        return new Granite(config, tokenizer, loadWeights(tensors, config, rope));
    }

    /**
     * Plain RoPE for granite (freq base + dimension count); honors a rope_freqs.weight
     * per-frequency scaling tensor if present. No YaRN.
     */
    static RoPE.Schedule buildRope(
            Configuration config, Map<String, MemoryView<MemorySegment>> tensors) {
        int ropeDim = Math.min(config.ropeDimensionCount, config.headSize);
        return ModelLoader.ropeFreqFactors(tensors)
                .map(freqs -> RoPE.withFreqFactors(ropeDim, config.ropeTheta, freqs))
                .orElseGet(() -> RoPE.plain(ropeDim, config.ropeTheta));
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors,
            Configuration config,
            RoPE.Schedule rope) {
        int n = config.numberOfLayers;
        MemoryView<MemorySegment> tokenEmbeddings =
                ModelLoader.require(tensors, "token_embd.weight");
        MemoryView<MemorySegment> wcls =
                ModelLoader.find(tensors, "output.weight").orElse(tokenEmbeddings);
        MemoryView<MemorySegment> finalNorm = ModelLoader.requireF32(tensors, "output_norm.weight");

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            layers[i] =
                    new LayerWeights(
                            ModelLoader.requireF32(tensors, p + "attn_norm.weight"),
                            ModelLoader.require(tensors, p + "attn_q.weight"),
                            ModelLoader.require(tensors, p + "attn_k.weight"),
                            ModelLoader.require(tensors, p + "attn_v.weight"),
                            ModelLoader.require(tensors, p + "attn_output.weight"),
                            ModelLoader.requireF32(tensors, p + "ffn_norm.weight"),
                            ModelLoader.require(tensors, p + "ffn_gate.weight"),
                            ModelLoader.require(tensors, p + "ffn_down.weight"),
                            ModelLoader.require(tensors, p + "ffn_up.weight"));
        }
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls);
    }
}
