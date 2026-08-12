// Granite (dense) against the x boundary: a port of jinfer-llama's Granite (cycle 3). The Llama
// graph minus the lazy last-layer tail, attention temperature, NoPE and YaRN - plus Granite's
// four metadata scalars (embedding / residual / logit / attention scale). Full forward (every
// layer runs; no deferred tail), head reads the residual directly. Text-only, dense FFN. Chat
// templates, stop tokens, and the cache codec are OUT of slice (the chat/cache cycles).
package com.qxotic.jinfer.x.models.llama;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.Parallel;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.BaseState;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Config;
import com.qxotic.jinfer.x.boundary.LanguageModel;
import com.qxotic.jinfer.x.kernels.Activations;
import com.qxotic.jinfer.x.kernels.Convert;
import com.qxotic.jinfer.x.kernels.FlashAttention;
import com.qxotic.jinfer.x.kernels.MatMul;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Norms;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jinfer.x.kernels.RoPE;
import com.qxotic.jinfer.x.kernels.Trace;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;

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
    public Configuration config() {
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
    public State newState(int contextCapacity, int batchCapacity, Arena arena) {
        return new State(configuration, contextCapacity, batchCapacity, arena);
    }

    @Override
    public void forward(State s, Batch batch) {
        int n = batch.count();
        if (n > s.batchCapacity) {
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + s.batchCapacity);
        }
        int from = s.position();
        if (from + n > s.contextCapacity) {
            throw new IllegalArgumentException(
                    "ingest of "
                            + n
                            + " at position "
                            + from
                            + " exceeds contextCapacity "
                            + s.contextCapacity);
        }
        switch (batch.input()) {
            case Batch.Input.Tokens t -> {
                int[] ids = t.ids();
                if (n == 1) {
                    Parallel.onDecodePool(
                            () -> {
                                forward(s, ids, 0, from, n);
                                return null;
                            });
                } else {
                    forward(s, ids, 0, from, n);
                }
            }
            case Batch.Input.Sequences seq ->
                    throw new UnsupportedOperationException(
                            "Granite is generative: packed sequences (batched embedding) not"
                                    + " supported");
            case Batch.Input.Embeddings ignored ->
                    throw new UnsupportedOperationException(
                            "Granite is text-only: embedding input is not supported");
        }
        s.advance(n, batch.outputs());
    }

    @Override
    public MemoryView<?> head(State s, int output) {
        int dim = configuration.embeddingLength;
        int row = s.lastChunkLen - s.outputCount + output;
        return Parallel.onDecodePool(
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
                        Ops.mapInPlace(s.logits, 0, configuration.vocabularySize, v -> v / ls);
                    }
                    return s.logits;
                });
    }

    // === Forward ===

    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        // ONCE for the batch: an angle never depends on the layer, so all of them read these rows
        RoPE.fill(state.ropeCos, state.ropeSin, startPos, seqLen, config.ropeHalf(), w.rope());
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;
        float embScale = config.embeddingScale, residScale = config.residualScale;

        Views.checkAlive(w.tokenEmbeddings(), "tokenEmbeddings"); // fail-fast on freed weights
        for (int s = 0; s < seqLen; s++) {
            Convert.copyToF32(
                    w.tokenEmbeddings(),
                    (long) tokens[tokenOffset + s] * dim,
                    state.residual,
                    (long) s * dim,
                    dim);
        }
        if (embScale != 1.0f) {
            Ops.mapInPlace(state.residual, 0, seqLen * dim, v -> v * embScale);
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
     * decode), output projection back to {@code state.normed}. (No attention-temperature tuning —
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
        Parallel.forRows(
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
        // (the xqwen3 lesson: a deferred commit writes the LAST layer's values everywhere)
        commitKv(state, layer, startPos, seqLen);
        MatMul.gemm(lw.wo(), state.attnOut, queryDim, state.normed, dim, dim, seqLen, queryDim);
    }

    /**
     * Commit this chunk's F32 K/V (state.batchK/batchV) into the F16 cache at {@code [startPos,
     * startPos+seqLen)}.
     */
    private void commitKv(State state, int l, int startPos, int seqLen) {
        int kvDim = configuration.kvDim();
        for (int s = 0; s < seqLen; s++) {
            long kvPos = startPos + s;
            Convert.f32ToF16(
                    state.batchK, (long) s * kvDim, state.keyCache[l], kvPos * kvDim, kvDim);
            Convert.f32ToF16(
                    state.batchV, (long) s * kvDim, state.valueCache[l], kvPos * kvDim, kvDim);
        }
    }

    /** Dense SwiGLU FFN over the pre-normed rows in {@code state.normed}, written back in place. */
    private void feedForward(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        LayerWeights lw = weights.layers()[l];
        MatMul.gemm(lw.w1(), state.normed, dim, state.hidden, hiddenDim, hiddenDim, seqLen, dim);
        MatMul.gemm(lw.w3(), state.normed, dim, state.hidden2, hiddenDim, hiddenDim, seqLen, dim);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.siluMultiply(
                                state.hidden,
                                s * hiddenDim,
                                state.hidden2,
                                s * hiddenDim,
                                hiddenDim));
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
            implements Config {
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

    public static final class State extends BaseState {
        final int contextCapacity, batchCapacity;

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
        public void reset() {
            resumeAt(0);
        }

        @SuppressWarnings("unchecked")
        State(Configuration config, int contextCapacity, int batchCapacity, Arena arena) {
            super(arena);
            if (contextCapacity > config.contextLength()) {
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds model contextLength "
                                + config.contextLength());
            }
            this.contextCapacity = contextCapacity;
            int c = Math.max(1, batchCapacity);
            this.batchCapacity = c;
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

        @Override
        public int contextCapacity() {
            return contextCapacity;
        }

        @Override
        public int batchCapacity() {
            return batchCapacity;
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
        float[] ropeFreqs = ModelLoader.ropeFreqFactors(tensors);
        return ropeFreqs != null
                ? RoPE.withFreqFactors(ropeDim, config.ropeTheta, ropeFreqs)
                : RoPE.plain(ropeDim, config.ropeTheta);
    }

    // ---- loadWeights helpers: the old ModelLoader.toF32Tensor/loadQuantized fail-fast contract --

    private static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        return Objects.requireNonNull(tensors.get(name), name);
    }

    /** F32 view by name (dtype checked AT LOAD, the old toF32Tensor fail-fast), or throw. */
    private static MemoryView<MemorySegment> requireF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> v = require(tensors, name);
        Views.requireDatatype(v, DataType.FP32, name);
        return v;
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors,
            Configuration config,
            RoPE.Schedule rope) {
        int n = config.numberOfLayers;
        MemoryView<MemorySegment> tokenEmbeddings = require(tensors, "token_embd.weight");
        MemoryView<MemorySegment> wcls =
                tensors.containsKey("output.weight")
                        ? require(tensors, "output.weight")
                        : tokenEmbeddings; // tied embeddings
        MemoryView<MemorySegment> finalNorm = requireF32(tensors, "output_norm.weight");

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            layers[i] =
                    new LayerWeights(
                            requireF32(tensors, p + "attn_norm.weight"),
                            require(tensors, p + "attn_q.weight"),
                            require(tensors, p + "attn_k.weight"),
                            require(tensors, p + "attn_v.weight"),
                            require(tensors, p + "attn_output.weight"),
                            requireF32(tensors, p + "ffn_norm.weight"),
                            require(tensors, p + "ffn_gate.weight"),
                            require(tensors, p + "ffn_down.weight"),
                            require(tensors, p + "ffn_up.weight"));
        }
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls);
    }
}
