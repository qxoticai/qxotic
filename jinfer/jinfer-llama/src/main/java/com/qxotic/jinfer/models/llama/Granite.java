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

/**
 * IBM Granite dense text models (GGUF architecture {@code granite}), with the same loading and
 * inference contract as {@link Llama}.
 */
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
        forward(s, ids, from, n);
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
        Views.checkAlive(weights.finalNorm(), "finalNorm");
        int dim = configuration.embeddingLength;
        int row = s.lastBatchSize() - s.outputCount() + output;
        {
            Norms.rmsnorm(
                    s.normed,
                    0,
                    s.residual,
                    (long) row * dim,
                    weights.finalNorm(),
                    dim,
                    configuration.rmsNormEps);
            MatMul.gemv(weights.wcls(), s.normed, s.logits);
            float ls = configuration.logitScale;
            if (ls != 1.0f) {
                Ops.divideInPlace(s.logits, 0, configuration.vocabularySize, ls);
            }
            return s.logits;
        }
    }

    // === Forward ===

    private void forward(State state, int[] tokens, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        // ONCE for the batch: an angle never depends on the layer, so all of them read these rows
        if (config.useRope) {
            RoPE.fill(state.ropeCos, state.ropeSin, startPos, seqLen, config.ropeHalf(), w.rope());
        }
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

        MatMul.gemm(lw.wq(), state.normed, state.query, seqLen);
        MatMul.gemm(lw.wk(), state.normed, state.batchK, seqLen);
        MatMul.gemm(lw.wv(), state.normed, state.batchV, seqLen);
        addBias(state.query, lw.bq(), seqLen, queryDim);
        addBias(state.batchK, lw.bk(), seqLen, kvDim);
        addBias(state.batchV, lw.bv(), seqLen, kvDim);
        if (config.useRope) {
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
        }

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
        MatMul.gemm(lw.wo(), state.attnOut, state.normed, seqLen);
        addBias(state.normed, lw.bo(), seqLen, dim);
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
        MatMul.gemm(lw.w1(), state.normed, state.hidden, seqLen);
        MatMul.gemm(lw.w3(), state.normed, state.hidden2, seqLen);
        addBias(state.hidden, lw.b1(), seqLen, hiddenDim);
        addBias(state.hidden2, lw.b3(), seqLen, hiddenDim);
        Activations.siluMultiply(
                state.hidden, 0, state.hidden2, 0, Math.multiplyExact(seqLen, hiddenDim));
        MatMul.gemm(lw.w2(), state.hidden, state.normed, seqLen);
        addBias(state.normed, lw.b2(), seqLen, dim);
    }

    private static void addBias(
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> bias,
            int rows,
            int columns) {
        if (bias != null) Ops.addRowBiasInPlace(output, 0, bias, 0, rows, columns);
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
            float attentionScaleValue,
            boolean useRope)
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
            MemoryView<MemorySegment> bq,
            MemoryView<MemorySegment> wk,
            MemoryView<MemorySegment> bk,
            MemoryView<MemorySegment> wv,
            MemoryView<MemorySegment> bv,
            MemoryView<MemorySegment> wo,
            MemoryView<MemorySegment> bo,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> w1,
            MemoryView<MemorySegment> b1,
            MemoryView<MemorySegment> w2,
            MemoryView<MemorySegment> b2,
            MemoryView<MemorySegment> w3,
            MemoryView<MemorySegment> b3) {}

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
            int dim = config.embeddingLength;
            int queryDim = config.queryDim();
            int kvDim = config.kvDim();
            int hidden = config.hiddenDim;
            this.residual = Views.allocateF32(memoryArena(), batchCapacity, dim);
            this.normed = Views.allocateF32(memoryArena(), batchCapacity, dim);
            this.batchK = Views.allocateF32(memoryArena(), batchCapacity, kvDim);
            this.batchV = Views.allocateF32(memoryArena(), batchCapacity, kvDim);
            this.query = Views.allocateF32(memoryArena(), batchCapacity, queryDim);
            this.attnOut = Views.allocateF32(memoryArena(), batchCapacity, queryDim);
            this.hidden = Views.allocateF32(memoryArena(), batchCapacity, hidden);
            this.hidden2 = Views.allocateF32(memoryArena(), batchCapacity, hidden);
            this.logits = Views.allocateF32(memoryArena(), 1, config.vocabularySize);
            // rotary values for the batch about to be ingested: sized by BATCH, never context
            this.ropeCos = Views.allocateF32(memoryArena(), batchCapacity, config.ropeHalf());
            this.ropeSin = Views.allocateF32(memoryArena(), batchCapacity, config.ropeHalf());
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
        String arch = gguf.getString("general.architecture");
        Configuration config = readConfiguration(gguf, arch, tokenizer.vocabulary().size());

        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(fileChannel, gguf, arena);
        RoPE.Schedule rope = buildRope(gguf, arch, config, tensors);
        return new Granite(config, tokenizer, loadWeights(tensors, config, rope));
    }

    static Configuration readConfiguration(GGUF gguf, String arch, int vocabularySize) {
        require(arch.equals("granite"), "unsupported architecture '" + arch + "'");

        int contextLength = gguf.getValue(int.class, arch + ".context_length");
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        require(
                contextLength > 0
                        && embeddingLength > 0
                        && numberOfLayers > 0
                        && numberOfHeads > 0
                        && vocabularySize > 0,
                "invalid core dimensions");
        int numberOfKeyValueHeads =
                gguf.getValueOrDefault(int.class, arch + ".attention.head_count_kv", numberOfHeads);
        int headSize =
                gguf.getValueOrDefault(
                        int.class, arch + ".attention.key_length", embeddingLength / numberOfHeads);
        int valueHeadSize =
                gguf.getValueOrDefault(int.class, arch + ".attention.value_length", headSize);
        int hiddenDim = gguf.getValue(int.class, arch + ".feed_forward_length");
        float rmsNormEps =
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10000f);
        int ropeDimensionCount =
                gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);

        // llama.cpp stores zero for an absent embedding/residual multiplier and treats it as a
        // no-op. Normalize that sentinel here so the hot path only sees the effective scale.
        float embeddingScale =
                effectiveScale(gguf.getValueOrDefault(float.class, arch + ".embedding_scale", 0f));
        float residualScale =
                effectiveScale(gguf.getValueOrDefault(float.class, arch + ".residual_scale", 0f));
        float logitScale = gguf.getValue(float.class, arch + ".logit_scale");
        float attentionScale = gguf.getValueOrDefault(float.class, arch + ".attention.scale", 0f);
        // llama.cpp's Granite build reads rope_finetuned as the NoPE switch
        boolean useRope =
                gguf.getValueOrDefault(boolean.class, arch + ".rope.scaling.finetuned", true);

        Configuration config =
                new Configuration(
                        embeddingLength,
                        numberOfLayers,
                        numberOfHeads,
                        numberOfKeyValueHeads,
                        headSize,
                        vocabularySize,
                        contextLength,
                        rmsNormEps,
                        ropeTheta,
                        ropeDimensionCount,
                        hiddenDim,
                        embeddingScale,
                        residualScale,
                        logitScale,
                        attentionScale,
                        useRope);
        require(
                numberOfKeyValueHeads > 0
                        && numberOfHeads % numberOfKeyValueHeads == 0
                        && headSize > 0
                        && valueHeadSize == headSize
                        && hiddenDim > 0,
                "invalid or unsupported attention/FFN dimensions");
        require(
                ropeDimensionCount >= 0
                        && (!useRope
                                || ropeDimensionCount > 0
                                        && (ropeDimensionCount & 1) == 0
                                        && ropeDimensionCount <= headSize),
                "invalid RoPE dimensions");
        require(
                rmsNormEps > 0f
                        && Float.isFinite(rmsNormEps)
                        && ropeTheta > 0f
                        && Float.isFinite(ropeTheta),
                "invalid normalization or RoPE metadata");
        require(
                embeddingScale > 0f
                        && Float.isFinite(embeddingScale)
                        && residualScale > 0f
                        && Float.isFinite(residualScale)
                        && logitScale > 0f
                        && Float.isFinite(logitScale),
                "invalid model scaling metadata");
        require(
                attentionScale == 0f || attentionScale > 0f && Float.isFinite(attentionScale),
                "invalid attention scaling metadata");
        require(
                gguf.getValueOrDefault(int.class, arch + ".vocab_size", vocabularySize)
                        == vocabularySize,
                "tokenizer vocabulary does not match the model");
        require(
                gguf.getValueOrDefault(int.class, arch + ".expert_count", 0) == 0,
                "MoE checkpoints are not supported");
        require(
                !gguf.containsKey(arch + ".deepstack_mapping"),
                "deepstack multimodal checkpoints are not supported");
        require(
                (long) numberOfHeads * headSize <= Integer.MAX_VALUE
                        && (long) numberOfKeyValueHeads * headSize <= Integer.MAX_VALUE,
                "attention dimensions overflow");
        return config;
    }

    /**
     * Granite RoPE: plain/linear scaling or per-frequency llama3 factors. YaRN and LongRoPE are
     * rejected instead of silently producing incorrect positions.
     */
    static RoPE.Schedule buildRope(
            GGUF gguf,
            String arch,
            Configuration config,
            Map<String, MemoryView<MemorySegment>> tensors) {
        int ropeDim = config.ropeDimensionCount;
        String scalingType =
                gguf.getValueOrDefault(String.class, arch + ".rope.scaling.type", "linear");
        Optional<float[]> factors = ModelLoader.ropeFreqFactors(tensors);
        require(
                scalingType.isEmpty()
                        || scalingType.equals("none")
                        || scalingType.equals("linear")
                        || scalingType.equals("llama3"),
                "unsupported rope.scaling.type '" + scalingType + "'");
        require(
                !scalingType.equals("llama3") || factors.isPresent(),
                "llama3 RoPE requires rope_freqs.weight");

        float factor = gguf.getValueOrDefault(float.class, arch + ".rope.scaling.factor", 0f);
        factor = scalingType.equals("none") || factor == 0f ? 1f : factor;
        require(factor > 0f && Float.isFinite(factor), "invalid RoPE scaling factor");

        if (factors.isEmpty() && factor == 1f) return RoPE.plain(ropeDim, config.ropeTheta);
        float[] values = factors.orElseGet(() -> new float[ropeDim / 2]);
        require(values.length == ropeDim / 2, "rope_freqs.weight has the wrong length");
        for (int i = 0; i < values.length; i++) {
            if (factors.isEmpty()) values[i] = factor;
            else values[i] *= factor;
            require(
                    values[i] > 0f && Float.isFinite(values[i]),
                    "rope_freqs.weight contains invalid data");
        }
        return RoPE.withFreqFactors(ropeDim, config.ropeTheta, values);
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
            require(
                    !tensors.containsKey(p + "attn_qkv.weight"),
                    "fused QKV checkpoints are not supported");
            layers[i] =
                    new LayerWeights(
                            ModelLoader.requireF32(tensors, p + "attn_norm.weight"),
                            ModelLoader.require(tensors, p + "attn_q.weight"),
                            optionalBias(tensors, p + "attn_q.bias", config.queryDim()),
                            ModelLoader.require(tensors, p + "attn_k.weight"),
                            optionalBias(tensors, p + "attn_k.bias", config.kvDim()),
                            ModelLoader.require(tensors, p + "attn_v.weight"),
                            optionalBias(tensors, p + "attn_v.bias", config.kvDim()),
                            ModelLoader.require(tensors, p + "attn_output.weight"),
                            optionalBias(tensors, p + "attn_output.bias", config.embeddingLength),
                            ModelLoader.requireF32(tensors, p + "ffn_norm.weight"),
                            ModelLoader.require(tensors, p + "ffn_gate.weight"),
                            optionalBias(tensors, p + "ffn_gate.bias", config.hiddenDim),
                            ModelLoader.require(tensors, p + "ffn_down.weight"),
                            optionalBias(tensors, p + "ffn_down.bias", config.embeddingLength),
                            ModelLoader.require(tensors, p + "ffn_up.weight"),
                            optionalBias(tensors, p + "ffn_up.bias", config.hiddenDim));
        }
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls);
    }

    private static MemoryView<MemorySegment> optionalBias(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int width) {
        MemoryView<MemorySegment> bias = ModelLoader.findF32(tensors, name).orElse(null);
        if (bias != null) require(bias.shape().size() == width, name + " has the wrong length");
        return bias;
    }

    private static float effectiveScale(float value) {
        return value == 0f ? 1f : value;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Granite: " + message);
    }
}
