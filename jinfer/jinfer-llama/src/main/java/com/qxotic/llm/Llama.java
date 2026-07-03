// The standard Llama transformer (RoPE GQA attention + SwiGLU FFN + RMSNorm) against the com.qxotic.llm
// model API: a port of the production jinfer Llama3, covering the "llama" GGUF architecture and its
// same-graph relatives (all distinguished by GGUF metadata, no extra classes):
//   - Llama 3.x: "llama3" RoPE frequency scaling (rope_freqs.weight).
//   - MiniCPM:   embedding_scale / residual_scale / logit_scale (default 1.0 → plain Llama).
//   - Mistral-3: YaRN RoPE scaling + Llama-4-style attention temperature tuning.
//   - SmolLM3:   NoPE - RoPE is skipped on every 4th layer (noRopeLayerStep); otherwise plain Llama.
//   - Granite (dense): the MiniCPM scalars plus a custom QK attention scale.
// Interleaved RoPE (the GGUF "llama" pair convention), KV written to the cache before a causalPrefill /
// flashDecode read, and a scalar residual scale on each sublayer output. Text-only, dense FFN.
package com.qxotic.llm;

import com.qxotic.format.gguf.GGUF;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.jinja.JinjaRenderer;

import static com.qxotic.jinfer.Norms.rmsnorm;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class Llama implements LanguageModel<Llama.Configuration, Llama.Weights, Llama.State> {

    private final Configuration configuration;
    private final GgufTokenizer tokenizer;
    private final Weights weights;

    Llama(Configuration configuration, GgufTokenizer tokenizer, Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
    }

    @Override public Configuration config() { return configuration; }
    @Override public Weights weights()       { return weights; }
    public GgufTokenizer tokenizer()          { return tokenizer; }

    @Override
    public State newState(int contextCapacity, int batchCapacity) {
        State state = new State(configuration, contextCapacity, batchCapacity);
        return state;
    }

    @Override
    public void ingest(State s, com.qxotic.jinfer.Batch batch) {
        int n = batch.count();
        if (n > s.batchCapacity) throw new IllegalArgumentException("batch " + n + " exceeds batchCapacity " + s.batchCapacity);
        int from = s.position();
        if (from + n > s.contextCapacity) {
            throw new IllegalArgumentException("ingest of " + n + " at position " + from + " exceeds contextCapacity " + s.contextCapacity);
        }
        switch (batch.input()) {
            case com.qxotic.jinfer.Batch.Input.Tokens t -> {
                int[] ids = t.ids();
                if (n == 1) Parallel.onDecodePool(() -> { forward(s, ids, 0, from, n); return null; });
                else forward(s, ids, 0, from, n);
            }
            case com.qxotic.jinfer.Batch.Input.Sequences seq ->
                throw new UnsupportedOperationException("Llama is generative: packed sequences (batched embedding) not supported");
            case com.qxotic.jinfer.Batch.Input.Embeddings e ->
                throw new UnsupportedOperationException("Llama is text-only: embedding input is not supported");
        }
        s.advance(n, batch.outputs());
    }

    @Override
    public FloatTensor logits(State s, int output) {
        int dim = configuration.embeddingLength;
        int row = s.lastChunkLen - s.outputCount + output;
        return Parallel.onDecodePool(() -> {
            tailAt(s, row);   // finish the deferred last-layer tail for this row -> s.th (s.x stays read-only)
            rmsnorm(s.xb, 0, s.th, 0, weights.outputNorm, dim, configuration.rmsNormEps);
            weights.outputWeight.matmul(s.xb, s.logits, configuration.vocabularySize, dim);
            float ls = configuration.logitScale;
            if (ls != 1.0f) s.logits.mapInPlace(0, configuration.vocabularySize, v -> v / ls);
            return s.logits;
        });
    }


    /** The eos / turn-delimiter ids that terminate generation (convenience for callers/tests). */
    public Set<Integer> stopTokens() {
        Set<Integer> stops = new HashSet<>();
        if (configuration.eosTokenId >= 0) stops.add(configuration.eosTokenId);
        for (String name : new String[]{"<|eot_id|>", "<|im_end|>", "<|endoftext|>", "<|end_of_text|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        return stops;
    }

    private com.qxotic.jinfer.chat.TurnTemplate turnTemplate;   // memoized: stateless, model-lifetime (pins any construction-time state)

    @Override
    public java.util.Optional<com.qxotic.jinfer.chat.TurnTemplate> turnTemplate() {
        if (turnTemplate == null) turnTemplate = new LlamaTurnTemplate(tokenizer());
        return java.util.Optional.of(turnTemplate);
    }

    @Override
    public java.util.Optional<com.qxotic.jinfer.cache.KvCodec<Llama.State>> kvCodec() {
        // uniform full attention: the shared dense codec over this State's KV arrays
        return java.util.Optional.of(new com.qxotic.jinfer.cache.DenseKvCodec<>(
                config().numberOfLayers(), config().kvDim(), s -> s.keyCache, s -> s.valueCache));
    }

    // === Forward ===


    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;
        float embScale = config.embeddingScale, residScale = config.residualScale;

        for (int s = 0; s < seqLen; s++) {
            w.tokenEmbeddingTable.copyTo((long) tokens[tokenOffset + s] * dim, state.x, (long) s * dim, dim);
        }
        if (embScale != 1.0f) state.x.mapInPlace(0, seqLen * dim, v -> v * embScale);

        int lastLayer = config.numberOfLayers - 1;
        for (int l = 0; l < lastLayer; l++) {
            F32FloatTensor attNormW = w.attnNorm[l], ffnNormW = w.ffnNorm[l];
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, (long) s * dim, state.x, (long) s * dim, attNormW, dim, eps));
            attention(state, l, startPos, seqLen);
            LLM.addScaled(state.x, state.xb, seqLen * dim, residScale);
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, (long) s * dim, state.x, (long) s * dim, ffnNormW, dim, eps));
            feedForward(state, l, state.xb, seqLen);
            LLM.addScaled(state.x, state.xb, seqLen * dim, residScale);
            if (LLM.TRACE) LLM.traceSum("l_out-" + l, state.x, seqLen * dim);
        }
        // Lazy last-layer split: write every row's K/V into the cache (so any row can attend later), but
        // DEFER the attention + FFN tail. state.x is left as the last-layer INPUT residual; a query
        // finishes exactly the rows it asks for via tailAt() in logits(). Prefill pays nothing
        // for the tail here; the saving is the last layer's attention+FFN skipped for every un-queried row.
        writeKv(state, lastLayer, startPos, seqLen);
    }

    /** Commit this chunk's F32 K/V (state.k/v) into the F16 cache at [startPos, startPos+seqLen). */
    private void commitKv(State state, int l, int startPos, int seqLen) {
        int kvDim = configuration.kvDim();
        FloatTensor keyCache = state.keyCache[l], valueCache = state.valueCache[l];
        for (int s = 0; s < seqLen; s++) {
            state.k.copyTo((long) s * kvDim, keyCache, (long) (startPos + s) * kvDim, kvDim);
            state.v.copyTo((long) s * kvDim, valueCache, (long) (startPos + s) * kvDim, kvDim);
        }
    }

    /** Last-layer K/V half: pre-norm all rows, project + RoPE (NoPE-aware) K, and commit K/V to the cache.
     *  No Q, no attention, no O, no FFN - and state.x is left untouched (the last-layer input residual). */
    private void writeKv(State state, int l, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength, kvDim = config.kvDim(), headSize = config.headSize;
        int kvHeads = config.numberOfKeyValueHeads, ropeHalf = config.ropeHalf();
        float eps = config.rmsNormEps;
        F32FloatTensor attNormW = w.attnNorm[l];
        Parallel.forRows(seqLen, s -> rmsnorm(state.xb, (long) s * dim, state.x, (long) s * dim, attNormW, dim, eps));
        w.wk[l].gemm(state.xb, dim, state.k, kvDim, seqLen, kvDim, dim);
        w.wv[l].gemm(state.xb, dim, state.v, kvDim, seqLen, kvDim, dim);
        if (config.useRope(l)) {
            Parallel.forRows(seqLen, s -> {
                for (int h = 0; h < kvHeads; h++) LLM.ropeInterleaved(state.k, s * kvDim + h * headSize, startPos + s, w.ropeCr, w.ropeCi, ropeHalf);
            });
        }
        commitKv(state, l, startPos, seqLen);   // the tail reads every row (incl. its own) from the F16 cache
    }

    /** Lazy tail: finish the last layer for retained chunk-row {@code i} into state.th, reading its input
     *  from state.x[i] and attending cache[0..pos] inclusive - a single causal query aimed at row i (its own
     *  K/V is already in the F16 cache from writeKv, read like any other position). state.x is never written. */
    private void tailAt(State state, int i) {
        Configuration config = configuration;
        Weights w = weights;
        int L = config.numberOfLayers - 1;
        int dim = config.embeddingLength, kvDim = config.kvDim(), queryDim = config.queryDim();
        int heads = config.numberOfHeads, headSize = config.headSize, kvMul = heads / config.numberOfKeyValueHeads;
        int ropeHalf = config.ropeHalf();
        float eps = config.rmsNormEps, residScale = config.residualScale, attScale = config.attentionScale();
        int startPos = state.position - state.lastChunkLen;   // global position of chunk row 0
        int pos = startPos + i;                                // global position of row i

        rmsnorm(state.tscratch, 0, state.x, (long) i * dim, w.attnNorm[L], dim, eps);   // pre-norm reads s.x[i] directly (read-only)
        w.wq[L].gemm(state.tscratch, dim, state.attnQ, queryDim, 1, queryDim, dim);   // Q for this row (attnQ is free scratch)
        if (config.useRope(L)) {
            for (int h = 0; h < heads; h++) LLM.ropeInterleaved(state.attnQ, h * headSize, pos, w.ropeCr, w.ropeCi, ropeHalf);
        }
        float aScale = config.attnTemp(pos);
        if (aScale != 1.0f) state.attnQ.mapInPlace(0, queryDim, v -> v * aScale);
        // Single causal query over cache[0..pos] INCLUSIVE (bK/bV = null): row i's own K/V is already in the
        // F16 cache from writeKv, read like every other position - no separate current-token buffer.
        FlashAttention.flashDecode((F32FloatTensor) state.attnQ, (F32FloatTensor) state.attnOut,
                state.keyCache[L], state.valueCache[L], null, null, heads, pos, 0, headSize, kvDim, kvMul, attScale, 0, null, state.decodeScratch);
        w.wo[L].gemm(state.attnOut, queryDim, state.tscratch, dim, 1, dim, queryDim);   // O -> tscratch
        LLM.addScaledInto(state.th, state.x, (long) i * dim, state.tscratch, dim, residScale);   // th = s.x[i] + residScale*O (born, no seed copy)
        rmsnorm(state.tscratch, 0, state.th, 0, w.ffnNorm[L], dim, eps);
        feedForward(state, L, state.tscratch, 1);                                        // SwiGLU (one row, in place on tscratch)
        LLM.addScaled(state.th, state.tscratch, dim, residScale);                        // FFN residual -> state.th finished
    }

    /** Standard RoPE GQA attention: Q/K/V projections, per-row interleaved RoPE (+ optional attn-temp),
     *  K/V appended to the contiguous cache, then causal flash attention (or scalar single-token decode),
     *  output projection written back to {@code state.xb}. */
    private void attention(State state, int layer, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int headSize = config.headSize;
        int heads = config.numberOfHeads;
        int kvHeads = config.numberOfKeyValueHeads;
        int kvDim = config.kvDim();
        int queryDim = config.queryDim();
        int kvMul = heads / kvHeads;
        int ropeHalf = config.ropeHalf();

        w.wq[layer].gemm(state.xb, dim, state.attnQ, queryDim, seqLen, queryDim, dim);
        w.wk[layer].gemm(state.xb, dim, state.k, kvDim, seqLen, kvDim, dim);
        w.wv[layer].gemm(state.xb, dim, state.v, kvDim, seqLen, kvDim, dim);
        int fHeadSz = headSize, fHeads = heads, fKvHeads = kvHeads, fQDim = queryDim, fKvDim = kvDim, fStart = startPos, fHalf = ropeHalf;
        boolean useRope = config.useRope(layer);   // SmolLM3 NoPE: some layers skip RoPE entirely
        Parallel.forRows(seqLen, s -> {
            if (useRope) {
                for (int h = 0; h < fHeads; h++) LLM.ropeInterleaved(state.attnQ, s * fQDim + h * fHeadSz, fStart + s, w.ropeCr, w.ropeCi, fHalf);
                for (int h = 0; h < fKvHeads; h++) LLM.ropeInterleaved(state.k, s * fKvDim + h * fHeadSz, fStart + s, w.ropeCr, w.ropeCi, fHalf);
            }
            float aScale = config.attnTemp(fStart + s);
            if (aScale != 1.0f) state.attnQ.mapInPlace((long) s * fQDim, fQDim, v -> v * aScale);
        });

        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        float attScale = config.attentionScale();
        // Full causal attention over cache[0..startPos) + this chunk's F32 K/V (state.k/v); window=0.
        if (seqLen == 1) {
            FlashAttention.flashDecode((F32FloatTensor) state.attnQ, (F32FloatTensor) state.attnOut,
                    keyCache, valueCache, state.k, state.v, heads, startPos, 0, headSize, kvDim, kvMul, attScale, 0, null, state.decodeScratch);
        } else {
            FlashAttention.slidingWindowPrefill(state.attnQ, state.attnOut,
                    keyCache, valueCache, state.k, state.v, heads, startPos, seqLen, headSize, kvDim, queryDim, kvDim, kvMul, attScale, 0, 0, null);
        }
        commitKv(state, layer, startPos, seqLen);   // commit this chunk's K/V to the cache for later positions
        w.wo[layer].gemm(state.attnOut, queryDim, state.xb, dim, seqLen, dim, queryDim);
    }



    /** Dense SwiGLU FFN over the pre-normed rows in {@code io}, written back to {@code io} in place. */
    private void feedForward(State state, int l, FloatTensor io, int seqLen) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        Weights w = weights;
        w.w1[l].gemm(io, dim, state.hb, hiddenDim, seqLen, hiddenDim, dim);     // gate
        w.w3[l].gemm(io, dim, state.hb2, hiddenDim, seqLen, hiddenDim, dim);    // up
        Parallel.forRows(seqLen, s -> Activations.siluMultiply(state.hb, s * hiddenDim, state.hb2, s * hiddenDim, hiddenDim));
        w.w2[l].gemm(state.hb, hiddenDim, io, dim, seqLen, dim, hiddenDim);     // down
    }

    // === Configuration ===

    public record Configuration(int embeddingLength, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                                int headSize, int vocabularySize, int contextLength, float rmsNormEps,
                                float ropeTheta, int ropeDimensionCount, int hiddenDim,
                                float embeddingScale, float residualScale, float logitScale,
                                int bosTokenId, int eosTokenId, boolean addBos,
                                float attnTempScale, int attnTempFloorScale, float attentionScaleValue,
                                int noRopeLayerStep)
            implements Config {
        public int queryDim() { return numberOfHeads * headSize; }
        public int kvDim() { return numberOfKeyValueHeads * headSize; }
        public int ropeHalf() { return Math.min(ropeDimensionCount, headSize) / 2; }
        public float attentionScale() {
            return attentionScaleValue != 0f ? attentionScaleValue : 1.0f / (float) Math.sqrt(headSize);
        }
        /** Llama-4 / Mistral-3 attention temperature tuning; 1.0 (no-op) below the floor, 0 = disabled. */
        public float attnTemp(int position) {
            if (attnTempScale == 0f || attnTempFloorScale <= 0) return 1.0f;
            return (float) (Math.log(Math.floor((double) position / attnTempFloorScale) + 1.0) * attnTempScale + 1.0);
        }
        /** SmolLM3 NoPE: RoPE is skipped on every {@code noRopeLayerStep}-th layer (1-indexed); 0 = always RoPE. */
        public boolean useRope(int layer) {
            return noRopeLayerStep <= 0 || (layer + 1) % noRopeLayerStep != 0;
        }
    }

    // === Weights ===

    public record Weights(FloatTensor tokenEmbeddingTable, F32FloatTensor outputNorm, FloatTensor outputWeight,
                          F32FloatTensor[] attnNorm, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo,
                          F32FloatTensor[] ffnNorm, FloatTensor[] w1, FloatTensor[] w3, FloatTensor[] w2,
                          float[] ropeCr, float[] ropeCi) {}

    // === State ===

    public static final class State extends com.qxotic.jinfer.BaseState {
        final int contextCapacity, batchCapacity;
        final FloatTensor x, xb, k, v, attnQ, attnOut, hb, hb2, logits;
        // Lazy last-layer tail: single-row scratch, kept DISTINCT from the batch buffers so x/k/v stay
        // read-only across queries (any retained row can be finished, in any order, repeatedly).
        final FloatTensor th, tscratch;
        final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch();
        final FloatTensor[] keyCache, valueCache;

        State(Configuration config, int contextCapacity, int batchCapacity) {
            if (contextCapacity > config.contextLength()) {
                throw new IllegalArgumentException("contextCapacity " + contextCapacity
                        + " exceeds model contextLength " + config.contextLength());
            }
            this.contextCapacity = contextCapacity;
            int c = Math.max(1, batchCapacity);
            this.batchCapacity = c;
            int dim = config.embeddingLength;
            int queryDim = config.queryDim();
            int kvDim = config.kvDim();
            int hidden = config.hiddenDim;
            this.x = FloatTensor.allocateF32(c * dim);
            this.xb = FloatTensor.allocateF32(c * dim);
            this.k = FloatTensor.allocateF32(c * kvDim);
            this.v = FloatTensor.allocateF32(c * kvDim);
            this.attnQ = FloatTensor.allocateF32(c * queryDim);
            this.attnOut = FloatTensor.allocateF32(c * queryDim);
            this.hb = FloatTensor.allocateF32(c * hidden);
            this.hb2 = FloatTensor.allocateF32(c * hidden);
            this.logits = FloatTensor.allocateF32(config.vocabularySize);
            this.th = FloatTensor.allocateF32(dim);
            this.tscratch = FloatTensor.allocateF32(dim);
            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            for (int l = 0; l < config.numberOfLayers; l++) {
                keyCache[l] = FloatTensor.allocateF16(contextCapacity, kvDim);
                valueCache[l] = FloatTensor.allocateF16(contextCapacity, kvDim);
            }
        }

        @Override public int contextCapacity() { return contextCapacity; }
        @Override public int batchCapacity()   { return batchCapacity; }
    }

    // === Loading ===

    public static Llama loadModel(Path ggufPath, int contextLength) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, contextLength, true);
        }
    }

    public static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        GgufTokenizer tokenizer = new GgufTokenizer(gguf, JinjaRenderer::template);
        String arch = gguf.getString("general.architecture");

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) contextLength = modelContextLength;

        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int numberOfKeyValueHeads = gguf.getValueOrDefault(int.class, arch + ".attention.head_count_kv", numberOfHeads);
        int headSize = gguf.getValueOrDefault(int.class, arch + ".attention.key_length", embeddingLength / numberOfHeads);
        int hiddenDim = gguf.getValue(int.class, arch + ".feed_forward_length");
        float rmsNormEps = gguf.getValueOrDefault(float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10000f);
        int ropeDimensionCount = gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);

        boolean isMiniCpm = arch.equals("minicpm");
        float embeddingScale = gguf.getValueOrDefault(float.class, arch + ".embedding_scale", isMiniCpm ? 12.0f : 1.0f);
        float residualScale = gguf.getValueOrDefault(float.class, arch + ".residual_scale",
                isMiniCpm ? (float) (1.4 / Math.sqrt(numberOfLayers)) : 1.0f);
        float logitScale = gguf.getValueOrDefault(float.class, arch + ".logit_scale",
                isMiniCpm ? embeddingLength / 256.0f : 1.0f);

        float attnTempScale = gguf.getValueOrDefault(float.class, arch + ".attention.temperature_scale", 0f);
        int attnTempFloorScale = gguf.getValueOrDefault(int.class, arch + ".rope.scaling.original_context_length", 0);
        float attentionScale = gguf.getValueOrDefault(float.class, arch + ".attention.scale", 0f);
        int noRopeLayerStep = arch.equals("smollm3") ? 4 : 0;   // SmolLM3 NoPE: skip RoPE on every 4th layer (llama.cpp hardcodes 4)

        int bosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.bos_token_id", 1);
        int eosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        boolean addBos = gguf.getValueOrDefault(boolean.class, "tokenizer.ggml.add_bos_token", true);

        Configuration config = new Configuration(embeddingLength, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                headSize, tokenizer.vocabularySize(), contextLength, rmsNormEps, ropeTheta, ropeDimensionCount, hiddenDim,
                embeddingScale, residualScale, logitScale, bosTokenId, eosTokenId, addBos, attnTempScale, attnTempFloorScale, attentionScale,
                noRopeLayerStep);

        if (!loadWeightsFlag) return new Llama(config, tokenizer, null);
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        Pair<float[], float[]> rope = buildRope(gguf, arch, config, tensors);
        return new Llama(config, tokenizer, loadWeights(tensors, config, rope));
    }

    /** RoPE flavor from GGUF metadata: YaRN (mistral3), "llama3" per-frequency scaling (rope_freqs.weight),
     *  or plain RoPE (Llama/MiniCPM). Returns the interleaved cos/sin tables applyInterleaved consumes. */
    static Pair<float[], float[]> buildRope(GGUF gguf, String arch, Configuration config, Map<String, GGMLTensorEntry> tensors) {
        int ropeDim = Math.min(config.ropeDimensionCount, config.headSize);
        String scalingType = gguf.getValueOrDefault(String.class, arch + ".rope.scaling.type", "");
        if (scalingType.equals("yarn")) {
            float factor = gguf.getValue(float.class, arch + ".rope.scaling.factor");
            int origCtx = gguf.getValue(int.class, arch + ".rope.scaling.original_context_length");
            float betaFast = gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_fast", 32f);
            float betaSlow = gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_slow", 1f);
            float logMul = gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_log_multiplier", 0f);
            float kMscale = factor <= 1f ? 1f : (float) (1.0 + 0.1 * Math.log(factor));
            float attnFactor = logMul != 0f ? 1.0f / kMscale : 1.0f;
            return RoPE.precomputeFreqsCisYarn(config.contextLength, ropeDim, config.ropeTheta, factor, origCtx, betaFast, betaSlow, 1f, attnFactor);
        }
        float[] ropeFreqs = ModelLoader.ropeFreqFactors(tensors);
        return ropeFreqs != null
                ? RoPE.precomputeFreqsCisFromFreqs(config.contextLength, ropeDim, config.ropeTheta, ropeFreqs)
                : RoPE.precomputeFreqsCis(config.contextLength, ropeDim, config.ropeTheta);
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config, Pair<float[], float[]> rope) {
        int n = config.numberOfLayers;
        FloatTensor tokenEmbeddingTable = ModelLoader.loadQuantized(tensors.get("token_embd.weight"));
        FloatTensor outputWeight = tensors.containsKey("output.weight")
                ? ModelLoader.loadQuantized(tensors.get("output.weight")) : tokenEmbeddingTable;
        return new Weights(
                tokenEmbeddingTable,
                ModelLoader.toF32Tensor(tensors.get("output_norm.weight")),
                outputWeight,
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_q.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_k.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_v.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_output.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down.weight")),
                rope.first(), rope.second());
    }
}
