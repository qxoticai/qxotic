// The standard Llama transformer (RoPE GQA attention + SwiGLU FFN + RMSNorm), covering the "llama"
// GGUF architecture and its same-graph relatives — all distinguished by GGUF metadata, no extra
// classes:
//   - Llama 3.x: "llama3" RoPE frequency scaling (rope_freqs.weight).
//   - MiniCPM:   three extra scalars — embedding_scale (x token embeddings), residual_scale (x each
//                sublayer output before the residual add), logit_scale (logits are DIVIDED by it) —
//                which default to 1.0, so a plain "llama"-arch model loads as an ordinary Llama.
//   - Mistral-3 / Ministral ("mistral3"): YaRN RoPE scaling + Llama-4-style attention temperature
//                tuning (Q scaled per position; a no-op below the original context length).
//   - Granite ("granite", dense): the three MiniCPM-style scalars (here non-trivial) plus a custom
//                QK attention scale (attention.scale, replacing 1/sqrt(headSize)).
// Kept entirely behind the Model seam; cross-checked against ../llama3.java/Llama3.java and
// llama.cpp's llama/minicpm/mistral3/granite graphs. Single-token forward + batched prefill.
package com.llama4j;

import com.qxotic.format.gguf.GGUF;

import static com.llama4j.Norms.rmsnorm;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

final class Llama3 implements Model {

    private final Configuration configuration;
    private final LFMTokenizer tokenizer;
    private final Weights weights;

    Llama3(Configuration configuration, LFMTokenizer tokenizer, Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
    }

    @Override
    public LFMTokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public int contextLength() {
        return configuration.contextLength;
    }

    @Override
    public int vocabularySize() {
        return configuration.vocabularySize;
    }

    /** Force the single-token reference forward even for multi-token chunks (parity debugging). */
    static final boolean SINGLE_TOKEN_PREFILL = System.getProperty("llama3.singleTokenPrefill") != null;

    @Override
    public int batchCapacity() {
        return SINGLE_TOKEN_PREFILL ? 1 : Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
    }


    @Override
    public State createNewState() {
        State state = new State(configuration);
        // -1 = "no prior token": for add_bos=false models nothing is prepended (the chat template /
        // raw prompt is fed verbatim); otherwise seed BOS so the prefill auto-prepends/dedups it.
        state.latestToken = configuration.addBos ? configuration.bosTokenId : -1;
        return state;
    }

    @Override
    public void ingest(InferenceState state, int[] tokens, int tokenOffset, int startPosition, int sequenceLength) {
        State s = (State) state;
        if (sequenceLength > s.capacity) {
            throw new IllegalArgumentException("sequenceLength " + sequenceLength + " exceeds batch capacity " + s.capacity);
        }
        if (SINGLE_TOKEN_PREFILL) {
            for (int i = 0; i < sequenceLength; i++) forward(s, tokens, tokenOffset + i, startPosition + i, 1);
        } else {
            forward(s, tokens, tokenOffset, startPosition, sequenceLength);
        }
        s.latestToken = tokens[tokenOffset + sequenceLength - 1];
        s.logitsValid = false;
    }

    @Override
    public FloatTensor computeLogits(InferenceState state) {
        State s = (State) state;
        if (s.logitsValid) {
            return s.logits;
        }
        int dim = configuration.embeddingLength;
        rmsnorm(s.xb, 0, s.x, s.lastRowOffset, weights.outputNorm, dim, configuration.rmsNormEps);
        weights.outputWeight.matmul(s.xb, s.logits, configuration.vocabularySize, dim);
        if (configuration.logitScale != 1.0f) {
            s.logits.divideInPlace(0, configuration.vocabularySize, configuration.logitScale);
        }
        s.logitsValid = true;
        return s.logits;
    }

    @Override
    public ChatFormat chatFormat() {
        return ChatFormats.forModel(tokenizer);
    }

    @Override
    public Set<Integer> stopTokens() {
        Set<Integer> stops = new HashSet<>();
        if (configuration.eosTokenId >= 0) stops.add(configuration.eosTokenId);
        for (String name : new String[]{"<|im_end|>", "<|endoftext|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        return stops;
    }

    // === Math helpers ===

    // === Forward (single token) ===

    /** Single forward pass over {@code seqLen} tokens at positions {@code [startPos, startPos+seqLen)}.
     *  One token (decode) is the {@code seqLen == 1} case — projections route gemm->gemv, attention
     *  takes the scalar single-token path. */
    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;
        float embScale = config.embeddingScale, residScale = config.residualScale;

        for (int s = 0; s < seqLen; s++) {
            w.tokenEmbeddingTable.copyTo(tokens[tokenOffset + s] * dim, state.x, s * dim, dim);
        }
        if (embScale != 1.0f) state.x.mapInPlace(0, seqLen * dim, v -> v * embScale);

        for (int l = 0; l < config.numberOfLayers; l++) {
            int fDim = dim;
            F32FloatTensor attNormW = w.attnNorm[l], ffnNormW = w.ffnNorm[l];
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, s * fDim, state.x, s * fDim, attNormW, fDim, eps));
            attention(state, l, startPos, seqLen);
            state.x.saxpyInPlace(0, state.xb, 0, seqLen * dim, residScale);
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, s * fDim, state.x, s * fDim, ffnNormW, fDim, eps));
            Ffn.dense(w.w1[l], w.w3[l], w.w2[l], state.xb, state.hb, state.hb2, state.xb, seqLen, dim, config.hiddenDim, Ffn.Act.SILU_GLU);
            state.x.saxpyInPlace(0, state.xb, 0, seqLen * dim, residScale);
        }
        state.lastRowOffset = (seqLen - 1) * dim;
    }

    /** Standard RoPE GQA attention. Q/K/V projections, per-row RoPE, K/V appended to the contiguous
     *  cache, then a scalar flat-softmax for a single decode token or causal flash attention for a
     *  chunk; output projection written back to {@code state.xb}. */
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
        Parallel.forRows(seqLen, s -> {
            for (int h = 0; h < fHeads; h++) RoPE.applyInterleaved(state.attnQ, s * fQDim + h * fHeadSz, fStart + s, w.ropeCr, w.ropeCi, fHalf);
            for (int h = 0; h < fKvHeads; h++) RoPE.applyInterleaved(state.k, s * fKvDim + h * fHeadSz, fStart + s, w.ropeCr, w.ropeCi, fHalf);
            float aScale = config.attnTemp(fStart + s);
            if (aScale != 1.0f) state.attnQ.mapInPlace(s * fQDim, fQDim, v -> v * aScale);
        });

        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        for (int s = 0; s < seqLen; s++) {
            state.k.copyTo(s * kvDim, keyCache, (startPos + s) * kvDim, kvDim);
            state.v.copyTo(s * kvDim, valueCache, (startPos + s) * kvDim, kvDim);
        }

        if (seqLen == 1) {
            int position = startPos;
            float attScale = config.attentionScale();
            Parallel.parallelFor(0, heads, h -> {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength;
                int kvHeadOffset = (h / kvMul) * headSize;
                for (int t = 0; t <= position; t++) {
                    state.att.setFloat(attOffset + t, state.attnQ.dot(qOffset, keyCache, t * kvDim + kvHeadOffset, headSize) * attScale);
                }
                state.att.softmaxInPlace(attOffset, position + 1);
                state.attnOut.fillInPlace(qOffset, headSize, 0f);
                for (int t = 0; t <= position; t++) {
                    state.attnOut.saxpyInPlace(qOffset, valueCache, t * kvDim + kvHeadOffset, headSize, state.att.getFloat(attOffset + t));
                }
            });
        } else {
            FlashAttention.causalPrefill((F32FloatTensor) state.attnQ, (F32FloatTensor) state.attnOut,
                    keyCache, valueCache, heads, startPos, seqLen, headSize, kvDim, queryDim, kvMul, config.attentionScale());
        }
        w.wo[layer].gemm(state.attnOut, queryDim, state.xb, dim, seqLen, dim, queryDim);
    }

    // === Configuration ===

    record Configuration(int embeddingLength, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                         int headSize, int vocabularySize, int contextLength, float rmsNormEps,
                         float ropeTheta, int ropeDimensionCount, int hiddenDim,
                         float embeddingScale, float residualScale, float logitScale,
                         int bosTokenId, int eosTokenId, boolean addBos,
                         float attnTempScale, int attnTempFloorScale, float attentionScale) {
        int queryDim() { return numberOfHeads * headSize; }
        int kvDim() { return numberOfKeyValueHeads * headSize; }
        int ropeHalf() { return Math.min(ropeDimensionCount, headSize) / 2; }


        /** Llama-4 / Mistral-3 attention temperature tuning: the per-position factor Q is scaled by
         *  (offset 0). 1.0 (no-op) for positions below the floor scale, so it only engages at very
         *  long context; 0 scale disables it entirely (plain Llama/MiniCPM). */
        float attnTemp(int position) {
            if (attnTempScale == 0f || attnTempFloorScale <= 0) return 1.0f;
            return (float) (Math.log(Math.floor((double) position / attnTempFloorScale) + 1.0) * attnTempScale + 1.0);
        }
    }

    // === Weights ===

    record Weights(FloatTensor tokenEmbeddingTable, F32FloatTensor outputNorm, FloatTensor outputWeight,
                   F32FloatTensor[] attnNorm, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo,
                   F32FloatTensor[] ffnNorm, FloatTensor[] w1, FloatTensor[] w3, FloatTensor[] w2,
                   float[] ropeCr, float[] ropeCi) {
    }

    // === State ===

    static final class State implements InferenceState {
        final int capacity;
        final FloatTensor x, xb, k, v, att, logits, hb, hb2;
        final FloatTensor attnQ, attnOut;
        final FloatTensor[] keyCache, valueCache;
        int latestToken;
        boolean logitsValid;
        int lastRowOffset;

        State(Configuration config) {
            int c = Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
            this.capacity = c;
            int dim = config.embeddingLength;
            int queryDim = config.queryDim();
            int kvDim = config.kvDim();
            int hidden = config.hiddenDim;

            this.x = F32FloatTensor.allocate(c * dim);
            this.xb = F32FloatTensor.allocate(c * dim);
            this.k = F32FloatTensor.allocate(c * kvDim);
            this.v = F32FloatTensor.allocate(c * kvDim);
            this.att = F32FloatTensor.allocate(config.numberOfHeads * config.contextLength);
            this.logits = F32FloatTensor.allocate(config.vocabularySize);
            this.hb = F32FloatTensor.allocate(c * hidden);
            this.hb2 = F32FloatTensor.allocate(c * hidden);
            this.attnQ = F32FloatTensor.allocate(c * queryDim);
            this.attnOut = F32FloatTensor.allocate(c * queryDim);

            // F32 KV cache: for this 1B model at typical context the cache is L2-resident, so an F16
            // cache's f16->f32 decode overhead in the flash kernel outweighs its halved memory traffic
            // (measured slower); F16 only wins when genuinely memory-bound (much larger model/context).
            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            for (int l = 0; l < config.numberOfLayers; l++) {
                keyCache[l] = F32FloatTensor.allocate(config.contextLength * kvDim);
                valueCache[l] = F32FloatTensor.allocate(config.contextLength * kvDim);
            }
        }

        @Override public int latestToken() { return latestToken; }

        @Override public void latestToken(int token) { this.latestToken = token; }
    }

    // === Loading ===

    static Llama3 loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load Llama model")) {
            try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
                return loadModel(fileChannel, gguf, contextLength, true);
            }
        }
    }

    static Llama3 loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        LFMTokenizer tokenizer = new LFMTokenizer(gguf);
        String arch = gguf.getString("general.architecture");

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) {
            contextLength = modelContextLength;
        }
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int numberOfKeyValueHeads = gguf.getValueOrDefault(int.class, arch + ".attention.head_count_kv", numberOfHeads);
        int headSize = gguf.getValueOrDefault(int.class, arch + ".attention.key_length", embeddingLength / numberOfHeads);
        int hiddenDim = gguf.getValue(int.class, arch + ".feed_forward_length");
        float rmsNormEps = gguf.getValueOrDefault(float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10000f);
        int ropeDimensionCount = gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);

        // MiniCPM scales: default to 1.0 (plain Llama). For a true "minicpm" arch with the keys
        // absent, fall back to llama.cpp's MiniCPM defaults so old exports behave identically.
        boolean isMiniCpm = arch.equals("minicpm");
        float embeddingScale = gguf.getValueOrDefault(float.class, arch + ".embedding_scale", isMiniCpm ? 12.0f : 1.0f);
        float residualScale = gguf.getValueOrDefault(float.class, arch + ".residual_scale",
                isMiniCpm ? (float) (1.4 / Math.sqrt(numberOfLayers)) : 1.0f);
        float logitScale = gguf.getValueOrDefault(float.class, arch + ".logit_scale",
                isMiniCpm ? embeddingLength / 256.0f : 1.0f);

        // Mistral-3 attention temperature tuning (Llama-4 style); 0 = disabled (plain Llama/MiniCPM).
        // The floor scale is the YaRN original context length (n_ctx_orig_yarn), per llama.cpp mistral3.
        float attnTempScale = gguf.getValueOrDefault(float.class, arch + ".attention.temperature_scale", 0f);
        int attnTempFloorScale = gguf.getValueOrDefault(int.class, arch + ".rope.scaling.original_context_length", 0);
        // QK score scale: Granite sets a custom attention.scale; everything else uses 1/sqrt(headSize).
        float attentionScale = gguf.getValueOrDefault(float.class, arch + ".attention.scale", 0f);
        if (attentionScale == 0f) attentionScale = 1.0f / (float) Math.sqrt(headSize);

        int bosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.bos_token_id", 1);
        int eosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        // Honor add_bos_token: Granite (and other GPT-2/tekken-tokenized models) set it false, so no
        // BOS is auto-prepended — the prompt is ingested verbatim, matching llama.cpp.
        boolean addBos = gguf.getValueOrDefault(boolean.class, "tokenizer.ggml.add_bos_token", true);

        Configuration config = new Configuration(embeddingLength, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                headSize, tokenizer.vocabularySize(), contextLength, rmsNormEps, ropeTheta, ropeDimensionCount, hiddenDim,
                embeddingScale, residualScale, logitScale, bosTokenId, eosTokenId, addBos, attnTempScale, attnTempFloorScale, attentionScale);

        if (!loadWeightsFlag) {
            return new Llama3(config, tokenizer, null);
        }
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        Pair<float[], float[]> rope = buildRope(gguf, arch, config, tensors);
        return new Llama3(config, tokenizer, loadWeights(tensors, config, rope));
    }

    /** Selects the RoPE flavor from GGUF metadata: YaRN scaling (mistral3), "llama3" per-frequency
     *  scaling (rope_freqs.weight), or plain RoPE (Llama/MiniCPM). Returns the interleaved cos/sin
     *  tables {@link RoPE#applyInterleaved} consumes. */
    static Pair<float[], float[]> buildRope(GGUF gguf, String arch, Configuration config, Map<String, GGMLTensorEntry> tensors) {
        int ropeDim = Math.min(config.ropeDimensionCount, config.headSize);
        String scalingType = gguf.getValueOrDefault(String.class, arch + ".rope.scaling.type", "");
        if (scalingType.equals("yarn")) {
            float factor = gguf.getValue(float.class, arch + ".rope.scaling.factor");
            int origCtx = gguf.getValue(int.class, arch + ".rope.scaling.original_context_length");
            float betaFast = gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_fast", 32f);
            float betaSlow = gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_slow", 1f);
            float logMul = gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_log_multiplier", 0f);
            // llama.cpp folds an extra yarn_attn_factor onto the in-kernel mscale (kMscale): for
            // log_mul==0 it nets to kMscale (gpt-oss style), for mistral3's log_mul==1 it cancels
            // kMscale so the net RoPE magnitude is 1.0. (See llama-context.cpp yarn_attn_factor.)
            float kMscale = factor <= 1f ? 1f : (float) (1.0 + 0.1 * Math.log(factor));
            float attnFactor = logMul != 0f ? 1.0f / kMscale : 1.0f;
            return RoPE.precomputeFreqsCisYarn(config.contextLength, ropeDim, config.ropeTheta, factor, origCtx, betaFast, betaSlow, 1f, attnFactor);
        }
        // Llama-3.x uses "llama3" RoPE frequency scaling (rope_freqs.weight); plain Llama/MiniCPM omit it.
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
