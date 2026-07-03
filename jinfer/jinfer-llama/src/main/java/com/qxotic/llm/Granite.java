// Granite 4.1 (dense "granite" architecture) against the com.qxotic.llm model API: a focused port of the
// production jinfer Llama3's granite path. A standard RoPE GQA + SwiGLU + RMSNorm transformer with
// Granite's four scalars baked in: embedding_scale (x token embeddings), residual_scale (x each sublayer
// output before the residual add), logit_scale (logits DIVIDED by it), and a custom attention.scale that
// REPLACES 1/sqrt(headSize). Interleaved RoPE (GGUF "llama" pair convention), plain freq base (no YaRN,
// no attention-temperature tuning). Text-only, dense FFN. Sibling of Llama/Gemma4/Lfm2.
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

public final class Granite implements LanguageModel<Granite.Configuration, Granite.Weights, Granite.State> {

    private final Configuration configuration;
    private final GgufTokenizer tokenizer;
    private final Weights weights;

    Granite(Configuration configuration, GgufTokenizer tokenizer, Weights weights) {
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
                throw new UnsupportedOperationException("Granite is generative: packed sequences (batched embedding) not supported");
            case com.qxotic.jinfer.Batch.Input.Embeddings e ->
                throw new UnsupportedOperationException("Granite is text-only: embedding input is not supported");
        }
        s.advance(n, batch.outputs());
    }

    @Override
    public FloatTensor logits(State s, int output) {
        int dim = configuration.embeddingLength;
        int row = s.lastChunkLen - s.outputCount + output;
        return Parallel.onDecodePool(() -> {
            rmsnorm(s.xb, 0, s.x, (long) row * dim, weights.outputNorm, dim, configuration.rmsNormEps);
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
        for (String name : new String[]{"<|end_of_text|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        return stops;
    }

    private com.qxotic.jinfer.chat.TurnTemplate turnTemplate;   // memoized: stateless, model-lifetime

    @Override
    public java.util.Optional<com.qxotic.jinfer.chat.TurnTemplate> turnTemplate() {
        if (turnTemplate == null) turnTemplate = new GraniteTurnTemplate(tokenizer());
        return java.util.Optional.of(turnTemplate);
    }

    @Override
    public java.util.Optional<com.qxotic.jinfer.cache.KvCodec<Granite.State>> kvCodec() {
        return java.util.Optional.of(new GraniteKvCodec(config()));
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

        for (int l = 0; l < config.numberOfLayers; l++) {
            int fDim = dim;
            F32FloatTensor attNormW = w.attnNorm[l], ffnNormW = w.ffnNorm[l];
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, (long) s * fDim, state.x, (long) s * fDim, attNormW, fDim, eps));
            attention(state, l, startPos, seqLen);
            LLM.addScaled(state.x, state.xb, seqLen * dim, residScale);
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, (long) s * fDim, state.x, (long) s * fDim, ffnNormW, fDim, eps));
            feedForward(state, l, seqLen);
            LLM.addScaled(state.x, state.xb, seqLen * dim, residScale);
            if (LLM.TRACE) LLM.traceSum("l_out-" + l, state.x, seqLen * dim);
        }
    }

    /** Standard RoPE GQA attention with Granite's custom attention scale: Q/K/V projections, per-row
     *  interleaved RoPE, K/V appended to the cache, causal flash attention (or scalar single-token decode),
     *  output projection back to {@code state.xb}. (No attention-temperature tuning — that's mistral3.) */
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
            for (int h = 0; h < fHeads; h++) LLM.ropeInterleaved(state.attnQ, s * fQDim + h * fHeadSz, fStart + s, w.ropeCr, w.ropeCi, fHalf);
            for (int h = 0; h < fKvHeads; h++) LLM.ropeInterleaved(state.k, s * fKvDim + h * fHeadSz, fStart + s, w.ropeCr, w.ropeCi, fHalf);
        });

        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        float attScale = config.attentionScale();
        if (seqLen == 1) {
            FlashAttention.flashDecode((F32FloatTensor) state.attnQ, (F32FloatTensor) state.attnOut,
                    keyCache, valueCache, state.k, state.v, heads, startPos, 0, headSize, kvDim, kvMul, attScale, 0, null, state.decodeScratch);
        } else {
            FlashAttention.slidingWindowPrefill(state.attnQ, state.attnOut,
                    keyCache, valueCache, state.k, state.v, heads, startPos, seqLen, headSize, kvDim, queryDim, kvDim, kvMul, attScale, 0, 0, null);
        }
        for (int s = 0; s < seqLen; s++) {   // commit this chunk's K/V to the cache for later positions
            state.k.copyTo((long) s * kvDim, keyCache, (long) (startPos + s) * kvDim, kvDim);
            state.v.copyTo((long) s * kvDim, valueCache, (long) (startPos + s) * kvDim, kvDim);
        }
        w.wo[layer].gemm(state.attnOut, queryDim, state.xb, dim, seqLen, dim, queryDim);
    }

    /** Dense SwiGLU FFN over the pre-normed rows in {@code state.xb}, written back to {@code state.xb}. */
    private void feedForward(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        Weights w = weights;
        w.w1[l].gemm(state.xb, dim, state.hb, hiddenDim, seqLen, hiddenDim, dim);     // gate
        w.w3[l].gemm(state.xb, dim, state.hb2, hiddenDim, seqLen, hiddenDim, dim);    // up
        Parallel.forRows(seqLen, s -> Activations.siluMultiply(state.hb, s * hiddenDim, state.hb2, s * hiddenDim, hiddenDim));
        w.w2[l].gemm(state.hb, hiddenDim, state.xb, dim, seqLen, dim, hiddenDim);     // down
    }

    // === Configuration ===

    public record Configuration(int embeddingLength, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                                int headSize, int vocabularySize, int contextLength, float rmsNormEps,
                                float ropeTheta, int ropeDimensionCount, int hiddenDim,
                                float embeddingScale, float residualScale, float logitScale,
                                int bosTokenId, int eosTokenId, boolean addBos, float attentionScaleValue)
            implements Config {
        public int queryDim() { return numberOfHeads * headSize; }
        public int kvDim() { return numberOfKeyValueHeads * headSize; }
        public int ropeHalf() { return Math.min(ropeDimensionCount, headSize) / 2; }
        /** Granite replaces the default 1/sqrt(headSize) with a metadata-supplied attention scale. */
        public float attentionScale() {
            return attentionScaleValue != 0f ? attentionScaleValue : 1.0f / (float) Math.sqrt(headSize);
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

    public static Granite loadModel(Path ggufPath, int contextLength) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, contextLength, true);
        }
    }

    public static Granite loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        GgufTokenizer tokenizer = new GgufTokenizer(gguf, JinjaRenderer::template);
        String arch = gguf.getString("general.architecture");   // "granite"

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

        // Granite's four scalars (default 1.0 / off → plain Llama, but Granite supplies real values).
        float embeddingScale = gguf.getValueOrDefault(float.class, arch + ".embedding_scale", 1.0f);
        float residualScale = gguf.getValueOrDefault(float.class, arch + ".residual_scale", 1.0f);
        float logitScale = gguf.getValueOrDefault(float.class, arch + ".logit_scale", 1.0f);
        float attentionScale = gguf.getValueOrDefault(float.class, arch + ".attention.scale", 0f);

        int bosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.bos_token_id", 1);
        int eosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        boolean addBos = gguf.getValueOrDefault(boolean.class, "tokenizer.ggml.add_bos_token", true);

        Configuration config = new Configuration(embeddingLength, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                headSize, tokenizer.vocabularySize(), contextLength, rmsNormEps, ropeTheta, ropeDimensionCount, hiddenDim,
                embeddingScale, residualScale, logitScale, bosTokenId, eosTokenId, addBos, attentionScale);

        if (!loadWeightsFlag) return new Granite(config, tokenizer, null);
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        Pair<float[], float[]> rope = buildRope(config, tensors);
        return new Granite(config, tokenizer, loadWeights(tensors, config, rope));
    }

    /** Plain RoPE for granite (freq base + dimension count); honors a rope_freqs.weight per-frequency
     *  scaling tensor if present. No YaRN. Returns the interleaved cos/sin tables. */
    static Pair<float[], float[]> buildRope(Configuration config, Map<String, GGMLTensorEntry> tensors) {
        int ropeDim = Math.min(config.ropeDimensionCount, config.headSize);
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
