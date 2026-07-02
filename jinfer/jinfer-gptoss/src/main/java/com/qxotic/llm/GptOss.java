// gpt-oss against the com.qxotic.llm model API: a faithful port of the production jinfer GptOss forward
// (jinfer-models). A MoE transformer with alternating sliding-window/full attention, per-head attention
// SINKS, biased Q/K/V/O projections, YaRN-scaled RoPE, and a clamped gated-SiLU expert activation.
// Every layer is attention + top-k MoE (no dense, no recurrent). Text-only (no media/MTP) so this
// implements only LanguageModel. Mirrors the Lfm2 port's structure: a LayerWeights[] record-of-records
// (AttentionWeights + MoeFfnWeights per layer), MoE via the shared CSR Moe.dispatch, and the
// attention/feedForward decomposition each doing its own pre-norm. Deltas vs Lfm2: attention sinks +
// projection biases, YaRN RoPE (float[] cos/sin), SWA ring cache on even layers, clampedSwiglu experts,
// top-k-by-raw-logit then softmax-over-the-selected gating, and an expertWeightsScale.
package com.qxotic.llm;

import com.qxotic.format.gguf.GGUF;

import com.qxotic.jinfer.*;

import static com.qxotic.jinfer.Norms.rmsnorm;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class GptOss implements LanguageModel<GptOss.Configuration, GptOss.Weights, GptOss.State> {

    private final Configuration configuration;
    private final LFMTokenizer tokenizer;
    private final Weights weights;

    GptOss(Configuration configuration, LFMTokenizer tokenizer, Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
    }

    @Override public Configuration config() { return configuration; }
    @Override public Weights weights()       { return weights; }
    public LFMTokenizer tokenizer()          { return tokenizer; }

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
                throw new UnsupportedOperationException("gpt-oss is generative: packed sequences (batched embedding) not supported");
            case com.qxotic.jinfer.Batch.Input.Embeddings e ->
                throw new UnsupportedOperationException("gpt-oss is text-only: embedding input is not supported");
        }
        s.lastChunkLen = n;
        s.outputCount = batch.outputs() == com.qxotic.jinfer.Batch.Outputs.ALL ? n : 1;
        s.position = from + n;
    }

    @Override
    public FloatTensor logits(State s, int output) {
        int dim = configuration.embeddingLength();
        int row = s.lastChunkLen - s.outputCount + output;
        return Parallel.onDecodePool(() -> {
            tailAt(s, row);   // finish the deferred last-layer tail for this row -> s.th (s.residual stays read-only)
            rmsnorm(s.xb, 0, s.th, 0, weights.outputNorm(), dim, configuration.rmsNormEps());
            weights.outputWeight().matmul(s.xb, s.logits, configuration.vocabularySize(), dim);
            return s.logits;
        });
    }

    @Override
    public State fork(State s) {
        State f = new State(configuration, s.contextCapacity, s.batchCapacity);
        for (int l = 0; l < configuration.numberOfLayers(); l++) {
            int len = configuration.kvCachePositions(l, s.contextCapacity) * configuration.kvDim();
            s.keyCache[l].copyTo(0, f.keyCache[l], 0, len);
            s.valueCache[l].copyTo(0, f.valueCache[l], 0, len);
        }
        f.position = s.position;
        return f;
    }

    /** The turn-delimiter / eos ids that terminate generation (convenience for callers/tests). */
    public Set<Integer> stopTokens() {
        Set<Integer> stops = new HashSet<>();
        for (String name : new String[]{"<|return|>", "<|call|>", "<|end|>", "<|endofprompt|>", "<|endoftext|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        return stops;
    }

    // === Math helpers (ported from the production GptOss) ===

    /** out[outOffset, +size] += bias[biasOffset, +size]. */
    static void addBias(FloatTensor out, long outOffset, F32FloatTensor bias, long biasOffset, int size) {
        out.addInPlace(outOffset, bias, biasOffset, size);   // SIMD via F32FloatTensor.addInPlace (bit-identical to the scalar add)
    }

    /** YaRN-scaled RoPE tables (cos/sin), attention mscale baked in; canonical params (betaFast=32,
     *  betaSlow=1, extFactor=1). */
    static float[][] precomputeYarnRope(int contextLength, int headSize, double ropeTheta,
                                        float ropeScalingFactor, int originalContextLength) {
        Pair<float[], float[]> rope = RoPE.precomputeFreqsCisYarn(
                contextLength, headSize, ropeTheta, ropeScalingFactor, originalContextLength, 32f, 1f, 1f, 1f);
        return new float[][]{rope.first(), rope.second()};
    }

    // === Forward ===

    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        embed(state, tokens, tokenOffset, seqLen);
        int lastLayer = configuration.numberOfLayers() - 1;
        for (int l = 0; l < lastLayer; l++) layer(state, l, startPos, seqLen);
        // Lazy last-layer split: write every row's K/V to the cache but DEFER the attention + MoE FFN tail.
        // state.residual is left as the last-layer INPUT; a query finishes exactly the rows it asks for via
        // tailAt() in logits(). Prefill skips the last layer's (expensive MoE) attention+FFN for un-queried rows.
        writeKv(state, lastLayer, startPos, seqLen);
    }

    /** Token-embedding lookup into the residual stream (no scaling). */
    private void embed(State state, int[] tokens, int tokenOffset, int seqLen) {
        int dim = configuration.embeddingLength();
        for (int s = 0; s < seqLen; s++) {
            weights.tokenEmbeddings().copyTo((long) tokens[tokenOffset + s] * dim, state.residual, (long) s * dim, dim);
        }
    }


    /** One block: attention (with sinks), then the MoE FFN, in place on the residual. */
    private void layer(State state, int l, int startPos, int seqLen) {
        attention(state, l, startPos, seqLen);
        moeFeedForward(state, l, state.residual, seqLen);
        if (LLM.TRACE) LLM.traceSum("l_out-" + l, state.residual, seqLen * configuration.embeddingLength());
    }

    // --- attention (GQA + sinks, biased projections, YaRN RoPE, SWA-or-full) ---

    /** Pre-norm GQA with biased Q/K/V/O, NeoX YaRN RoPE (no QK norms), sink-aware flash attention
     *  ({@code scale = 1/sqrt(headSize)}), output projection + bias, residual add. Commits the chunk's
     *  K/V to the (ring on SWA, linear on full) cache. */
    private void attention(State state, int l, int startPos, int seqLen) {
        Configuration config = configuration;
        int dim = config.embeddingLength();
        int headSize = config.headSize(), halfHead = headSize / 2;
        int heads = config.numberOfHeads(), kvHeads = config.numberOfKeyValueHeads();
        int queryDim = config.queryDim(), kvDim = config.kvDim(), kvMul = heads / kvHeads;
        float eps = config.rmsNormEps();
        boolean isSWA = config.isSWA(l);
        AttentionWeights attn = weights.layers()[l].attention();
        float[] cos = weights.ropeCos(), sin = weights.ropeSin();
        FloatTensor bK = state.batchK[l], bV = state.batchV[l];

        F32FloatTensor attNormW = weights.layers()[l].attnNorm();
        Parallel.forRows(seqLen, s -> rmsnorm(state.xb, (long) s * dim, state.residual, (long) s * dim, attNormW, dim, eps));

        // Q = wq @ xb + bias, RoPE per row/head.
        attn.wq().gemm(state.xb, dim, state.q, queryDim, seqLen, queryDim, dim);
        Parallel.forRows(seqLen, s -> {
            addBias(state.q, (long) s * queryDim, attn.qBias(), 0, queryDim);
            for (int h = 0; h < heads; h++) {
                RoPE.applyNeox(state.q, (long) s * queryDim + (long) h * headSize, startPos + s, cos, sin, halfHead);
            }
        });

        // K/V = wk/wv @ xb + biases into the linear batch buffer; RoPE on K per row/head.
        attn.wk().gemm(state.xb, dim, bK, kvDim, seqLen, kvDim, dim);
        attn.wv().gemm(state.xb, dim, bV, kvDim, seqLen, kvDim, dim);
        Parallel.forRows(seqLen, s -> {
            addBias(bK, (long) s * kvDim, attn.kBias(), 0, kvDim);
            addBias(bV, (long) s * kvDim, attn.vBias(), 0, kvDim);
            for (int h = 0; h < kvHeads; h++) {
                RoPE.applyNeox(bK, (long) s * kvDim + (long) h * headSize, startPos + s, cos, sin, halfHead);
            }
        });

        float scale = 1.0f / (float) Math.sqrt(headSize);
        if (seqLen > 1) {
            int window = isSWA ? config.slidingWindow() : 0;
            int ringMask = isSWA ? config.slidingWindow() - 1 : 0;
            FlashAttention.slidingWindowPrefill(state.q, state.xbK,
                    state.keyCache[l], state.valueCache[l], bK, bV,
                    heads, startPos, seqLen, headSize, kvDim, queryDim, kvDim, kvMul,
                    scale, window, ringMask, attn.sinks());
        } else {
            int attStart = config.attentionStart(l, startPos);
            int ringMask = isSWA ? config.slidingWindow() - 1 : 0;
            FlashAttention.flashDecode((F32FloatTensor) state.q, (F32FloatTensor) state.xbK,
                    state.keyCache[l], state.valueCache[l], bK, bV,
                    heads, startPos, attStart, headSize, kvDim, kvMul, scale, ringMask, attn.sinks(), state.decodeScratch);
        }

        // Output projection + bias, residual add to the stream.
        attn.wo().gemm(state.xbK, queryDim, state.xb2, dim, seqLen, dim, queryDim);
        Parallel.forRows(seqLen, s -> addBias(state.xb2, (long) s * dim, attn.oBias(), 0, dim));
        state.residual.addInPlace(0, state.xb2, 0, seqLen * dim);

        // Commit the chunk's K/V to the cache (ring slot on SWA, linear position on full).
        for (int s = 0; s < seqLen; s++) {
            int kvPos = config.kvCacheIndex(l, startPos + s);
            bK.copyTo((long) s * kvDim, state.keyCache[l], (long) kvPos * kvDim, kvDim);
            bV.copyTo((long) s * kvDim, state.valueCache[l], (long) kvPos * kvDim, kvDim);
        }
    }

    // --- MoE FFN (all layers; gpt-oss top-k-by-raw-logit, softmax-over-selected, clampedSwiglu, biases) ---

    /** Pre-norm (post_attention_norm) → router (+bias) → top-k by RAW logit → softmax over the selected
     *  (×expertWeightsScale) → per-expert SEPARATE gate/up (+bias) → clampedSwiglu → down (+bias), routed
     *  via the shared CSR {@link Moe#dispatch} and added to the residual. */
    private void moeFeedForward(State state, int l, FloatTensor resid, int seqLen) {
        Configuration config = configuration;
        int dim = config.embeddingLength(), expertFF = config.expertFeedForwardLength();
        int numExperts = config.expertCount(), topK = Math.min(config.expertUsedCount(), numExperts);
        float scaleW = config.expertWeightsScale(), eps = config.rmsNormEps();
        MoeFfnWeights moe = weights.layers()[l].moe();
        F32FloatTensor postW = weights.layers()[l].postAttnNorm();

        // post-attention norm into xb, route on it.
        Parallel.forRows(seqLen, s -> rmsnorm(state.xb, (long) s * dim, resid, (long) s * dim, postW, dim, eps));
        moe.router().gemm(state.xb, dim, state.moeRouterB, numExperts, seqLen, numExperts, dim);
        Parallel.forRows(seqLen, s -> addBias(state.moeRouterB, (long) s * numExperts, moe.routerBias(), 0, numExperts));

        int[] counts = state.moeExpertCounts;
        Arrays.fill(counts, 0);
        for (int s = 0; s < seqLen; s++) {
            long rb = (long) s * numExperts;
            int base = s * topK;
            for (int k = 0; k < topK; k++) {                       // top-k by RAW logit (argmax + destructive mask)
                int best = -1;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int e = 0; e < numExperts; e++) {
                    float v = state.moeRouterB.getFloat(rb + e);
                    if (v > bestVal) { bestVal = v; best = e; }
                }
                state.moeRowTopE[base + k] = best;
                state.moeRowTopP[base + k] = bestVal;
                state.moeRouterB.setFloat(rb + best, Float.NEGATIVE_INFINITY);
                counts[best]++;
            }
            float maxW = Float.NEGATIVE_INFINITY;                  // softmax over the selected, ×expertWeightsScale
            for (int k = 0; k < topK; k++) maxW = Math.max(maxW, state.moeRowTopP[base + k]);
            float sum = 0f;
            for (int k = 0; k < topK; k++) { float ev = (float) Math.exp(state.moeRowTopP[base + k] - maxW); state.moeRowTopP[base + k] = ev; sum += ev; }
            float inv = sum == 0f ? 0f : 1f / sum;
            for (int k = 0; k < topK; k++) state.moeRowTopP[base + k] = state.moeRowTopP[base + k] * inv * scaleW;
        }

        Moe.Routing r = state.moeRouting;
        r.seqLen = seqLen; r.topK = topK; r.numExperts = numExperts;
        Moe.dispatch(r, dim, state.xb, state.moeGather, state.moeDownB, state.moeOutB, null,
                (e, n, gather, out) -> {
                    long gateUpOffset = (long) e * expertFF * dim, downOffset = (long) e * dim * expertFF;
                    moe.gateExps().gemm(gather, dim, state.hb, expertFF, n, expertFF, dim, gateUpOffset);
                    moe.upExps().gemm(gather, dim, state.hb2, expertFF, n, expertFF, dim, gateUpOffset);
                    Parallel.forRows(n, j -> {
                        addBias(state.hb, (long) j * expertFF, moe.gateBias(), (long) e * expertFF, expertFF);
                        addBias(state.hb2, (long) j * expertFF, moe.upBias(), (long) e * expertFF, expertFF);
                        Activations.clampedSwigluMultiply(state.hb, j * expertFF, state.hb2, j * expertFF, expertFF);
                    });
                    moe.downExps().gemm(state.hb, expertFF, out, dim, n, dim, expertFF, downOffset);
                    Parallel.forRows(n, j -> addBias(out, (long) j * dim, moe.downBias(), (long) e * dim, dim));
                });

        Parallel.forRows(seqLen, s -> resid.addInPlace((long) s * dim, state.moeOutB, (long) s * dim, dim));
    }

    // === Lazy last-layer split (writeKv + tailAt) ===

    /** Last-layer K/V half: pre-norm all rows, biased K/V projection + NeoX RoPE(K), commit to the cache
     *  (ring on SWA, linear on full). No Q, no attention, no O, no MoE - state.residual left untouched. */
    private void writeKv(State state, int l, int startPos, int seqLen) {
        Configuration config = configuration;
        int dim = config.embeddingLength();
        int headSize = config.headSize(), halfHead = headSize / 2;
        int kvHeads = config.numberOfKeyValueHeads(), kvDim = config.kvDim();
        float eps = config.rmsNormEps();
        AttentionWeights attn = weights.layers()[l].attention();
        float[] cos = weights.ropeCos(), sin = weights.ropeSin();
        FloatTensor bK = state.batchK[l], bV = state.batchV[l];
        F32FloatTensor attNormW = weights.layers()[l].attnNorm();
        Parallel.forRows(seqLen, s -> rmsnorm(state.xb, (long) s * dim, state.residual, (long) s * dim, attNormW, dim, eps));
        attn.wk().gemm(state.xb, dim, bK, kvDim, seqLen, kvDim, dim);
        attn.wv().gemm(state.xb, dim, bV, kvDim, seqLen, kvDim, dim);
        Parallel.forRows(seqLen, s -> {
            addBias(bK, (long) s * kvDim, attn.kBias(), 0, kvDim);
            addBias(bV, (long) s * kvDim, attn.vBias(), 0, kvDim);
            for (int h = 0; h < kvHeads; h++) RoPE.applyNeox(bK, (long) s * kvDim + (long) h * headSize, startPos + s, cos, sin, halfHead);
        });
        for (int s = 0; s < seqLen; s++) {
            int kvPos = config.kvCacheIndex(l, startPos + s);
            bK.copyTo((long) s * kvDim, state.keyCache[l], (long) kvPos * kvDim, kvDim);
            bV.copyTo((long) s * kvDim, state.valueCache[l], (long) kvPos * kvDim, kvDim);
        }
    }

    /** Lazy tail: finish the last layer for retained chunk-row {@code i} into state.th, reading its input from
     *  state.residual[i] and attending cache[attStart..pos] inclusive (its own K/V already committed by writeKv;
     *  bK/bV = null, sink-aware, SWA-windowed), then the MoE FFN on state.th. state.residual is never written. */
    private void tailAt(State state, int i) {
        Configuration config = configuration;
        int L = config.numberOfLayers() - 1;
        int dim = config.embeddingLength();
        int headSize = config.headSize(), halfHead = headSize / 2;
        int heads = config.numberOfHeads(), kvHeads = config.numberOfKeyValueHeads();
        int queryDim = config.queryDim(), kvDim = config.kvDim(), kvMul = heads / kvHeads;
        float eps = config.rmsNormEps();
        boolean isSWA = config.isSWA(L);
        AttentionWeights attn = weights.layers()[L].attention();
        float[] cos = weights.ropeCos(), sin = weights.ropeSin();
        int startPos = state.position - state.lastChunkLen;
        int pos = startPos + i;

        // attention: biased Q + NeoX RoPE + sink/SWA single-query flash over the cache + biased O -> tscratch
        rmsnorm(state.tscratch, 0, state.residual, (long) i * dim, weights.layers()[L].attnNorm(), dim, eps);
        attn.wq().gemm(state.tscratch, dim, state.q, queryDim, 1, queryDim, dim);
        addBias(state.q, 0, attn.qBias(), 0, queryDim);
        for (int h = 0; h < heads; h++) RoPE.applyNeox(state.q, (long) h * headSize, pos, cos, sin, halfHead);
        float scale = 1.0f / (float) Math.sqrt(headSize);
        int attStart = config.attentionStart(L, pos);
        int ringMask = isSWA ? config.slidingWindow() - 1 : 0;
        FlashAttention.flashDecode((F32FloatTensor) state.q, (F32FloatTensor) state.xbK,
                state.keyCache[L], state.valueCache[L], null, null,
                heads, pos, attStart, headSize, kvDim, kvMul, scale, ringMask, attn.sinks(), state.decodeScratch);
        attn.wo().gemm(state.xbK, queryDim, state.tscratch, dim, 1, dim, queryDim);
        addBias(state.tscratch, 0, attn.oBias(), 0, dim);
        LLM.addScaledInto(state.th, state.residual, (long) i * dim, state.tscratch, dim, 1.0f);   // th = residual[i] + O
        moeFeedForward(state, L, state.th, 1);   // MoE FFN for this one row, in place on state.th
    }

    // === Configuration ===

    public record Configuration(int embeddingLength, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                                int headSize, int vocabularySize, int contextLength, float rmsNormEps, double ropeTheta,
                                float ropeScalingFactor, int ropeOrigCtx, int slidingWindow, boolean[] swaMask,
                                int expertCount, int expertUsedCount, int expertFeedForwardLength, float expertWeightsScale)
            implements Config {
        public int queryDim() { return numberOfHeads * headSize; }
        public int kvDim() { return numberOfKeyValueHeads * headSize; }
        public boolean isSWA(int layer) { return swaMask[layer]; }
        /** SWA layers keep a window-sized ring (capped at the requested context); full layers index the whole context. */
        public int kvCachePositions(int layer, int cap) { return swaMask[layer] ? Math.min(cap, slidingWindow) : cap; }
        public int kvCacheIndex(int layer, int position) { return swaMask[layer] ? (position & (slidingWindow - 1)) : position; }
        public int attentionStart(int layer, int position) { return swaMask[layer] ? Math.max(0, position - slidingWindow + 1) : 0; }
    }

    // === Weights (per-layer record-of-records: attention + MoE) ===

    public record AttentionWeights(FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
                                   F32FloatTensor qBias, F32FloatTensor kBias, F32FloatTensor vBias, F32FloatTensor oBias,
                                   F32FloatTensor sinks) {}
    public record MoeFfnWeights(FloatTensor router, F32FloatTensor routerBias,
                                FloatTensor gateExps, F32FloatTensor gateBias,
                                FloatTensor upExps, F32FloatTensor upBias,
                                FloatTensor downExps, F32FloatTensor downBias) {}
    public record LayerWeights(F32FloatTensor attnNorm, F32FloatTensor postAttnNorm,
                               AttentionWeights attention, MoeFfnWeights moe) {}
    public record Weights(FloatTensor tokenEmbeddings, LayerWeights[] layers, F32FloatTensor outputNorm,
                          FloatTensor outputWeight, float[] ropeCos, float[] ropeSin) {}

    // === State ===

    public static final class State implements RuntimeState {
        final int contextCapacity, batchCapacity;
        int position, outputCount, lastChunkLen;
        final FloatTensor residual, xb, xb2, xbK, q, logits, th, tscratch;
        final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch();
        final FloatTensor[] keyCache, valueCache, batchK, batchV;
        // MoE scratch (chunk-wide CSR routing); gpt-oss is all-MoE so always allocated.
        final FloatTensor moeRouterB, moeGather, moeDownB, moeOutB, hb, hb2;
        final int[] moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeRowTopE;
        final float[] moeProbByExpert, moeRowTopP;
        final Moe.Routing moeRouting;

        State(Configuration config, int contextCapacity, int batchCapacity) {
            if (contextCapacity > config.contextLength()) {
                throw new IllegalArgumentException("contextCapacity " + contextCapacity
                        + " exceeds model contextLength " + config.contextLength());
            }
            this.contextCapacity = contextCapacity;
            int c = Math.max(1, batchCapacity);
            this.batchCapacity = c;
            int dim = config.embeddingLength();
            int queryDim = config.queryDim(), kvDim = config.kvDim();
            int expertFF = config.expertFeedForwardLength();
            int numExperts = config.expertCount(), topK = Math.max(1, config.expertUsedCount());
            this.residual = FloatTensor.allocateF32(c * dim);
            this.xb = FloatTensor.allocateF32(c * dim);
            this.xb2 = FloatTensor.allocateF32(c * dim);
            this.xbK = FloatTensor.allocateF32(c * queryDim);
            this.q = FloatTensor.allocateF32(c * queryDim);
            this.logits = FloatTensor.allocateF32(config.vocabularySize());
            this.th = FloatTensor.allocateF32(dim);
            this.tscratch = FloatTensor.allocateF32(dim);
            this.hb = FloatTensor.allocateF32(c * expertFF);
            this.hb2 = FloatTensor.allocateF32(c * expertFF);
            this.moeRouterB = FloatTensor.allocateF32(c * numExperts);
            this.moeGather = FloatTensor.allocateF32(c * dim);
            this.moeDownB = FloatTensor.allocateF32(c * dim);
            this.moeOutB = FloatTensor.allocateF32(c * dim);
            this.moeExpertCounts = new int[numExperts];
            this.moeExpertOffsets = new int[numExperts + 1];
            this.moeCursor = new int[numExperts];
            this.moeRowByExpert = new int[c * topK];
            this.moeRowTopE = new int[c * topK];
            this.moeProbByExpert = new float[c * topK];
            this.moeRowTopP = new float[c * topK];
            this.moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeProbByExpert);
            int n = config.numberOfLayers();
            this.keyCache = new FloatTensor[n];
            this.valueCache = new FloatTensor[n];
            this.batchK = new FloatTensor[n];
            this.batchV = new FloatTensor[n];
            for (int l = 0; l < n; l++) {
                int positions = config.kvCachePositions(l, contextCapacity);
                keyCache[l] = FloatTensor.allocateF16(positions, kvDim);
                valueCache[l] = FloatTensor.allocateF16(positions, kvDim);
                batchK[l] = FloatTensor.allocateF32(c * kvDim);
                batchV[l] = FloatTensor.allocateF32(c * kvDim);
            }
        }

        @Override public int contextCapacity() { return contextCapacity; }
        @Override public int batchCapacity()   { return batchCapacity; }
        @Override public int position()         { return position; }
        @Override public int outputCount()      { return outputCount; }
    }

    // === Loading ===

    public static GptOss loadModel(Path ggufPath, int contextLength) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, contextLength, true);
        }
    }

    static GptOss loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        LFMTokenizer tokenizer = new LFMTokenizer(gguf);
        String arch = "gpt-oss";

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) contextLength = modelContextLength;

        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int numberOfKeyValueHeads = gguf.getValue(int.class, arch + ".attention.head_count_kv");
        int headSize = gguf.getValueOrDefault(int.class, arch + ".attention.key_length", embeddingLength / numberOfHeads);
        float rmsNormEps = gguf.getValueOrDefault(float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        double ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 150000f);
        float ropeScalingFactor = gguf.getValueOrDefault(float.class, arch + ".rope.scaling.factor", 1f);
        int ropeOrigCtx = gguf.getValueOrDefault(int.class, arch + ".rope.scaling.original_context_length", contextLength);
        int slidingWindow = gguf.getValue(int.class, arch + ".attention.sliding_window");
        int expertCount = gguf.getValueOrDefault(int.class, arch + ".expert_count", 0);
        int expertUsedCount = gguf.getValueOrDefault(int.class, arch + ".expert_used_count", 0);
        int expertFeedForwardLength = gguf.getValueOrDefault(int.class, arch + ".expert_feed_forward_length", 0);
        float expertWeightsScale = gguf.getValueOrDefault(float.class, arch + ".expert_weights_scale", 1.0f);

        // gpt-oss alternates SWA/full: even layers sliding-window, odd layers full attention.
        boolean[] swaMask = new boolean[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) swaMask[i] = (i % 2) == 0;

        Configuration config = new Configuration(embeddingLength, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                headSize, tokenizer.vocabularySize(), contextLength, rmsNormEps, ropeTheta, ropeScalingFactor,
                ropeOrigCtx, slidingWindow, swaMask, expertCount, expertUsedCount, expertFeedForwardLength, expertWeightsScale);

        if (!loadWeightsFlag) return new GptOss(config, tokenizer, null);
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        return new GptOss(config, tokenizer, loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
        int n = config.numberOfLayers();
        FloatTensor tokenEmbeddings = ModelLoader.loadQuantized(tensors.get("token_embd.weight"));
        FloatTensor outputWeight = tensors.containsKey("output.weight")
                ? ModelLoader.loadQuantized(tensors.get("output.weight")) : tokenEmbeddings;   // tied embeddings
        float[][] rope = precomputeYarnRope(config.contextLength(), config.headSize(), config.ropeTheta(),
                config.ropeScalingFactor(), config.ropeOrigCtx());
        F32FloatTensor outputNorm = ModelLoader.toF32Tensor(tensors.get("output_norm.weight"));

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            AttentionWeights attention = new AttentionWeights(
                    ModelLoader.loadQuantized(tensors.get(p + "attn_q.weight")),
                    ModelLoader.loadQuantized(tensors.get(p + "attn_k.weight")),
                    ModelLoader.loadQuantized(tensors.get(p + "attn_v.weight")),
                    ModelLoader.loadQuantized(tensors.get(p + "attn_output.weight")),
                    ModelLoader.toF32Tensor(tensors.get(p + "attn_q.bias")),
                    ModelLoader.toF32Tensor(tensors.get(p + "attn_k.bias")),
                    ModelLoader.toF32Tensor(tensors.get(p + "attn_v.bias")),
                    ModelLoader.toF32Tensor(tensors.get(p + "attn_output.bias")),
                    ModelLoader.toF32Tensor(tensors.get(p + "attn_sinks.weight")));
            MoeFfnWeights moe = new MoeFfnWeights(
                    ModelLoader.loadQuantized(tensors.get(p + "ffn_gate_inp.weight")),
                    ModelLoader.toF32Tensor(tensors.get(p + "ffn_gate_inp.bias")),
                    ModelLoader.loadQuantized(tensors.get(p + "ffn_gate_exps.weight")),
                    ModelLoader.toF32Tensor(tensors.get(p + "ffn_gate_exps.bias")),
                    ModelLoader.loadQuantized(tensors.get(p + "ffn_up_exps.weight")),
                    ModelLoader.toF32Tensor(tensors.get(p + "ffn_up_exps.bias")),
                    ModelLoader.loadQuantized(tensors.get(p + "ffn_down_exps.weight")),
                    ModelLoader.toF32Tensor(tensors.get(p + "ffn_down_exps.bias")));
            F32FloatTensor attnNorm = ModelLoader.toF32Tensor(tensors.get(p + "attn_norm.weight"));
            F32FloatTensor postAttnNorm = ModelLoader.toF32Tensor(tensors.get(p + "post_attention_norm.weight"));
            layers[i] = new LayerWeights(attnNorm, postAttnNorm, attention, moe);
        }
        return new Weights(tokenEmbeddings, layers, outputNorm, outputWeight, rope[0], rope[1]);
    }
}
