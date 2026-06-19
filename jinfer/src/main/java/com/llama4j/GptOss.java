// gpt-oss ("gpt-oss") support: a MoE transformer with alternating sliding-window/full attention,
// per-head attention SINKS, biased projections, YARN-scaled RoPE, and a clamped gated-SiLU expert
// activation. Kept entirely behind the Model seam; ported from ../gptoss.java/GptOss.java. Single-token
// forward (batchCapacity 1) first; batched prefill added on top (mirrors Gemma4/Qwen35).
package com.llama4j;

import com.qxotic.format.gguf.GGUF;

import static com.llama4j.Norms.rmsnorm;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

final class GptOss implements Model {

    private final Configuration configuration;
    private final LFMTokenizer tokenizer;
    private final Weights weights;

    GptOss(Configuration configuration, LFMTokenizer tokenizer, Weights weights) {
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
    static final boolean SINGLE_TOKEN_PREFILL = System.getProperty("gptoss.singleTokenPrefill") != null;

    @Override
    public int batchCapacity() {
        return SINGLE_TOKEN_PREFILL ? 1 : Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
    }

    @Override
    public State createNewState() {
        State state = new State(configuration);
        Integer bos = tokenizer.getSpecialTokens().get("<|startoftext|>");
        state.latestToken = bos != null ? bos : 199998;
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
        for (String name : new String[]{"<|return|>", "<|call|>", "<|end|>", "<|endofprompt|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        Integer eos = tokenizer.getSpecialTokens().get("<|endoftext|>");
        if (eos != null) stops.add(eos);
        return stops;
    }

    // === Math helpers ===

    /** out[outOffset, +size] += bias[biasOffset, +size]. */
    static void addBias(FloatTensor out, int outOffset, F32FloatTensor bias, int biasOffset, int size) {
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, out.getFloat(outOffset + i) + bias.getFloat(biasOffset + i));
        }
    }

    /** gpt-oss clamped gated activation: gate clamped above at 7 with swish(alpha=1.702); up clamped
     *  to [-7,7] then (up+1) as the multiplicand. */
    static float clampedSwiglu(float gate, float up) {
        float x = Math.min(gate, 7.0f);
        float y = Math.clamp(up, -7.0f, 7.0f);
        return (float) (x / (1.0 + Math.exp(1.702f * -x)) * (y + 1.0));
    }

    /** YARN-scaled RoPE tables (cos/sin), with the attention mscale baked in. gpt-oss uses the
     *  canonical YaRN parameters (betaFast=32, betaSlow=1, extFactor=1); see
     *  {@link RoPE#precomputeFreqsCisYarn}. */
    static float[][] precomputeYarnRope(int contextLength, int headSize, double ropeTheta,
                                        float ropeScalingFactor, int originalContextLength) {
        Pair<float[], float[]> rope = RoPE.precomputeFreqsCisYarn(
                contextLength, headSize, ropeTheta, ropeScalingFactor, originalContextLength, 32f, 1f, 1f, 1f);
        return new float[][]{rope.first(), rope.second()};
    }

    // === Forward (single token) ===

    /** Per-head softmax with an attention sink: the sink adds exp(sink-max) to the denominator only. */
    static void softmaxWithSink(FloatTensor tensor, int offset, int size, float sink) {
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) maxVal = Math.max(maxVal, tensor.getFloat(offset + i));
        maxVal = Math.max(maxVal, sink);
        float sum = 0f;
        for (int i = 0; i < size; i++) {
            float e = (float) Math.exp(tensor.getFloat(offset + i) - maxVal);
            tensor.setFloat(offset + i, e);
            sum += e;
        }
        sum += (float) Math.exp(sink - maxVal);
        float inv = 1f / sum;
        for (int i = 0; i < size; i++) tensor.setFloat(offset + i, tensor.getFloat(offset + i) * inv);
    }

    private void moeForward(State state, int layer) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int expertFF = config.expertFeedForwardLength;
        int numExperts = config.expertCount;
        int topK = Math.min(config.expertUsedCount, numExperts);

        rmsnorm(state.moeInput, 0, state.x, 0, w.postAttnNorm[layer], dim, config.rmsNormEps);
        w.ffnGateInp[layer].matmul(state.moeInput, state.routerLogits, numExperts, dim);
        addBias(state.routerLogits, 0, w.ffnGateInpBias[layer], 0, numExperts);

        // Top-k by raw logit (argmax + mask), then softmax over the selected logits (renormalized).
        int[] topExperts = state.topExperts;
        float[] topWeights = state.topWeights;
        for (int k = 0; k < topK; k++) {
            int best = -1;
            float bestVal = Float.NEGATIVE_INFINITY;
            for (int e = 0; e < numExperts; e++) {
                float v = state.routerLogits.getFloat(e);
                if (v > bestVal) { bestVal = v; best = e; }
            }
            topExperts[k] = best;
            topWeights[k] = bestVal;
            state.routerLogits.setFloat(best, Float.NEGATIVE_INFINITY);
        }
        float maxW = Float.NEGATIVE_INFINITY;
        for (int k = 0; k < topK; k++) maxW = Math.max(maxW, topWeights[k]);
        float sum = 0f;
        for (int k = 0; k < topK; k++) { topWeights[k] = (float) Math.exp(topWeights[k] - maxW); sum += topWeights[k]; }
        float inv = sum == 0f ? 0f : 1f / sum;

        state.moeOutput.fillInPlace(0, dim, 0f);
        for (int k = 0; k < topK; k++) {
            int e = topExperts[k];
            float prob = topWeights[k] * inv * config.expertWeightsScale;
            int gateUpOffset = e * expertFF * dim;
            int downOffset = e * dim * expertFF;
            w.ffnGateExps[layer].matmul(state.moeInput, state.hb, expertFF, dim, gateUpOffset);
            addBias(state.hb, 0, w.ffnGateExpsBias[layer], e * expertFF, expertFF);
            w.ffnUpExps[layer].matmul(state.moeInput, state.hb2, expertFF, dim, gateUpOffset);
            addBias(state.hb2, 0, w.ffnUpExpsBias[layer], e * expertFF, expertFF);
            for (int i = 0; i < expertFF; i++) {
                state.hb.setFloat(i, clampedSwiglu(state.hb.getFloat(i), state.hb2.getFloat(i)));
            }
            w.ffnDownExps[layer].matmul(state.hb, state.expertDown, dim, expertFF, downOffset);
            addBias(state.expertDown, 0, w.ffnDownExpsBias[layer], e * dim, dim);
            state.moeOutput.saxpyInPlace(0, state.expertDown, 0, dim, prob);
        }
        state.x.addInPlace(0, state.moeOutput, 0, dim);
    }

    // === Batched forward (prompt processing): seqLen tokens in one pass ===

    /**
     * Processes {@code seqLen} tokens at positions {@code [startPos, startPos+seqLen)} in a single
     * pass, turning the per-token projections (Q/K/V/O, router, experts) into GEMMs. The chunk's K/V
     * is staged in a linear per-layer buffer (batchK/batchV) so attention reads in-chunk keys there
     * and prior keys from the ring/full cache; the chunk is committed to the cache at the end. Leaves
     * the post-final-layer residual in {@code x}; {@link #computeLogits} finalizes the last row.
     * Token-exact (greedy) vs the single-token {@link #forward} path.
     */
    /** Single forward pass over {@code seqLen} tokens. One token (decode) is the {@code seqLen == 1}
     *  case: projections route gemm->gemv and attention/MoE take their single-token cores. */
    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;

        for (int s = 0; s < seqLen; s++) {
            w.tokenEmbeddingTable.copyTo(tokens[tokenOffset + s] * dim, state.x, s * dim, dim);
        }

        for (int l = 0; l < config.numberOfLayers; l++) {
            int fDim = dim;
            F32FloatTensor attNormW = w.attnNorm[l];
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, s * fDim, state.x, s * fDim, attNormW, fDim, eps));
            attention(state, l, startPos, seqLen);
            moe(state, l, seqLen);
        }

        state.lastRowOffset = (seqLen - 1) * dim;
        state.logitsValid = false;
    }

    /** MoE FFN: single-token expert loop for decode, CSR-grouped GEMMs for a chunk. */
    private void moe(State state, int l, int seqLen) {
        if (seqLen == 1) moeForward(state, l);
        else moeForwardBatch(state, l, seqLen);
    }

    /**
     * Batched attention for the chunk (mirrors the single-token attention's biases/RoPE/sinks and
     * Gemma4.flashAttention's SWA tiling). Q/K/V projections become GEMMs; K/V is staged in the
     * per-layer linear batch buffer (norm-free, gpt-oss has no QK norms), then flash attention reads
     * in-chunk keys from there and prior keys from the cache. The per-head attention SINK is folded
     * into each row's final softmax normalizer. Output proj + bias, residual add to x. The chunk's
     * K/V is committed to the ring/full cache at the end (position order).
     */
    private void attention(State state, int l, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int headSize = config.headSize;
        int halfHead = headSize / 2;
        int heads = config.numberOfHeads;
        int kvHeads = config.numberOfKeyValueHeads;
        int kvDim = config.kvDim();
        int queryDim = config.queryDim();
        int kvMul = heads / kvHeads;
        boolean isSWA = config.isSWA[l];
        FloatTensor bK = state.batchK[l], bV = state.batchV[l];

        // Q = wq @ xb (GEMM) + bias, RoPE per row/head.
        w.wq[l].gemm(state.xb, dim, state.q, queryDim, seqLen, queryDim, dim);
        int fQDim = queryDim, fHeadSz = headSize, fHalf = halfHead, fStart = startPos;
        Parallel.forRows(seqLen, s -> {
            addBias(state.q, s * fQDim, w.attnQBias[l], 0, fQDim);
            for (int h = 0; h < heads; h++) {
                RoPE.applyNeox(state.q, s * fQDim + h * fHeadSz, fStart + s, w.ropeCos, w.ropeSin, fHalf);
            }
        });

        // K/V = wk/wv @ xb (GEMM) + biases into the linear batch buffer; RoPE on K per row/head.
        w.wk[l].gemm(state.xb, dim, bK, kvDim, seqLen, kvDim, dim);
        w.wv[l].gemm(state.xb, dim, bV, kvDim, seqLen, kvDim, dim);
        int fKvDim = kvDim;
        Parallel.forRows(seqLen, s -> {
            addBias(bK, s * fKvDim, w.attnKBias[l], 0, fKvDim);
            addBias(bV, s * fKvDim, w.attnVBias[l], 0, fKvDim);
            for (int h = 0; h < kvHeads; h++) {
                RoPE.applyNeox(bK, s * fKvDim + h * fHeadSz, fStart + s, w.ropeCos, w.ropeSin, fHalf);
            }
        });

        // Single decode token: scalar flat-softmax with the per-head sink, reading prior keys from the
        // cache and this token's key from the staged batch buffer. A chunk uses sink-aware flash attention.
        if (seqLen == 1) {
            int position = startPos;
            int attStart = config.attentionStart(l, position);
            float attScale = 1.0f / (float) Math.sqrt(headSize);
            FloatTensor keyCache = state.keyCache[l], valueCache = state.valueCache[l];
            int finalL = l;
            Parallel.parallelFor(0, heads, h -> {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength;
                int kvHeadOffset = (h / kvMul) * headSize;
                for (int t = attStart; t <= position; t++) {
                    int off = t < position ? config.kvCacheIndex(finalL, t) * kvDim + kvHeadOffset : kvHeadOffset;
                    state.att.setFloat(attOffset + t, state.q.dot(qOffset, t < position ? keyCache : bK, off, headSize) * attScale);
                }
                softmaxWithSink(state.att, attOffset + attStart, position - attStart + 1, w.attnSinks[finalL].getFloat(h));
                state.xbK.fillInPlace(qOffset, headSize, 0f);
                for (int t = attStart; t <= position; t++) {
                    int off = t < position ? config.kvCacheIndex(finalL, t) * kvDim + kvHeadOffset : kvHeadOffset;
                    state.xbK.saxpyInPlace(qOffset, t < position ? valueCache : bV, off, headSize, state.att.getFloat(attOffset + t));
                }
            });
        } else {
            flashAttention(state, l, startPos, seqLen, headSize, kvDim, queryDim, kvMul, isSWA);
        }

        // Output projection (GEMM) + bias, residual add to x.
        w.wo[l].gemm(state.xbK, queryDim, state.xb2, dim, seqLen, dim, queryDim);
        int fDim = dim;
        Parallel.forRows(seqLen, s -> addBias(state.xb2, s * fDim, w.attnOutputBias[l], 0, fDim));
        state.x.addInPlace(0, state.xb2, 0, seqLen * dim);

        // Commit the chunk's K/V to the cache for future chunks/decode (position order; for SWA the
        // later positions overwrite the oldest ring slots, leaving the last `window` positions live).
        for (int s = 0; s < seqLen; s++) {
            int kvPos = config.kvCacheIndex(l, startPos + s);
            bK.copyTo(s * kvDim, state.keyCache[l], kvPos * kvDim, kvDim);
            bV.copyTo(s * kvDim, state.valueCache[l], kvPos * kvDim, kvDim);
        }
    }

    /**
     * gpt-oss adapter: ring-SWA (or full) flash attention via the shared
     * {@link FlashAttention#slidingWindowPrefill} block, passing this layer's per-head attention sinks.
     * SWA layers ring their KV cache (slot = {@code pos & (slidingWindow-1)}, power-of-two enforced in
     * {@link Configuration}); full layers store linearly. The batch K/V buffer is stride {@code kvDim}.
     */
    private void flashAttention(State state, int layer, int startPos, int seqLen,
                               int headSize, int kvDim, int queryDim, int kvMul, boolean isSWA) {
        int window = isSWA ? configuration.slidingWindow : 0;
        int ringMask = isSWA ? configuration.slidingWindow - 1 : 0;
        FlashAttention.slidingWindowPrefill(state.q, state.xbK,
                state.keyCache[layer], state.valueCache[layer], state.batchK[layer], state.batchV[layer],
                configuration.numberOfHeads, startPos, seqLen, headSize, kvDim, queryDim, kvDim, kvMul,
                1.0f / (float) Math.sqrt(headSize), window, ringMask, weights.attnSinks[layer]);
    }

    /**
     * Batched MoE (mirrors Gemma4.moeFfnBatch grouped MoE; gpt-oss specifics: top-4 by RAW router
     * logit, softmax-over-top4 renorm, SEPARATE gate/up, clampedSwiglu, biases). The chunk's tokens
     * are grouped by routed expert (CSR) so each expert's gate/up/down weights are read once per
     * chunk via GEMM. Token-exact vs the single-token {@link #moeForward}. Output added to x.
     */
    private void moeForwardBatch(State state, int l, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        float eps = config.rmsNormEps;
        int dim = config.embeddingLength;
        int expertFF = config.expertFeedForwardLength;
        int numExperts = config.expertCount;
        int topK = Math.min(config.expertUsedCount, numExperts);

        // post-attention norm -> expert input (rows parallel).
        F32FloatTensor postW = w.postAttnNorm[l];
        Parallel.forRows(seqLen, s -> rmsnorm(state.moeInputB, s * dim, state.x, s * dim, postW, dim, eps));

        // Router = ffn_gate_inp @ moeInputB (GEMM) + bias per row.
        w.ffnGateInp[l].gemm(state.moeInputB, dim, state.moeRouterB, numExperts, seqLen, numExperts, dim);
        Parallel.forRows(seqLen, s -> addBias(state.moeRouterB, s * numExperts, w.ffnGateInpBias[l], 0, numExperts));

        // Per-row top-k by RAW logit (argmax + mask), softmax over the selected (renorm, subtract
        // maxW); store (expert, prob) with prob *= expertWeightsScale. Bucket into CSR by expert.
        int[] counts = state.moeExpertCounts;
        Arrays.fill(counts, 0);
        for (int s = 0; s < seqLen; s++) {
            int rb = s * numExperts;
            for (int ki = 0; ki < topK; ki++) {
                int best = -1;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int e = 0; e < numExperts; e++) {
                    float v = state.moeRouterB.getFloat(rb + e);
                    if (v > bestVal) { bestVal = v; best = e; }
                }
                state.moeRowTopE[s * topK + ki] = best;
                state.moeRowTopP[s * topK + ki] = bestVal;
                state.moeRouterB.setFloat(rb + best, Float.NEGATIVE_INFINITY);
                counts[best]++;
            }
            float maxW = Float.NEGATIVE_INFINITY;
            for (int ki = 0; ki < topK; ki++) maxW = Math.max(maxW, state.moeRowTopP[s * topK + ki]);
            float sum = 0f;
            for (int ki = 0; ki < topK; ki++) {
                float ev = (float) Math.exp(state.moeRowTopP[s * topK + ki] - maxW);
                state.moeRowTopP[s * topK + ki] = ev;
                sum += ev;
            }
            float inv = sum == 0f ? 0f : 1f / sum;
            for (int ki = 0; ki < topK; ki++) {
                state.moeRowTopP[s * topK + ki] = state.moeRowTopP[s * topK + ki] * inv * config.expertWeightsScale;
            }
        }
        int[] off = state.moeExpertOffsets;
        off[0] = 0;
        for (int e = 0; e < numExperts; e++) off[e + 1] = off[e] + counts[e];
        int[] cursor = state.moeCursor;
        System.arraycopy(off, 0, cursor, 0, numExperts);
        for (int s = 0; s < seqLen; s++) {
            for (int ki = 0; ki < topK; ki++) {
                int e = state.moeRowTopE[s * topK + ki];
                int pos = cursor[e]++;
                state.moeRowByExpert[pos] = s;
                state.moeProbByExpert[pos] = state.moeRowTopP[s * topK + ki];
            }
        }

        // Experts (grouped): per expert with n>0 rows, one GEMM each for gate/up/down; clampedSwiglu;
        // scatter-add prob-weighted output. gpt-oss has SEPARATE gate/up -> two buffers.
        state.moeOutB.fillInPlace(0, seqLen * dim, 0f);
        for (int e = 0; e < numExperts; e++) {
            int start = off[e], n = off[e + 1] - start;
            if (n == 0) continue;
            int fE = e;
            // gather this expert's rows (parallel; each writes a distinct moeGather row).
            Parallel.forRows(n, j -> state.moeInputB.copyTo(state.moeRowByExpert[start + j] * dim, state.moeGather, j * dim, dim));
            int gateUpOffset = e * expertFF * dim;
            int downOffset = e * dim * expertFF;
            // gate + bias, up + bias, clampedSwiglu(gate, up) -> moeGateB (per row).
            w.ffnGateExps[l].gemm(state.moeGather, dim, state.moeGateB, expertFF, n, expertFF, dim, gateUpOffset);
            w.ffnUpExps[l].gemm(state.moeGather, dim, state.moeUpB, expertFF, n, expertFF, dim, gateUpOffset);
            Parallel.forRows(n, j -> {
                addBias(state.moeGateB, j * expertFF, w.ffnGateExpsBias[l], fE * expertFF, expertFF);
                addBias(state.moeUpB, j * expertFF, w.ffnUpExpsBias[l], fE * expertFF, expertFF);
                for (int i = 0; i < expertFF; i++) {
                    state.moeGateB.setFloat(j * expertFF + i,
                            clampedSwiglu(state.moeGateB.getFloat(j * expertFF + i), state.moeUpB.getFloat(j * expertFF + i)));
                }
            });
            // down + bias (per row).
            w.ffnDownExps[l].gemm(state.moeGateB, expertFF, state.moeDownB, dim, n, dim, expertFF, downOffset);
            Parallel.forRows(n, j -> addBias(state.moeDownB, j * dim, w.ffnDownExpsBias[l], fE * dim, dim));
            // scatter-add prob-weighted (parallel; rows within an expert are distinct, so race-free).
            Parallel.forRows(n, j -> state.moeOutB.saxpyInPlace(state.moeRowByExpert[start + j] * dim, state.moeDownB, j * dim, dim,
                    state.moeProbByExpert[start + j]));
        }

        // Add the MoE output to the residual (per row).
        Parallel.forRows(seqLen, s -> state.x.addInPlace(s * dim, state.moeOutB, s * dim, dim));
    }

    // === Configuration ===

    static final class Configuration {
        final int embeddingLength;
        final int numberOfLayers;
        final int numberOfHeads;
        final int numberOfKeyValueHeads;
        final int headSize;
        final int vocabularySize;
        final int contextLength;
        final float rmsNormEps;
        final double ropeTheta;
        final float ropeScalingFactor;
        final int ropeOrigCtx;
        final int slidingWindow;
        final boolean[] isSWA;
        final int expertCount;
        final int expertUsedCount;
        final int expertFeedForwardLength;
        final float expertWeightsScale;

        Configuration(int embeddingLength, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                      int headSize, int vocabularySize, int contextLength, float rmsNormEps, double ropeTheta,
                      float ropeScalingFactor, int ropeOrigCtx, int slidingWindow, boolean[] isSWA,
                      int expertCount, int expertUsedCount, int expertFeedForwardLength, float expertWeightsScale) {
            if (slidingWindow <= 0 || Integer.bitCount(slidingWindow) != 1) {
                throw new IllegalArgumentException("slidingWindow must be a power of 2, got " + slidingWindow);
            }
            this.embeddingLength = embeddingLength;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.headSize = headSize;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.ropeScalingFactor = ropeScalingFactor;
            this.ropeOrigCtx = ropeOrigCtx;
            this.slidingWindow = slidingWindow;
            this.isSWA = isSWA;
            this.expertCount = expertCount;
            this.expertUsedCount = expertUsedCount;
            this.expertFeedForwardLength = expertFeedForwardLength;
            this.expertWeightsScale = expertWeightsScale;
        }

        int queryDim() {
            return numberOfHeads * headSize;
        }

        int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        /** SWA layers keep a window-sized ring; full layers index the whole context. */
        int kvCachePositions(int layer) {
            return isSWA[layer] ? Math.min(contextLength, slidingWindow) : contextLength;
        }

        int kvCacheIndex(int layer, int position) {
            return isSWA[layer] ? (position & (slidingWindow - 1)) : position;
        }

        int attentionStart(int layer, int position) {
            return isSWA[layer] ? Math.max(0, position - slidingWindow + 1) : 0;
        }
    }

    // === Weights ===

    static final class Weights {
        final FloatTensor tokenEmbeddingTable, outputWeight;
        final F32FloatTensor outputNorm;
        final F32FloatTensor[] attnNorm, postAttnNorm;
        final FloatTensor[] wq, wk, wv, wo;
        final F32FloatTensor[] attnQBias, attnKBias, attnVBias, attnOutputBias, attnSinks;
        final FloatTensor[] ffnGateInp, ffnGateExps, ffnUpExps, ffnDownExps;
        final F32FloatTensor[] ffnGateInpBias, ffnGateExpsBias, ffnUpExpsBias, ffnDownExpsBias;
        final float[] ropeCos, ropeSin;

        Weights(FloatTensor tokenEmbeddingTable, FloatTensor outputWeight, F32FloatTensor outputNorm,
                F32FloatTensor[] attnNorm, F32FloatTensor[] postAttnNorm, FloatTensor[] wq, FloatTensor[] wk,
                FloatTensor[] wv, FloatTensor[] wo, F32FloatTensor[] attnQBias, F32FloatTensor[] attnKBias,
                F32FloatTensor[] attnVBias, F32FloatTensor[] attnOutputBias, F32FloatTensor[] attnSinks,
                FloatTensor[] ffnGateInp, FloatTensor[] ffnGateExps, FloatTensor[] ffnUpExps, FloatTensor[] ffnDownExps,
                F32FloatTensor[] ffnGateInpBias, F32FloatTensor[] ffnGateExpsBias, F32FloatTensor[] ffnUpExpsBias,
                F32FloatTensor[] ffnDownExpsBias, float[] ropeCos, float[] ropeSin) {
            this.tokenEmbeddingTable = tokenEmbeddingTable;
            this.outputWeight = outputWeight;
            this.outputNorm = outputNorm;
            this.attnNorm = attnNorm;
            this.postAttnNorm = postAttnNorm;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
            this.attnQBias = attnQBias;
            this.attnKBias = attnKBias;
            this.attnVBias = attnVBias;
            this.attnOutputBias = attnOutputBias;
            this.attnSinks = attnSinks;
            this.ffnGateInp = ffnGateInp;
            this.ffnGateExps = ffnGateExps;
            this.ffnUpExps = ffnUpExps;
            this.ffnDownExps = ffnDownExps;
            this.ffnGateInpBias = ffnGateInpBias;
            this.ffnGateExpsBias = ffnGateExpsBias;
            this.ffnUpExpsBias = ffnUpExpsBias;
            this.ffnDownExpsBias = ffnDownExpsBias;
            this.ropeCos = ropeCos;
            this.ropeSin = ropeSin;
        }
    }

    // === State ===

    static final class State implements InferenceState {
        // Batched scratch (capacity rows): the residual stream and projections hold one row per token
        // in the current chunk; att stays single-row (only the single-token reference path uses it).
        // The ring/full KV caches are the cross-row source of truth for prior chunks.
        final int capacity;
        final FloatTensor x, xb, xb2, xbK, q, att, logits;
        // Linear (non-ring) K/V for the current chunk, per layer: flash attention reads in-chunk keys
        // here and prior keys from the cache, so a chunk longer than the SWA window never overwrites a
        // ring slot another row in the same chunk still needs. Committed to the cache at chunk end.
        final FloatTensor[] batchK, batchV;
        // Single-token MoE scratch (decode / reference path).
        final FloatTensor moeInput, routerLogits, hb, hb2, expertDown, moeOutput;
        final int[] topExperts;
        final float[] topWeights;
        // Batched MoE scratch (chunk-wide): the expert FFN groups the chunk's tokens by routed expert
        // (CSR) so each expert's gate/up/down weights are read once per chunk via GEMM.
        final FloatTensor moeInputB, moeRouterB, moeOutB, moeGather, moeGateB, moeUpB, moeDownB;
        final int[] moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeRowTopE;
        final float[] moeProbByExpert, moeRowTopP;
        final FloatTensor[] keyCache, valueCache;
        int latestToken;
        boolean logitsValid;
        int lastRowOffset;     // offset into x of the row whose logits computeLogits finalizes

        State(Configuration config) {
            int c = Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
            this.capacity = c;
            int dim = config.embeddingLength;
            int queryDim = config.queryDim();
            int kvDim = config.kvDim();
            int expertFF = config.expertFeedForwardLength;
            int numExperts = config.expertCount;
            int topK = Math.max(1, config.expertUsedCount);
            this.x = F32FloatTensor.allocate(c * dim);
            this.xb = F32FloatTensor.allocate(c * dim);
            this.xb2 = F32FloatTensor.allocate(c * dim);
            this.xbK = F32FloatTensor.allocate(c * queryDim);
            this.q = F32FloatTensor.allocate(c * queryDim);
            this.att = F32FloatTensor.allocate(config.numberOfHeads * config.contextLength);
            this.logits = F32FloatTensor.allocate(config.vocabularySize);
            this.moeInput = F32FloatTensor.allocate(dim);
            this.routerLogits = F32FloatTensor.allocate(numExperts);
            this.hb = F32FloatTensor.allocate(expertFF);
            this.hb2 = F32FloatTensor.allocate(expertFF);
            this.expertDown = F32FloatTensor.allocate(dim);
            this.moeOutput = F32FloatTensor.allocate(dim);
            this.topExperts = new int[topK];
            this.topWeights = new float[topK];
            // Batched MoE scratch.
            this.moeInputB = F32FloatTensor.allocate(c * dim);
            this.moeRouterB = F32FloatTensor.allocate(c * numExperts);
            this.moeOutB = F32FloatTensor.allocate(c * dim);
            this.moeGather = F32FloatTensor.allocate(c * dim);
            this.moeGateB = F32FloatTensor.allocate(c * expertFF);
            this.moeUpB = F32FloatTensor.allocate(c * expertFF);
            this.moeDownB = F32FloatTensor.allocate(c * dim);
            this.moeExpertCounts = new int[numExperts];
            this.moeExpertOffsets = new int[numExperts + 1];
            this.moeCursor = new int[numExperts];
            this.moeRowByExpert = new int[c * topK];
            this.moeRowTopE = new int[c * topK];
            this.moeProbByExpert = new float[c * topK];
            this.moeRowTopP = new float[c * topK];
            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            this.batchK = new FloatTensor[config.numberOfLayers];
            this.batchV = new FloatTensor[config.numberOfLayers];
            for (int l = 0; l < config.numberOfLayers; l++) {
                int positions = config.kvCachePositions(l);
                keyCache[l] = F32FloatTensor.allocate(positions * kvDim);
                valueCache[l] = F32FloatTensor.allocate(positions * kvDim);
                batchK[l] = F32FloatTensor.allocate(c * kvDim);
                batchV[l] = F32FloatTensor.allocate(c * kvDim);
            }
        }

        @Override public int latestToken() { return latestToken; }

        @Override public void latestToken(int token) { this.latestToken = token; }
    }

    // === Loading ===

    static GptOss loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load gpt-oss model")) {
            try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
                return loadModel(fileChannel, gguf, contextLength, true);
            }
        }
    }

    static GptOss loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        LFMTokenizer tokenizer = new LFMTokenizer(gguf);
        String arch = "gpt-oss";

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) {
            contextLength = modelContextLength;
        }
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
        boolean[] isSWA = new boolean[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            isSWA[i] = (i % 2) == 0;
        }

        Configuration config = new Configuration(embeddingLength, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                headSize, tokenizer.vocabularySize(), contextLength, rmsNormEps, ropeTheta, ropeScalingFactor,
                ropeOrigCtx, slidingWindow, isSWA, expertCount, expertUsedCount, expertFeedForwardLength, expertWeightsScale);

        if (!loadWeightsFlag) {
            return new GptOss(config, tokenizer, null);
        }
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        return new GptOss(config, tokenizer, loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
        int n = config.numberOfLayers;
        FloatTensor tokenEmbeddingTable = ModelLoader.loadQuantized(tensors.get("token_embd.weight"));
        FloatTensor outputWeight = tensors.containsKey("output.weight")
                ? ModelLoader.loadQuantized(tensors.get("output.weight")) : tokenEmbeddingTable;
        float[][] rope = precomputeYarnRope(config.contextLength, config.headSize, config.ropeTheta,
                config.ropeScalingFactor, config.ropeOrigCtx);

        return new Weights(
                tokenEmbeddingTable, outputWeight,
                ModelLoader.toF32Tensor(tensors.get("output_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".post_attention_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_q.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_k.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_v.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_output.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_q.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_k.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_v.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_output.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_sinks.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate_inp.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate_exps.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up_exps.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down_exps.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_gate_inp.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_gate_exps.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_up_exps.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_down_exps.bias")),
                rope[0], rope[1]);
    }
}
