// Qwen3.5 ("qwen35" / "qwen35moe") support: a hybrid gated-delta-net (linear-attention) + periodic
// full-attention transformer, dense or MoE. Kept entirely behind the Model seam; ported from the
// reference ../qwen35.java/Qwen35.java. Layers are SSM (gated delta-net) by default; every
// full_attention_interval-th layer is full softmax attention. Single-token forward (batchCapacity 1):
// the delta-net recurrence is inherently sequential, so prefill ingests one token at a time.
package com.llama4j;

import com.qxotic.format.gguf.GGUF;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

final class Qwen35 implements Model {

    private final Configuration configuration;
    private final LFMTokenizer tokenizer;
    private final Weights weights;

    Qwen35(Configuration configuration, LFMTokenizer tokenizer, Weights weights) {
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
    static final boolean SINGLE_TOKEN_PREFILL = System.getProperty("qwen.singleTokenPrefill") != null;

    /** The gated delta-net recurrence is sequential, but the per-token projections (Q/K/V, conv,
     *  FFN/MoE) batch into GEMMs; only the recurrence stays sequential within a chunk. */
    @Override
    public int batchCapacity() {
        return SINGLE_TOKEN_PREFILL ? 1 : Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
    }

    @Override
    public State createNewState() {
        State state = new State(configuration);
        // Qwen3.5 has no BOS (add_bos=false): -1 = "no prior token", so the engine's prefill feeds the
        // rendered prompt verbatim without prepending anything.
        state.latestToken = -1;
        return state;
    }

    @Override
    public void ingest(InferenceState state, int[] tokens, int tokenOffset, int startPosition, int sequenceLength) {
        State s = (State) state;
        if (sequenceLength > s.capacity) {
            throw new IllegalArgumentException("sequenceLength " + sequenceLength + " exceeds batch capacity " + s.capacity);
        }
        if (SINGLE_TOKEN_PREFILL || sequenceLength == 1) {
            for (int i = 0; i < sequenceLength; i++) {
                forward(s, tokens[tokenOffset + i], startPosition + i);
            }
        } else {
            forwardBatch(s, tokens, tokenOffset, startPosition, sequenceLength);
        }
        s.latestToken = tokens[tokenOffset + sequenceLength - 1];
        s.logitsValid = false;
    }

    /** Logits for the last ingested token: forward leaves the final residual in {@code state.x};
     *  this finalizes the output norm + head. Idempotent between ingests. */
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
        for (String name : new String[]{"<|im_end|>", "<|endoftext|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        return stops;
    }

    // === Math helpers ===

    static void rmsnorm(FloatTensor out, FloatTensor x, F32FloatTensor weight, int size, float eps) {
        rmsnorm(out, 0, x, 0, weight, size, eps);
    }

    static void rmsnorm(FloatTensor out, int outOffset, FloatTensor x, int xOffset, F32FloatTensor weight, int size, float eps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss = (float) (1.0 / Math.sqrt(ss / size + eps));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, weight.getFloat(i) * ss * x.getFloat(xOffset + i));
        }
    }

    /** Interleaved-pair (GPT-J) rotary over the first {@code 2*ropeHalf} dims of one head. For
     *  text-only decoding MRoPE reduces to standard RoPE (the 3D position deltas collapse to pos). */
    // === Forward ===

    private void forward(State state, int token, int position) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;

        w.tokenEmbeddingTable.copyTo(token * dim, state.x, 0, dim);

        for (int l = 0; l < config.numberOfLayers; l++) {
            rmsnorm(state.xb, state.x, w.attnNorm[l], dim, eps);
            if (config.isFullAttention[l]) {
                attentionForward(state, l, position);
            } else {
                ssmForward(state, l);
            }
            // attention/SSM residual, then post-attention norm acts as the pre-FFN norm
            state.xb.addInPlace(0, state.x, 0, dim);
            state.xb.copyTo(0, state.x, 0, dim);
            rmsnorm(state.xb, state.xb, w.postAttentionNorm[l], dim, eps);

            if (config.isMoE()) {
                moeForward(state, l);
            } else {
                ffnForward(state, l);
            }
            state.x.addInPlace(state.xb);
        }
        state.lastRowOffset = 0;
    }

    /** Full softmax attention with QK-norm, fused query/output gate (attn_q -> [q | gate]), GQA and
     *  RoPE; output gated by sigmoid(gate). */
    private void attentionForward(State state, int layer, int position) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int headSize = config.headSize;
        int heads = config.numberOfHeads;
        int kvHeads = config.numberOfKeyValueHeads;
        int kvDim = config.kvDim();
        int queryDim = config.queryDim();
        int kvMul = heads / kvHeads;
        float eps = config.rmsNormEps;

        // attn_q -> 2*queryDim, interleaved per head as [q(headSize) | gate(headSize)]
        w.attnQ[layer].matmul(state.xb, state.q, 2 * queryDim, dim);
        float[] gateArr = state.attnGateArr;
        for (int h = 0; h < heads; h++) {
            int base = h * 2 * headSize;
            for (int d = 0; d < headSize; d++) {
                gateArr[h * headSize + d] = state.q.getFloat(base + headSize + d);
                state.q.setFloat(h * headSize + d, state.q.getFloat(base + d));
            }
        }
        for (int h = 0; h < heads; h++) {
            rmsnorm(state.q, h * headSize, state.q, h * headSize, w.attnQNorm[layer], headSize, eps);
        }
        w.attnK[layer].matmul(state.xb, state.k, kvDim, dim);
        w.attnV[layer].matmul(state.xb, state.v, kvDim, dim);
        for (int h = 0; h < kvHeads; h++) {
            rmsnorm(state.k, h * headSize, state.k, h * headSize, w.attnKNorm[layer], headSize, eps);
        }
        if (w.ropeHalf > 0) {
            for (int h = 0; h < heads; h++) {
                RoPE.applyInterleaved(state.q, h * headSize, position, w.ropeCr, w.ropeCi, w.ropeHalf);
            }
            for (int h = 0; h < kvHeads; h++) {
                RoPE.applyInterleaved(state.k, h * headSize, position, w.ropeCr, w.ropeCi, w.ropeHalf);
            }
        }
        state.k.copyTo(0, state.keyCache[layer], position * kvDim, kvDim);
        state.v.copyTo(0, state.valueCache[layer], position * kvDim, kvDim);

        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        float attScale = 1.0f / (float) Math.sqrt(headSize);
        Parallel.parallelFor(0, heads, h -> {
            int qOffset = h * headSize;
            int attOffset = h * config.contextLength;
            int kvHeadOffset = (h / kvMul) * headSize;
            for (int t = 0; t <= position; t++) {
                float score = state.q.dot(qOffset, keyCache, t * kvDim + kvHeadOffset, headSize) * attScale;
                state.att.setFloat(attOffset + t, score);
            }
            state.att.softmaxInPlace(attOffset, position + 1);
            state.xb2.fillInPlace(qOffset, headSize, 0f);
            for (int t = 0; t <= position; t++) {
                state.xb2.saxpyInPlace(qOffset, valueCache, t * kvDim + kvHeadOffset, headSize, state.att.getFloat(attOffset + t));
            }
        });

        for (int i = 0; i < queryDim; i++) {
            state.xb2.setFloat(i, state.xb2.getFloat(i) * Activations.sigmoid(gateArr[i]));
        }
        w.attnOutput[layer].matmul(state.xb2, state.xb, dim, queryDim);
    }

    /** Gated delta-net (linear-attention) layer: depthwise causal conv -> SiLU -> per-group L2-norm
     *  of Q/K -> tile to value heads -> delta-net recurrence over a [headVDim,headVDim] state ->
     *  SiLU(z)-gated RMSNorm -> output projection. */
    private void ssmForward(State state, int layer) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int dInner = config.ssmInnerSize;
        int nGroup = config.ssmGroupCount;
        int dtRank = config.ssmTimeStepRank;
        int dState = config.ssmStateSize;
        int convKernel = config.ssmConvKernel;
        int headVDim = config.headVDim();
        int convChannels = config.convChannels();
        int kOff = dState * nGroup;
        int vOff = 2 * dState * nGroup;
        float eps = config.rmsNormEps;

        // 1. QKV projection (feeds the conv) and 2. z gate projection
        w.attnQkv[layer].matmul(state.xb, state.ssmQkv, convChannels, dim);
        float[] z = state.ssmZ;
        w.attnGate[layer].matmul(state.xb, state.ssmTmp, dInner, dim);
        for (int i = 0; i < dInner; i++) z[i] = state.ssmTmp.getFloat(i);

        // 3. causal depthwise 1D conv over the cached history + current step
        FloatTensor convState = state.ssmConvState[layer];
        F32FloatTensor convWeight = w.ssmConv1d[layer];
        FloatTensor qkv = state.ssmQkv;
        float[] convOut = state.ssmConvOut;
        Parallel.parallelFor(0, convChannels, c -> {
            float sum = 0;
            int wOff = c * convKernel;
            for (int k = 0; k < convKernel - 1; k++) {
                sum += convWeight.getFloat(wOff + k) * convState.getFloat(k * convChannels + c);
            }
            sum += convWeight.getFloat(wOff + (convKernel - 1)) * qkv.getFloat(c);
            convOut[c] = Activations.silu(sum);
        });
        // update conv ring (per channel: shift left, append current qkv as newest)
        Parallel.parallelFor(0, convChannels, c -> {
            for (int k = 0; k < convKernel - 2; k++) {
                convState.setFloat(k * convChannels + c, convState.getFloat((k + 1) * convChannels + c));
            }
            convState.setFloat((convKernel - 2) * convChannels + c, qkv.getFloat(c));
        });

        // 4. split + per-group L2-norm of Q,K (Q folds in 1/sqrt(headVDim)); 5. tile nGroup -> dtRank
        float scale = (float) (1.0 / Math.sqrt(headVDim));
        float[] qGroup = state.ssmQGroup, kGroup = state.ssmKGroup;
        Parallel.parallelFor(0, nGroup, h -> {
            float qNormSq = 0, kNormSq = 0;
            int hOff = h * headVDim;
            for (int d = 0; d < headVDim; d++) {
                float qv = convOut[hOff + d];
                float kv = convOut[kOff + hOff + d];
                qNormSq += qv * qv;
                kNormSq += kv * kv;
            }
            float qInv = (float) (1.0 / Math.sqrt(qNormSq + eps)) * scale;
            float kInv = (float) (1.0 / Math.sqrt(kNormSq + eps));
            for (int d = 0; d < headVDim; d++) {
                qGroup[hOff + d] = convOut[hOff + d] * qInv;
                kGroup[hOff + d] = convOut[kOff + hOff + d] * kInv;
            }
        });
        float[] qArr = state.ssmQ, kArr = state.ssmK, vArr = state.ssmV;
        Parallel.parallelFor(0, dtRank, h -> {
            int dstOff = h * headVDim, srcOff = (h % nGroup) * headVDim, vSrc = vOff + h * headVDim;
            for (int d = 0; d < headVDim; d++) {
                qArr[dstOff + d] = qGroup[srcOff + d];
                kArr[dstOff + d] = kGroup[srcOff + d];
                vArr[dstOff + d] = convOut[vSrc + d];
            }
        });

        // 6. gate = softplus(alpha@x + dt_bias) * A ; beta = sigmoid(beta@x)
        w.ssmAlpha[layer].matmul(state.xb, state.ssmTmp, dtRank, dim);
        float[] gate = state.ssmGate;
        for (int h = 0; h < dtRank; h++) {
            gate[h] = Activations.softplus(state.ssmTmp.getFloat(h) + w.ssmDtBias[layer].getFloat(h)) * w.ssmA[layer].getFloat(h);
        }
        w.ssmBeta[layer].matmul(state.xb, state.ssmTmp, dtRank, dim);
        float[] beta = state.ssmBeta;
        for (int h = 0; h < dtRank; h++) {
            beta[h] = Activations.sigmoid(state.ssmTmp.getFloat(h));
        }

        // 7. delta-net recurrence per head; state element (i,j,h) at h*HV^2 + j*HV + i
        float[] output = state.ssmOutput;
        FloatTensor ssmState = state.ssmState[layer];
        float[] sk = state.ssmSk, d = state.ssmD;   // per-head scratch (sized dtRank*headVDim)
        Parallel.parallelFor(0, dtRank, h -> {
            float expGate = (float) Math.exp(gate[h]);
            float betaH = beta[h];
            int stateBase = h * headVDim * headVDim;
            int headOff = h * headVDim;
            for (int idx = 0; idx < headVDim * headVDim; idx++) {
                int si = stateBase + idx;
                ssmState.setFloat(si, ssmState.getFloat(si) * expGate);
            }
            for (int j = 0; j < headVDim; j++) {
                float sum = 0;
                for (int i = 0; i < headVDim; i++) {
                    sum += ssmState.getFloat(stateBase + j * headVDim + i) * kArr[headOff + i];
                }
                sk[headOff + j] = sum;
            }
            for (int i = 0; i < headVDim; i++) {
                d[headOff + i] = (vArr[headOff + i] - sk[headOff + i]) * betaH;
            }
            for (int i = 0; i < headVDim; i++) {
                float ki = kArr[headOff + i];
                for (int j = 0; j < headVDim; j++) {
                    int si = stateBase + j * headVDim + i;
                    ssmState.setFloat(si, ssmState.getFloat(si) + ki * d[headOff + j]);
                }
            }
            for (int j = 0; j < headVDim; j++) {
                float sum = 0;
                for (int i = 0; i < headVDim; i++) {
                    sum += ssmState.getFloat(stateBase + j * headVDim + i) * qArr[headOff + i];
                }
                output[headOff + j] = sum;
            }
        });

        // 8. SiLU(z)-gated RMSNorm per head, 9. output projection
        Parallel.parallelFor(0, dtRank, h -> {
            int headOff = h * headVDim;
            float ss = 0;
            for (int dd = 0; dd < headVDim; dd++) {
                float val = output[headOff + dd];
                ss += val * val;
            }
            float invRms = (float) (1.0 / Math.sqrt(ss / headVDim + eps));
            for (int dd = 0; dd < headVDim; dd++) {
                float normed = output[headOff + dd] * invRms * w.ssmNorm[layer].getFloat(dd);
                state.ssmTmp.setFloat(headOff + dd, normed * Activations.silu(z[headOff + dd]));
            }
        });
        w.ssmOut[layer].matmul(state.ssmTmp, state.xb, dim, dInner);
    }

    /** Dense SwiGLU FFN. */
    private void ffnForward(State state, int layer) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int hiddenDim = config.hiddenDim;
        w.ffnGate[layer].matmul(state.xb, state.xb2, hiddenDim, dim);
        w.ffnUp[layer].matmul(state.xb, state.ffnUp, hiddenDim, dim);
        state.xb2.siluMultiplyInPlace(0, state.ffnUp, 0, hiddenDim);
        w.ffnDown[layer].matmul(state.xb2, state.xb, dim, hiddenDim);
    }

    /** Top-k expert MoE (softmax over all experts, top-k, renormalize) + optional shared expert. */
    private void moeForward(State state, int layer) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int expertFFN = config.expertFeedForwardLength;
        int numExperts = config.expertCount;
        int topK = Math.min(config.expertUsedCount, numExperts);

        FloatTensor routerLogits = state.moeRouterLogits;
        w.moeRouter[layer].matmul(state.xb, routerLogits, numExperts, dim);
        routerLogits.softmaxInPlace(0, numExperts);

        int[] topExperts = state.moeTopExperts;
        float[] topWeights = state.moeTopWeights;
        for (int i = 0; i < topK; i++) {
            topExperts[i] = -1;
            topWeights[i] = Float.NEGATIVE_INFINITY;
        }
        for (int e = 0; e < numExperts; e++) {
            float prob = routerLogits.getFloat(e);
            int insertPos = -1;
            for (int k = 0; k < topK; k++) {
                if (prob > topWeights[k]) {
                    insertPos = k;
                    break;
                }
            }
            if (insertPos >= 0) {
                for (int k = topK - 1; k > insertPos; k--) {
                    topWeights[k] = topWeights[k - 1];
                    topExperts[k] = topExperts[k - 1];
                }
                topWeights[insertPos] = prob;
                topExperts[insertPos] = e;
            }
        }
        float topKSum = 0f;
        for (int i = 0; i < topK; i++) topKSum += topWeights[i];
        float invTopK = topKSum == 0f ? 0f : 1f / topKSum;

        FloatTensor moeOutput = state.moeOutput;
        moeOutput.fillInPlace(0, dim, 0f);
        int gateUpStride = expertFFN * dim;
        int downStride = dim * expertFFN;
        for (int k = 0; k < topK; k++) {
            int expertIdx = topExperts[k];
            if (expertIdx < 0) continue;
            float weight = topWeights[k] * invTopK;
            if (weight <= 0f) continue;
            int gateUpOffset = expertIdx * gateUpStride;
            int downOffset = expertIdx * downStride;
            w.moeExpertGate[layer].matmul(state.xb, state.moeGateResult, expertFFN, dim, gateUpOffset);
            w.moeExpertUp[layer].matmul(state.xb, state.moeUpResult, expertFFN, dim, gateUpOffset);
            state.moeGateResult.siluMultiplyInPlace(0, state.moeUpResult, 0, expertFFN);
            w.moeExpertDown[layer].matmul(state.moeGateResult, state.moeExpertOut, dim, expertFFN, downOffset);
            moeOutput.saxpyInPlace(0, state.moeExpertOut, 0, dim, weight);
        }

        if (config.expertSharedFeedForwardLength > 0 && w.moeSharedGate[layer] != null) {
            int sharedFFN = config.expertSharedFeedForwardLength;
            w.moeSharedGate[layer].matmul(state.xb, state.moeSharedGate, sharedFFN, dim);
            w.moeSharedUp[layer].matmul(state.xb, state.moeSharedUp, sharedFFN, dim);
            state.moeSharedGate.siluMultiplyInPlace(0, state.moeSharedUp, 0, sharedFFN);
            w.moeSharedDown[layer].matmul(state.moeSharedGate, state.moeSharedOut, dim, sharedFFN);
            float sharedScale = 1.0f;
            if (w.moeSharedInputGate[layer] != null) {
                w.moeSharedInputGate[layer].matmul(state.xb, state.moeSharedInputGate, 1, dim);
                sharedScale = Activations.sigmoid(state.moeSharedInputGate.getFloat(0));
            }
            moeOutput.saxpyInPlace(0, state.moeSharedOut, 0, dim, sharedScale);
        }

        moeOutput.copyTo(0, state.xb, 0, dim);
    }

    // === Batched forward (prompt processing): seqLen tokens in one pass ===

    /**
     * Processes {@code seqLen} tokens at positions {@code [startPos, startPos+seqLen)} in a single
     * pass. The per-token projections (Q/K/V/O, conv, gates, FFN/MoE) become GEMMs over the chunk's
     * rows; the gated delta-net recurrence stays sequential over rows (the SSM state carries
     * forward), and full-attention layers run causal flash attention against the contiguous KV
     * cache. Token-exact vs the single-token {@link #forward}. Leaves the post-final-layer residual
     * in {@code x[lastRowOffset]}; {@link #computeLogits} finalizes the last row.
     */
    void forwardBatch(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;

        // Embed all rows
        for (int s = 0; s < seqLen; s++) {
            w.tokenEmbeddingTable.copyTo(tokens[tokenOffset + s] * dim, state.x, s * dim, dim);
        }

        for (int l = 0; l < config.numberOfLayers; l++) {
            F32FloatTensor attNormW = w.attnNorm[l], postW = w.postAttentionNorm[l];
            int fDim = dim;
            // attention/SSM norm (rows independent -> parallel)
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, s * fDim, state.x, s * fDim, attNormW, fDim, eps));

            if (config.isFullAttention[l]) {
                attentionForwardBatch(state, l, startPos, seqLen);
            } else {
                ssmForwardBatch(state, l, seqLen);
            }
            // sublayer residual, then post-attention norm acts as the pre-FFN norm (mirror forward)
            state.xb.addInPlace(0, state.x, 0, seqLen * dim);
            state.xb.copyTo(0, state.x, 0, seqLen * dim);
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, s * fDim, state.xb, s * fDim, postW, fDim, eps));

            if (config.isMoE()) {
                moeForwardBatch(state, l, seqLen);
            } else {
                ffnForwardBatch(state, l, seqLen);
            }
            state.x.addInPlace(0, state.xb, 0, seqLen * dim);
        }

        state.lastRowOffset = (seqLen - 1) * dim;
        state.logitsValid = false;
    }

    /** Batched full attention: GEMM Q/K/V projections, per-row QK-norm + RoPE, KV written to the
     *  contiguous cache, then causal flash attention over [0, startPos+seqLen), output gated by
     *  sigmoid(gate) and projected. Mirrors the single-token {@link #attentionForward}. */
    private void attentionForwardBatch(State state, int layer, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int headSize = config.headSize;
        int heads = config.numberOfHeads;
        int kvHeads = config.numberOfKeyValueHeads;
        int kvDim = config.kvDim();
        int queryDim = config.queryDim();
        int kvMul = heads / kvHeads;
        float eps = config.rmsNormEps;

        // q proj -> [q(headSize) | gate(headSize)] interleaved per head, into state.q (2*queryDim/row)
        w.attnQ[layer].gemm(state.xb, dim, state.q, 2 * queryDim, seqLen, 2 * queryDim, dim);
        // k/v proj into k/v (kvDim/row)
        w.attnK[layer].gemm(state.xb, dim, state.k, kvDim, seqLen, kvDim, dim);
        w.attnV[layer].gemm(state.xb, dim, state.v, kvDim, seqLen, kvDim, dim);

        int fHeadSz = headSize, fHeads = heads, fKvHeads = kvHeads, fQDim = queryDim, fKvDim = kvDim, fStart = startPos;
        F32FloatTensor qNormW = w.attnQNorm[layer], kNormW = w.attnKNorm[layer];
        float[] gateArr = state.attnGateArr;
        // deinterleave [q|gate] -> queries into state.attnQ, gates into gateArr (per row/head)
        Parallel.forRows(seqLen, s -> {
            int qBase = s * 2 * fQDim;
            int qDst = s * fQDim;
            for (int h = 0; h < fHeads; h++) {
                int base = qBase + h * 2 * fHeadSz;
                for (int d = 0; d < fHeadSz; d++) {
                    gateArr[qDst + h * fHeadSz + d] = state.q.getFloat(base + fHeadSz + d);
                    state.attnQ.setFloat(qDst + h * fHeadSz + d, state.q.getFloat(base + d));
                }
            }
            // qk-norm + rope per head (queries)
            for (int h = 0; h < fHeads; h++) {
                rmsnorm(state.attnQ, qDst + h * fHeadSz, state.attnQ, qDst + h * fHeadSz, qNormW, fHeadSz, eps);
            }
            // kv-norm per kv head
            int kBase = s * fKvDim;
            for (int h = 0; h < fKvHeads; h++) {
                rmsnorm(state.k, kBase + h * fHeadSz, state.k, kBase + h * fHeadSz, kNormW, fHeadSz, eps);
            }
            if (w.ropeHalf > 0) {
                for (int h = 0; h < fHeads; h++) {
                    RoPE.applyInterleaved(state.attnQ, qDst + h * fHeadSz, fStart + s, w.ropeCr, w.ropeCi, w.ropeHalf);
                }
                for (int h = 0; h < fKvHeads; h++) {
                    RoPE.applyInterleaved(state.k, kBase + h * fHeadSz, fStart + s, w.ropeCr, w.ropeCi, w.ropeHalf);
                }
            }
        });

        // write this chunk's K/V into the contiguous full-context cache
        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        for (int s = 0; s < seqLen; s++) {
            state.k.copyTo(s * kvDim, keyCache, (startPos + s) * kvDim, kvDim);
            state.v.copyTo(s * kvDim, valueCache, (startPos + s) * kvDim, kvDim);
        }

        // causal flash attention over the contiguous cache [0, startPos+seqLen) -> state.attnOut
        FlashAttention.causalPrefill((F32FloatTensor) state.attnQ, (F32FloatTensor) state.attnOut,
                keyCache, valueCache, configuration.numberOfHeads, startPos, seqLen, headSize, kvDim, queryDim, kvMul);

        // sigmoid(gate) * attn output, then output projection into xb
        int total = seqLen * queryDim;
        for (int i = 0; i < total; i++) {
            state.attnOut.setFloat(i, state.attnOut.getFloat(i) * Activations.sigmoid(gateArr[i]));
        }
        w.attnOutput[layer].gemm(state.attnOut, queryDim, state.xb, dim, seqLen, dim, queryDim);
    }

    /** Batched gated delta-net: GEMM projections + batched causal conv, then a SEQUENTIAL recurrence
     *  over rows (state carries forward), parallel over heads within each row. Token-exact vs the
     *  single-token {@link #ssmForward}. */
    private void ssmForwardBatch(State state, int layer, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int dInner = config.ssmInnerSize;
        int nGroup = config.ssmGroupCount;
        int dtRank = config.ssmTimeStepRank;
        int dState = config.ssmStateSize;
        int convKernel = config.ssmConvKernel;
        int headVDim = config.headVDim();
        int convChannels = config.convChannels();
        int kOff = dState * nGroup;
        int vOff = 2 * dState * nGroup;
        float eps = config.rmsNormEps;

        // 1. QKV projection (feeds the conv) and 2. z gate projection (into ssmZ per row)
        w.attnQkv[layer].gemm(state.xb, dim, state.ssmQkv, convChannels, seqLen, convChannels, dim);
        w.attnGate[layer].gemm(state.xb, dim, state.ssmTmp, dInner, seqLen, dInner, dim);
        for (int i = 0; i < seqLen * dInner; i++) state.ssmZ[i] = state.ssmTmp.getFloat(i);

        // 3. causal depthwise 1D conv over the cached history (pre-chunk) + the chunk rows, then SiLU.
        // For row s, channel c: sum over taps of qkv[s-(K-1)+k]; positions < 0 read the conv ring.
        FloatTensor convState = state.ssmConvState[layer];
        F32FloatTensor convWeight = w.ssmConv1d[layer];
        FloatTensor qkv = state.ssmQkv;
        float[] convOut = state.ssmConvOut;
        int fK = convKernel, fCC = convChannels, fHist = convKernel - 1;
        Parallel.parallelFor(0, convChannels, c -> {
            int wOff = c * fK;
            for (int s = 0; s < seqLen; s++) {
                float sum = 0;
                for (int k = 0; k < fK; k++) {
                    int pos = s - fHist + k;     // input row index relative to the chunk
                    float in = pos < 0 ? convState.getFloat((pos + fHist) * fCC + c)
                                       : qkv.getFloat(pos * fCC + c);
                    sum += convWeight.getFloat(wOff + k) * in;
                }
                convOut[s * fCC + c] = Activations.silu(sum);
            }
        });
        // roll conv ring: keep the last (K-1) qkv rows of the chunk (positions read the ring above)
        Parallel.parallelFor(0, convChannels, c -> {
            for (int k = 0; k < fHist; k++) {
                int pos = seqLen - fHist + k;
                float v = pos < 0 ? convState.getFloat((pos + fHist) * fCC + c) : qkv.getFloat(pos * fCC + c);
                convState.setFloat(k * fCC + c, v);
            }
        });

        // 4. split + per-group L2-norm of Q,K; 5. tile nGroup -> dtRank (per row)
        float scale = (float) (1.0 / Math.sqrt(headVDim));
        float[] qGroup = state.ssmQGroup, kGroup = state.ssmKGroup;
        float[] qArr = state.ssmQ, kArr = state.ssmK, vArr = state.ssmV;
        int fNGroup = nGroup, fDtRank = dtRank, fHV = headVDim, fKOff = kOff, fVOff = vOff, fDInner = dInner;
        Parallel.forRows(seqLen, s -> {
            int cBase = s * fCC, gBase = s * fNGroup * fHV, dBase = s * fDtRank * fHV;
            for (int h = 0; h < fNGroup; h++) {
                float qNormSq = 0, kNormSq = 0;
                int hOff = h * fHV;
                for (int d = 0; d < fHV; d++) {
                    float qv = convOut[cBase + hOff + d];
                    float kv = convOut[cBase + fKOff + hOff + d];
                    qNormSq += qv * qv;
                    kNormSq += kv * kv;
                }
                float qInv = (float) (1.0 / Math.sqrt(qNormSq + eps)) * scale;
                float kInv = (float) (1.0 / Math.sqrt(kNormSq + eps));
                for (int d = 0; d < fHV; d++) {
                    qGroup[gBase + hOff + d] = convOut[cBase + hOff + d] * qInv;
                    kGroup[gBase + hOff + d] = convOut[cBase + fKOff + hOff + d] * kInv;
                }
            }
            for (int h = 0; h < fDtRank; h++) {
                int dstOff = dBase + h * fHV, srcOff = gBase + (h % fNGroup) * fHV, vSrc = cBase + fVOff + h * fHV;
                for (int d = 0; d < fHV; d++) {
                    qArr[dstOff + d] = qGroup[srcOff + d];
                    kArr[dstOff + d] = kGroup[srcOff + d];
                    vArr[dstOff + d] = convOut[vSrc + d];
                }
            }
        });

        // 6. gate = softplus(alpha@x + dt_bias) * A ; beta = sigmoid(beta@x), per row (into ssmTmp)
        float[] gate = state.ssmGate, beta = state.ssmBeta;
        w.ssmAlpha[layer].gemm(state.xb, dim, state.ssmTmp, dtRank, seqLen, dtRank, dim);
        for (int s = 0; s < seqLen; s++) {
            for (int h = 0; h < dtRank; h++) {
                gate[s * dtRank + h] = Activations.softplus(state.ssmTmp.getFloat(s * dtRank + h) + w.ssmDtBias[layer].getFloat(h)) * w.ssmA[layer].getFloat(h);
            }
        }
        w.ssmBeta[layer].gemm(state.xb, dim, state.ssmTmp, dtRank, seqLen, dtRank, dim);
        for (int s = 0; s < seqLen; s++) {
            for (int h = 0; h < dtRank; h++) {
                beta[s * dtRank + h] = Activations.sigmoid(state.ssmTmp.getFloat(s * dtRank + h));
            }
        }

        // 7. delta-net recurrence: SEQUENTIAL over rows (state carries forward), parallel over heads.
        // Per-head sk/d scratch reuses the preallocated state.ssmSk/ssmD sliced by head (distinct
        // [h*headVDim] regions -> race-free, no per-row allocation).
        float[] output = state.ssmOutput;
        float[] skArr = state.ssmSk, dArr = state.ssmD;
        FloatTensor ssmState = state.ssmState[layer];
        for (int s = 0; s < seqLen; s++) {
            int dBase = s * fDtRank * fHV, gBase = s * fDtRank;
            Parallel.parallelFor(0, fDtRank, h -> {
                int sd = h * fHV;
                float expGate = (float) Math.exp(gate[gBase + h]);
                float betaH = beta[gBase + h];
                int stateBase = h * fHV * fHV;
                int headOff = dBase + h * fHV;
                for (int idx = 0; idx < fHV * fHV; idx++) {
                    int si = stateBase + idx;
                    ssmState.setFloat(si, ssmState.getFloat(si) * expGate);
                }
                for (int j = 0; j < fHV; j++) {
                    float sum = 0;
                    for (int i = 0; i < fHV; i++) {
                        sum += ssmState.getFloat(stateBase + j * fHV + i) * kArr[headOff + i];
                    }
                    skArr[sd + j] = sum;
                }
                for (int i = 0; i < fHV; i++) {
                    dArr[sd + i] = (vArr[headOff + i] - skArr[sd + i]) * betaH;
                }
                for (int i = 0; i < fHV; i++) {
                    float ki = kArr[headOff + i];
                    for (int j = 0; j < fHV; j++) {
                        int si = stateBase + j * fHV + i;
                        ssmState.setFloat(si, ssmState.getFloat(si) + ki * dArr[sd + j]);
                    }
                }
                for (int j = 0; j < fHV; j++) {
                    float sum = 0;
                    for (int i = 0; i < fHV; i++) {
                        sum += ssmState.getFloat(stateBase + j * fHV + i) * qArr[headOff + i];
                    }
                    output[headOff + j] = sum;
                }
            });
        }

        // 8. SiLU(z)-gated RMSNorm per head; 9. output projection (per row)
        float[] z = state.ssmZ;
        Parallel.forRows(seqLen, s -> {
            int dBase = s * fDtRank * fHV;
            for (int h = 0; h < fDtRank; h++) {
                int headOff = dBase + h * fHV;
                float ss = 0;
                for (int d = 0; d < fHV; d++) {
                    float val = output[headOff + d];
                    ss += val * val;
                }
                float invRms = (float) (1.0 / Math.sqrt(ss / fHV + eps));
                for (int d = 0; d < fHV; d++) {
                    float normed = output[headOff + d] * invRms * w.ssmNorm[layer].getFloat(d);
                    state.ssmTmp.setFloat(headOff + d, normed * Activations.silu(z[headOff + d]));
                }
            }
        });
        w.ssmOut[layer].gemm(state.ssmTmp, dInner, state.xb, dim, seqLen, dim, dInner);
    }

    /** Batched dense SwiGLU FFN: gate/up/down GEMMs over the chunk, SiLU-multiply per row. */
    private void ffnForwardBatch(State state, int layer, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int hiddenDim = config.hiddenDim;
        w.ffnGate[layer].gemm(state.xb, dim, state.ffnGate, hiddenDim, seqLen, hiddenDim, dim);
        w.ffnUp[layer].gemm(state.xb, dim, state.ffnUp, hiddenDim, seqLen, hiddenDim, dim);
        int fHidden = hiddenDim;
        Parallel.forRows(seqLen, s -> state.ffnGate.siluMultiplyInPlace(s * fHidden, state.ffnUp, s * fHidden, fHidden));
        w.ffnDown[layer].gemm(state.ffnGate, hiddenDim, state.xb, dim, seqLen, dim, hiddenDim);
    }

    /** Batched top-k MoE + shared expert: router GEMM, per-row softmax+top-k+renorm, CSR
     *  gather-by-expert GEMMs, plus the batched shared expert with its sigmoid input gate. Sums
     *  experts + shared*sharedScale into {@code xb}. Token-exact vs the single-token {@link #moeForward}. */
    private void moeForwardBatch(State state, int layer, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int expertFFN = config.expertFeedForwardLength;
        int numExperts = config.expertCount;
        int topK = Math.min(config.expertUsedCount, numExperts);
        float eps = config.rmsNormEps;
        int gateUpStride = expertFFN * dim;
        int downStride = dim * expertFFN;

        // The pre-FFN normed input is already in state.xb (post-attention norm applied by caller).
        // Snapshot it as the per-row expert/router/shared input (CSR gather scrambles row order).
        state.xb.copyTo(0, state.moeInputB, 0, seqLen * dim);

        // router: softmax over all experts, top-k, renormalize top-k (per row)
        w.moeRouter[layer].gemm(state.moeInputB, dim, state.moeRouterB, numExperts, seqLen, numExperts, dim);
        int[] counts = state.moeExpertCounts;
        java.util.Arrays.fill(counts, 0);
        for (int s = 0; s < seqLen; s++) {
            state.moeRouterB.softmaxInPlace(s * numExperts, numExperts);
            // select top-k by descending prob (stable insertion, matching single-token order)
            int[] topE = state.moeTopExperts;
            float[] topP = state.moeTopWeights;
            for (int i = 0; i < topK; i++) { topE[i] = -1; topP[i] = Float.NEGATIVE_INFINITY; }
            for (int e = 0; e < numExperts; e++) {
                float prob = state.moeRouterB.getFloat(s * numExperts + e);
                int insertPos = -1;
                for (int k = 0; k < topK; k++) {
                    if (prob > topP[k]) { insertPos = k; break; }
                }
                if (insertPos >= 0) {
                    for (int k = topK - 1; k > insertPos; k--) { topP[k] = topP[k - 1]; topE[k] = topE[k - 1]; }
                    topP[insertPos] = prob; topE[insertPos] = e;
                }
            }
            float topKSum = 0f;
            for (int i = 0; i < topK; i++) topKSum += topP[i];
            float invTopK = topKSum == 0f ? 0f : 1f / topKSum;
            for (int ki = 0; ki < topK; ki++) {
                int e = topE[ki];
                state.moeRowTopE[s * topK + ki] = e;
                state.moeRowTopP[s * topK + ki] = (e < 0 ? 0f : topP[ki] * invTopK);
                if (e >= 0) counts[e]++;
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
                if (e < 0) continue;
                int pos = cursor[e]++;
                state.moeRowByExpert[pos] = s;
                state.moeProbByExpert[pos] = state.moeRowTopP[s * topK + ki];
            }
        }

        // experts (grouped): one gate/up/down GEMM per expert over its rows; scatter-add weighted
        state.moeOutB.fillInPlace(0, seqLen * dim, 0f);
        int fDim = dim, fEFF = expertFFN;
        for (int e = 0; e < numExperts; e++) {
            int start = off[e], n = off[e + 1] - start;
            if (n == 0) continue;
            Parallel.forRows(n, j -> state.moeInputB.copyTo(state.moeRowByExpert[start + j] * fDim, state.moeGather, j * fDim, fDim));
            int gateUpOffset = e * gateUpStride;
            int downOffset = e * downStride;
            w.moeExpertGate[layer].gemm(state.moeGather, dim, state.moeGateUpB, expertFFN, n, expertFFN, dim, gateUpOffset);
            w.moeExpertUp[layer].gemm(state.moeGather, dim, state.moeUpB, expertFFN, n, expertFFN, dim, gateUpOffset);
            Parallel.forRows(n, j -> state.moeGateUpB.siluMultiplyInPlace(j * fEFF, state.moeUpB, j * fEFF, fEFF));
            w.moeExpertDown[layer].gemm(state.moeGateUpB, expertFFN, state.moeDownB, dim, n, dim, expertFFN, downOffset);
            Parallel.forRows(n, j -> state.moeOutB.saxpyInPlace(state.moeRowByExpert[start + j] * fDim, state.moeDownB, j * fDim, fDim,
                    state.moeProbByExpert[start + j]));
        }

        // shared expert (batched) + sigmoid input gate, added per row
        if (config.expertSharedFeedForwardLength > 0 && w.moeSharedGate[layer] != null) {
            int sharedFFN = config.expertSharedFeedForwardLength;
            int fSFF = sharedFFN;
            w.moeSharedGate[layer].gemm(state.moeInputB, dim, state.moeSharedGateB, sharedFFN, seqLen, sharedFFN, dim);
            w.moeSharedUp[layer].gemm(state.moeInputB, dim, state.moeSharedUpB, sharedFFN, seqLen, sharedFFN, dim);
            Parallel.forRows(seqLen, s -> state.moeSharedGateB.siluMultiplyInPlace(s * fSFF, state.moeSharedUpB, s * fSFF, fSFF));
            w.moeSharedDown[layer].gemm(state.moeSharedGateB, sharedFFN, state.moeSharedOutB, dim, seqLen, dim, sharedFFN);
            if (w.moeSharedInputGate[layer] != null) {
                w.moeSharedInputGate[layer].gemm(state.moeInputB, dim, state.moeSharedInputGateB, 1, seqLen, 1, dim);
                Parallel.forRows(seqLen, s -> {
                    float sharedScale = Activations.sigmoid(state.moeSharedInputGateB.getFloat(s));
                    state.moeOutB.saxpyInPlace(s * fDim, state.moeSharedOutB, s * fDim, fDim, sharedScale);
                });
            } else {
                state.moeOutB.addInPlace(0, state.moeSharedOutB, 0, seqLen * dim);
            }
        }

        state.moeOutB.copyTo(0, state.xb, 0, seqLen * dim);
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
        final float ropeTheta;
        final int ropeDimensionCount;
        final int hiddenDim;            // dense FFN hidden (0 for MoE)
        final boolean[] isFullAttention;
        final int ssmInnerSize;
        final int ssmGroupCount;
        final int ssmTimeStepRank;
        final int ssmStateSize;
        final int ssmConvKernel;
        final int expertCount;          // 0 = dense
        final int expertUsedCount;
        final int expertFeedForwardLength;
        final int expertSharedFeedForwardLength;

        Configuration(int embeddingLength, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                      int headSize, int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta,
                      int ropeDimensionCount, int hiddenDim, boolean[] isFullAttention, int ssmInnerSize,
                      int ssmGroupCount, int ssmTimeStepRank, int ssmStateSize, int ssmConvKernel,
                      int expertCount, int expertUsedCount, int expertFeedForwardLength, int expertSharedFeedForwardLength) {
            this.embeddingLength = embeddingLength;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.headSize = headSize;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.ropeDimensionCount = ropeDimensionCount;
            this.hiddenDim = hiddenDim;
            this.isFullAttention = isFullAttention;
            this.ssmInnerSize = ssmInnerSize;
            this.ssmGroupCount = ssmGroupCount;
            this.ssmTimeStepRank = ssmTimeStepRank;
            this.ssmStateSize = ssmStateSize;
            this.ssmConvKernel = ssmConvKernel;
            this.expertCount = expertCount;
            this.expertUsedCount = expertUsedCount;
            this.expertFeedForwardLength = expertFeedForwardLength;
            this.expertSharedFeedForwardLength = expertSharedFeedForwardLength;
        }

        int queryDim() {
            return numberOfHeads * headSize;
        }

        int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        int headVDim() {
            return ssmInnerSize / ssmTimeStepRank;
        }

        int convChannels() {
            return ssmInnerSize + 2 * ssmGroupCount * ssmStateSize;
        }

        boolean isMoE() {
            return expertCount > 0;
        }
    }

    // === Weights ===

    static final class Weights {
        final FloatTensor tokenEmbeddingTable;
        final F32FloatTensor outputNorm;
        final FloatTensor outputWeight;
        final F32FloatTensor[] attnNorm, postAttentionNorm;
        // full-attention layers
        final FloatTensor[] attnQ, attnK, attnV, attnOutput;
        final F32FloatTensor[] attnQNorm, attnKNorm;
        // SSM layers
        final FloatTensor[] attnQkv, attnGate, ssmAlpha, ssmBeta, ssmOut;
        final F32FloatTensor[] ssmConv1d, ssmA, ssmDtBias, ssmNorm;
        // dense FFN
        final FloatTensor[] ffnGate, ffnUp, ffnDown;
        // MoE FFN
        final FloatTensor[] moeRouter, moeExpertGate, moeExpertUp, moeExpertDown;
        final FloatTensor[] moeSharedGate, moeSharedUp, moeSharedDown, moeSharedInputGate;
        // RoPE tables
        final float[] ropeCr, ropeCi;
        final int ropeHalf;

        Weights(FloatTensor tokenEmbeddingTable, F32FloatTensor outputNorm, FloatTensor outputWeight,
                F32FloatTensor[] attnNorm, F32FloatTensor[] postAttentionNorm, FloatTensor[] attnQ, FloatTensor[] attnK,
                FloatTensor[] attnV, FloatTensor[] attnOutput, F32FloatTensor[] attnQNorm, F32FloatTensor[] attnKNorm,
                FloatTensor[] attnQkv, FloatTensor[] attnGate, FloatTensor[] ssmAlpha, FloatTensor[] ssmBeta,
                FloatTensor[] ssmOut, F32FloatTensor[] ssmConv1d, F32FloatTensor[] ssmA, F32FloatTensor[] ssmDtBias,
                F32FloatTensor[] ssmNorm, FloatTensor[] ffnGate, FloatTensor[] ffnUp, FloatTensor[] ffnDown,
                FloatTensor[] moeRouter, FloatTensor[] moeExpertGate, FloatTensor[] moeExpertUp, FloatTensor[] moeExpertDown,
                FloatTensor[] moeSharedGate, FloatTensor[] moeSharedUp, FloatTensor[] moeSharedDown,
                FloatTensor[] moeSharedInputGate, float[] ropeCr, float[] ropeCi, int ropeHalf) {
            this.tokenEmbeddingTable = tokenEmbeddingTable;
            this.outputNorm = outputNorm;
            this.outputWeight = outputWeight;
            this.attnNorm = attnNorm;
            this.postAttentionNorm = postAttentionNorm;
            this.attnQ = attnQ;
            this.attnK = attnK;
            this.attnV = attnV;
            this.attnOutput = attnOutput;
            this.attnQNorm = attnQNorm;
            this.attnKNorm = attnKNorm;
            this.attnQkv = attnQkv;
            this.attnGate = attnGate;
            this.ssmAlpha = ssmAlpha;
            this.ssmBeta = ssmBeta;
            this.ssmOut = ssmOut;
            this.ssmConv1d = ssmConv1d;
            this.ssmA = ssmA;
            this.ssmDtBias = ssmDtBias;
            this.ssmNorm = ssmNorm;
            this.ffnGate = ffnGate;
            this.ffnUp = ffnUp;
            this.ffnDown = ffnDown;
            this.moeRouter = moeRouter;
            this.moeExpertGate = moeExpertGate;
            this.moeExpertUp = moeExpertUp;
            this.moeExpertDown = moeExpertDown;
            this.moeSharedGate = moeSharedGate;
            this.moeSharedUp = moeSharedUp;
            this.moeSharedDown = moeSharedDown;
            this.moeSharedInputGate = moeSharedInputGate;
            this.ropeCr = ropeCr;
            this.ropeCi = ropeCi;
            this.ropeHalf = ropeHalf;
        }
    }

    // === State ===

    static final class State implements InferenceState {
        // Chunk-sized scratch (capacity c rows): the residual stream and per-token projections hold
        // one row per chunk token; the single-token reference forward uses row 0. Per-head recurrence
        // scratch (att/ssmSk/ssmD) and the single-token MoE buffers stay single-row and are reused.
        final int capacity;
        final FloatTensor x, xb, xb2, q, k, v, att, logits, ffnUp, ffnGate, ssmQkv, ssmTmp;
        final float[] attnGateArr, ssmZ, ssmConvOut, ssmQ, ssmK, ssmV, ssmQGroup, ssmKGroup, ssmGate, ssmBeta, ssmOutput, ssmSk, ssmD;
        // Batched-attention scratch (chunk rows): queries deinterleaved from q, attention output.
        final FloatTensor attnQ, attnOut;
        final FloatTensor[] keyCache, valueCache, ssmConvState, ssmState;
        // Single-token MoE scratch (used by the seqLen==1 path / reference forward).
        final FloatTensor moeRouterLogits, moeOutput, moeExpertOut, moeGateResult, moeUpResult,
                moeSharedGate, moeSharedUp, moeSharedOut, moeSharedInputGate;
        final int[] moeTopExperts;
        final float[] moeTopWeights;
        // Batched grouped-MoE scratch (chunk-wide; CSR grouping of rows by routed expert).
        final FloatTensor moeInputB, moeRouterB, moeOutB, moeGather, moeGateUpB, moeUpB, moeDownB,
                moeSharedGateB, moeSharedUpB, moeSharedOutB, moeSharedInputGateB;
        final int[] moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeRowTopE;
        final float[] moeProbByExpert, moeRowTopP;
        int latestToken;
        boolean logitsValid;
        int lastRowOffset;     // offset into x of the row whose logits computeLogits finalizes

        State(Configuration config) {
            int c = Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
            this.capacity = c;
            int dim = config.embeddingLength;
            int queryDim = config.queryDim();
            int kvDim = config.kvDim();
            int dInner = config.ssmInnerSize;
            int convChannels = config.convChannels();
            int headVDim = config.headVDim();
            int dtRank = config.ssmTimeStepRank;
            int nGroup = config.ssmGroupCount;
            int hiddenDim = config.hiddenDim;
            int xb2Size = Math.max(queryDim, hiddenDim);

            this.x = F32FloatTensor.allocate(c * dim);
            this.xb = F32FloatTensor.allocate(c * dim);
            this.xb2 = F32FloatTensor.allocate(c * xb2Size);
            this.q = F32FloatTensor.allocate(c * 2 * queryDim);
            this.k = F32FloatTensor.allocate(c * kvDim);
            this.v = F32FloatTensor.allocate(c * kvDim);
            this.att = F32FloatTensor.allocate(config.numberOfHeads * config.contextLength);
            this.logits = F32FloatTensor.allocate(config.vocabularySize);
            this.ffnUp = hiddenDim > 0 ? F32FloatTensor.allocate(c * hiddenDim) : null;
            this.ffnGate = hiddenDim > 0 ? F32FloatTensor.allocate(c * hiddenDim) : null;
            this.attnGateArr = new float[c * queryDim];
            this.attnQ = F32FloatTensor.allocate(c * queryDim);
            this.attnOut = F32FloatTensor.allocate(c * queryDim);

            this.ssmQkv = F32FloatTensor.allocate(c * convChannels);
            this.ssmTmp = F32FloatTensor.allocate(c * dInner);
            this.ssmZ = new float[c * dInner];
            this.ssmConvOut = new float[c * convChannels];
            this.ssmQ = new float[c * dtRank * headVDim];
            this.ssmK = new float[c * dtRank * headVDim];
            this.ssmV = new float[c * dtRank * headVDim];
            this.ssmQGroup = new float[c * nGroup * headVDim];
            this.ssmKGroup = new float[c * nGroup * headVDim];
            this.ssmGate = new float[c * dtRank];
            this.ssmBeta = new float[c * dtRank];
            this.ssmOutput = new float[c * dtRank * headVDim];
            this.ssmSk = new float[dtRank * headVDim];   // per-row recurrence scratch (sequential)
            this.ssmD = new float[dtRank * headVDim];

            if (config.isMoE()) {
                int e = config.expertCount, eff = config.expertFeedForwardLength;
                int sff = Math.max(1, config.expertSharedFeedForwardLength);
                int tk = Math.max(1, config.expertUsedCount);
                this.moeRouterLogits = F32FloatTensor.allocate(e);
                this.moeOutput = F32FloatTensor.allocate(dim);
                this.moeExpertOut = F32FloatTensor.allocate(dim);
                this.moeGateResult = F32FloatTensor.allocate(eff);
                this.moeUpResult = F32FloatTensor.allocate(eff);
                this.moeSharedGate = F32FloatTensor.allocate(sff);
                this.moeSharedUp = F32FloatTensor.allocate(sff);
                this.moeSharedOut = F32FloatTensor.allocate(dim);
                this.moeSharedInputGate = F32FloatTensor.allocate(1);
                this.moeTopExperts = new int[tk];
                this.moeTopWeights = new float[tk];
                // batched grouped-MoE scratch
                this.moeInputB = F32FloatTensor.allocate(c * dim);
                this.moeRouterB = F32FloatTensor.allocate(c * e);
                this.moeOutB = F32FloatTensor.allocate(c * dim);
                this.moeGather = F32FloatTensor.allocate(c * dim);
                this.moeGateUpB = F32FloatTensor.allocate(c * eff);
                this.moeUpB = F32FloatTensor.allocate(c * eff);
                this.moeDownB = F32FloatTensor.allocate(c * dim);
                this.moeSharedGateB = F32FloatTensor.allocate(c * sff);
                this.moeSharedUpB = F32FloatTensor.allocate(c * sff);
                this.moeSharedOutB = F32FloatTensor.allocate(c * dim);
                this.moeSharedInputGateB = F32FloatTensor.allocate(c);
                this.moeExpertCounts = new int[e];
                this.moeExpertOffsets = new int[e + 1];
                this.moeCursor = new int[e];
                this.moeRowByExpert = new int[c * tk];
                this.moeRowTopE = new int[c * tk];
                this.moeProbByExpert = new float[c * tk];
                this.moeRowTopP = new float[c * tk];
            } else {
                this.moeRouterLogits = this.moeOutput = this.moeExpertOut = this.moeGateResult = this.moeUpResult = null;
                this.moeSharedGate = this.moeSharedUp = this.moeSharedOut = this.moeSharedInputGate = null;
                this.moeTopExperts = null;
                this.moeTopWeights = null;
                this.moeInputB = this.moeRouterB = this.moeOutB = this.moeGather = null;
                this.moeGateUpB = this.moeUpB = this.moeDownB = this.moeSharedGateB = this.moeSharedUpB = null;
                this.moeSharedOutB = this.moeSharedInputGateB = null;
                this.moeExpertCounts = this.moeExpertOffsets = this.moeCursor = this.moeRowByExpert = this.moeRowTopE = null;
                this.moeProbByExpert = this.moeRowTopP = null;
            }

            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            this.ssmConvState = new FloatTensor[config.numberOfLayers];
            this.ssmState = new FloatTensor[config.numberOfLayers];
            for (int l = 0; l < config.numberOfLayers; l++) {
                if (config.isFullAttention[l]) {
                    keyCache[l] = F32FloatTensor.allocate(config.contextLength * kvDim);
                    valueCache[l] = F32FloatTensor.allocate(config.contextLength * kvDim);
                } else {
                    ssmConvState[l] = F32FloatTensor.allocate((config.ssmConvKernel - 1) * convChannels);
                    ssmState[l] = F32FloatTensor.allocate(headVDim * headVDim * dtRank);
                }
            }
        }

        @Override public int latestToken() { return latestToken; }

        @Override public void latestToken(int token) { this.latestToken = token; }
    }

    // === Loading ===

    static Qwen35 loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load Qwen3.5 model")) {
            try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
                return loadModel(fileChannel, gguf, contextLength, true);
            }
        }
    }

    static Qwen35 loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        LFMTokenizer tokenizer = new LFMTokenizer(gguf);
        String arch = gguf.getString("general.architecture");

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) {
            contextLength = modelContextLength;
        }
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int numberOfKeyValueHeads = gguf.getValue(int.class, arch + ".attention.head_count_kv");
        int headSize = gguf.getValueOrDefault(int.class, arch + ".attention.key_length", embeddingLength / numberOfHeads);
        float rmsNormEps = gguf.getValueOrDefault(float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 1000000f);
        int ropeDimensionCount = gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);
        int fullAttentionInterval = gguf.getValueOrDefault(int.class, arch + ".full_attention_interval", 4);
        int hiddenDim = gguf.getValueOrDefault(int.class, arch + ".feed_forward_length", 0);

        int ssmInnerSize = gguf.getValueOrDefault(int.class, arch + ".ssm.inner_size", 0);
        int ssmGroupCount = gguf.getValueOrDefault(int.class, arch + ".ssm.group_count", 0);
        int ssmTimeStepRank = gguf.getValueOrDefault(int.class, arch + ".ssm.time_step_rank", 0);
        int ssmStateSize = gguf.getValueOrDefault(int.class, arch + ".ssm.state_size", 0);
        int ssmConvKernel = gguf.getValueOrDefault(int.class, arch + ".ssm.conv_kernel", 0);

        int expertCount = gguf.getValueOrDefault(int.class, arch + ".expert_count", 0);
        int expertUsedCount = gguf.getValueOrDefault(int.class, arch + ".expert_used_count", 0);
        int expertFeedForwardLength = gguf.getValueOrDefault(int.class, arch + ".expert_feed_forward_length", 0);
        int expertSharedFeedForwardLength = gguf.getValueOrDefault(int.class, arch + ".expert_shared_feed_forward_length", 0);

        boolean[] isFullAttention = new boolean[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            isFullAttention[i] = (i + 1) % fullAttentionInterval == 0;
        }

        Configuration config = new Configuration(embeddingLength, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                headSize, tokenizer.vocabularySize(), contextLength, rmsNormEps, ropeTheta, ropeDimensionCount, hiddenDim,
                isFullAttention, ssmInnerSize, ssmGroupCount, ssmTimeStepRank, ssmStateSize, ssmConvKernel,
                expertCount, expertUsedCount, expertFeedForwardLength, expertSharedFeedForwardLength);

        if (!loadWeightsFlag) {
            return new Qwen35(config, tokenizer, null);
        }
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        return new Qwen35(config, tokenizer, loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
        int n = config.numberOfLayers;
        FloatTensor tokenEmbeddingTable = ModelLoader.loadQuantized(tensors.get("token_embd.weight"));
        FloatTensor outputWeight = tensors.containsKey("output.weight")
                ? ModelLoader.loadQuantized(tensors.get("output.weight")) : tokenEmbeddingTable;

        int ropeDim = Math.max(0, Math.min(config.ropeDimensionCount, config.headSize) & ~1);
        Pair<float[], float[]> rope = ropeDim > 0 ? RoPE.precomputeFreqsCis(config.contextLength, ropeDim, config.ropeTheta) : null;

        return new Weights(
                tokenEmbeddingTable,
                ModelLoader.toF32Tensor(tensors.get("output_norm.weight")),
                outputWeight,
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".post_attention_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_q.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_k.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_v.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_output.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_q_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_k_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_qkv.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_gate.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ssm_alpha.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ssm_beta.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ssm_out.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_conv1d.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_a")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_dt.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down.weight")),
                ModelLoader.quantArray(n, i -> ModelLoader.firstPresent(tensors, "blk." + i + ".ffn_gate_inp.weight", "blk." + i + ".ffn_router.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate_exps.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up_exps.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down_exps.weight")),
                ModelLoader.quantArray(n, i -> ModelLoader.firstPresent(tensors, "blk." + i + ".ffn_gate_shexp.weight", "blk." + i + ".ffn_shared_expert_gate.weight")),
                ModelLoader.quantArray(n, i -> ModelLoader.firstPresent(tensors, "blk." + i + ".ffn_up_shexp.weight", "blk." + i + ".ffn_shared_expert_up.weight")),
                ModelLoader.quantArray(n, i -> ModelLoader.firstPresent(tensors, "blk." + i + ".ffn_down_shexp.weight", "blk." + i + ".ffn_shared_expert_down.weight")),
                ModelLoader.quantArray(n, i -> ModelLoader.firstPresent(tensors, "blk." + i + ".ffn_gate_inp_shexp.weight", "blk." + i + ".ffn_shared_expert_gate_inp.weight")),
                rope != null ? rope.first() : null, rope != null ? rope.second() : null, ropeDim / 2);
    }
}
