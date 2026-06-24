// Qwen3.5 ("qwen35" / "qwen35moe") support: a hybrid gated-delta-net (linear-attention) + periodic
// full-attention transformer, dense or MoE. Kept entirely behind the Model seam; ported from the
// reference ../qwen35.java/Qwen35.java. Layers are SSM (gated delta-net) by default; every
// full_attention_interval-th layer is full softmax attention. Single-token forward + batched prefill:
// the delta-net recurrence stays sequential within a chunk, but the per-token projections batch into GEMMs.
package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static com.qxotic.jinfer.Norms.rmsnorm;

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
    static final boolean SINGLE_TOKEN_PREFILL = System.getProperty("jinfer.singleTokenPrefill") != null;

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
        if (SINGLE_TOKEN_PREFILL) {
            for (int i = 0; i < sequenceLength; i++) forward(s, tokens, tokenOffset + i, startPosition + i, 1);
        } else {
            forward(s, tokens, tokenOffset, startPosition, sequenceLength);
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

    // For text-only decoding Qwen3.5's MRoPE reduces to standard interleaved RoPE (the 3D position
    // deltas collapse to pos), so attention uses RoPE.applyInterleaved.

    // === Forward ===

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
        FlashAttention.flashDecode((F32FloatTensor) state.q, (F32FloatTensor) state.xb2,
                keyCache, valueCache, null, null, heads, position, 0, headSize, kvDim, kvMul, attScale, 0, null, state.decodeScratch);

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
        float[] S = state.ssmState[layer];          // raw heap state (no segment accessor)
        float[] sk = state.ssmSk, d = state.ssmD;   // per-head scratch (sized dtRank*headVDim)
        Parallel.parallelFor(0, dtRank, h -> {
            float expGate = (float) Math.exp(gate[h]);
            float betaH = beta[h];
            int stateBase = h * headVDim * headVDim;
            int headOff = h * headVDim;
            for (int idx = stateBase; idx < stateBase + headVDim * headVDim; idx++) S[idx] *= expGate;
            for (int j = 0; j < headVDim; j++) {                 // sk = S k  (contiguous dot)
                float sum = 0; int row = stateBase + j * headVDim;
                for (int i = 0; i < headVDim; i++) sum += S[row + i] * kArr[headOff + i];
                sk[headOff + j] = sum;
            }
            for (int i = 0; i < headVDim; i++) d[headOff + i] = (vArr[headOff + i] - sk[headOff + i]) * betaH;
            for (int j = 0; j < headVDim; j++) {                 // S row j += d[j]*k  (contiguous, autovec)
                float dj = d[headOff + j]; int row = stateBase + j * headVDim;
                for (int i = 0; i < headVDim; i++) S[row + i] += dj * kArr[headOff + i];
            }
            for (int j = 0; j < headVDim; j++) {                 // out = S q  (contiguous dot)
                float sum = 0; int row = stateBase + j * headVDim;
                for (int i = 0; i < headVDim; i++) sum += S[row + i] * qArr[headOff + i];
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
        Ffn.dense(w.ffnGate[layer], w.ffnUp[layer], w.ffnDown[layer], state.xb, state.xb2, state.ffnUp, state.xb,
                1, dim, hiddenDim, Ffn.Act.SILU_GLU);
    }

    /** Insertion-sort top-k of the softmaxed router probs (read at {@code probs[probBase ..]}) into
     *  {@code topE}/{@code topP}, descending and stable so ties keep the lower expert index. Shared by the
     *  decode and batch paths so they stay token-exact. Unfilled slots are left as {@code -1}/-INF. */
    private static void selectTopK(FloatTensor probs, int probBase, int numExperts, int topK,
                                   int[] topE, float[] topP) {
        for (int i = 0; i < topK; i++) { topE[i] = -1; topP[i] = Float.NEGATIVE_INFINITY; }
        for (int e = 0; e < numExperts; e++) {
            float prob = probs.getFloat(probBase + e);
            int insertPos = -1;
            for (int k = 0; k < topK; k++) {
                if (prob > topP[k]) { insertPos = k; break; }
            }
            if (insertPos >= 0) {
                for (int k = topK - 1; k > insertPos; k--) { topP[k] = topP[k - 1]; topE[k] = topE[k - 1]; }
                topP[insertPos] = prob; topE[insertPos] = e;
            }
        }
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
        selectTopK(routerLogits, 0, numExperts, topK, topExperts, topWeights);
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
    /** Single forward pass over {@code seqLen} tokens. The mixer (gated-delta-net SSM or full
     *  attention) and the FFN (dense or MoE) each take their single-token core at {@code seqLen == 1}
     *  (decode) or their batched core for a chunk; the norms + residuals are seqLen-agnostic. */
    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;

        for (int s = 0; s < seqLen; s++) {
            w.tokenEmbeddingTable.copyTo(tokens[tokenOffset + s] * dim, state.x, s * dim, dim);
        }

        for (int l = 0; l < config.numberOfLayers; l++) {
            F32FloatTensor attNormW = w.attnNorm[l], postW = w.postAttentionNorm[l];
            int fDim = dim;
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, s * fDim, state.x, s * fDim, attNormW, fDim, eps));

            if (config.isFullAttention[l]) {
                if (seqLen == 1) attentionForward(state, l, startPos); else attentionForwardBatch(state, l, startPos, seqLen);
            } else {
                if (seqLen == 1) ssmForward(state, l); else ssmForwardBatch(state, l, seqLen);
            }
            // sublayer residual, then post-attention norm acts as the pre-FFN norm
            state.xb.addInPlace(0, state.x, 0, seqLen * dim);
            state.xb.copyTo(0, state.x, 0, seqLen * dim);
            Parallel.forRows(seqLen, s -> rmsnorm(state.xb, s * fDim, state.xb, s * fDim, postW, fDim, eps));

            if (config.isMoE()) {
                if (seqLen == 1) moeForward(state, l); else moeForwardBatch(state, l, seqLen);
            } else {
                if (seqLen == 1) ffnForward(state, l); else ffnForwardBatch(state, l, seqLen);
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

    /** Chunk size for the parallel delta-net scan (intra-chunk dependency -> matmuls; carry across chunks). */
    static final int DELTANET_CHUNK = 64;

    // Vector primitives over the contiguous headVDim dimension (scalar fallback when no Vector API).
    private static float vdot(float[] A, int ao, float[] B, int bo, int n) {
        if (FloatTensor.USE_VECTOR_API) {
            VectorSpecies<Float> sp = FloatTensor.F_SPECIES; int u = sp.length(), i = 0;
            var acc = FloatVector.zero(sp);
            for (; i + u <= n; i += u) acc = FloatVector.fromArray(sp, A, ao+i).fma(FloatVector.fromArray(sp, B, bo+i), acc);
            float s = acc.reduceLanes(VectorOperators.ADD);
            for (; i < n; i++) s += A[ao+i] * B[bo+i];
            return s;
        }
        float s = 0; for (int i = 0; i < n; i++) s += A[ao+i] * B[bo+i]; return s;
    }
    private static void vaxpy(float[] Y, int yo, float s, float[] X, int xo, int n) {   // Y += s*X
        if (FloatTensor.USE_VECTOR_API) {
            VectorSpecies<Float> sp = FloatTensor.F_SPECIES; int u = sp.length(), i = 0;
            var sv = FloatVector.broadcast(sp, s);
            for (; i + u <= n; i += u) FloatVector.fromArray(sp, X, xo+i).fma(sv, FloatVector.fromArray(sp, Y, yo+i)).intoArray(Y, yo+i);
            for (; i < n; i++) Y[yo+i] += s * X[xo+i];
            return;
        }
        for (int i = 0; i < n; i++) Y[yo+i] += s * X[xo+i];
    }
    private static void vscale(float[] Y, int yo, float s, int n) {
        if (FloatTensor.USE_VECTOR_API) {
            VectorSpecies<Float> sp = FloatTensor.F_SPECIES; int u = sp.length(), i = 0;
            var sv = FloatVector.broadcast(sp, s);
            for (; i + u <= n; i += u) FloatVector.fromArray(sp, Y, yo+i).mul(sv).intoArray(Y, yo+i);
            for (; i < n; i++) Y[yo+i] *= s;
            return;
        }
        for (int i = 0; i < n; i++) Y[yo+i] *= s;
    }

    /** Blocked transpose: Bt[btB + k*J + j] = B[bB + j*ldb + k], for j in [0,J), k in [0,K). */
    private static void transpose(float[] B, int bB, int ldb, int J, int K, float[] Bt, int btB) {
        final int BL = 16;
        for (int j0 = 0; j0 < J; j0 += BL) for (int k0 = 0; k0 < K; k0 += BL) {
            int jE = Math.min(j0+BL, J), kE = Math.min(k0+BL, K);
            for (int j = j0; j < jE; j++) for (int k = k0; k < kE; k++) Bt[btB + k*J + j] = B[bB + j*ldb + k];
        }
    }

    /** Broadcast/outer-product GEMM C[i,j] = sum_k A[aB+i*lda+k]*Bt[btB+k*ldbt+j]; 4x4 register tile, no reduces. */
    private static void abMul(float[] A, int aB, int lda, float[] Bt, int btB, int ldbt, float[] C, int cB, int ldc, int I, int J, int K) {
        if (!FloatTensor.USE_VECTOR_API) {
            for (int i=0;i<I;i++) for (int j=0;j<J;j++){ float s=0; for(int k=0;k<K;k++) s+=A[aB+i*lda+k]*Bt[btB+k*ldbt+j]; C[cB+i*ldc+j]=s; }
            return;
        }
        VectorSpecies<Float> sp = FloatTensor.F_SPECIES; int U = sp.length(), JV = 4*U, i = 0;
        for (; i + 4 <= I; i += 4) {
            int a0=aB+i*lda,a1=a0+lda,a2=a1+lda,a3=a2+lda, r0=cB+i*ldc,r1=r0+ldc,r2=r1+ldc,r3=r2+ldc, j = 0;
            for (; j + JV <= J; j += JV) {
                var x00=FloatVector.zero(sp);var x01=x00;var x02=x00;var x03=x00;var x10=x00;var x11=x00;var x12=x00;var x13=x00;
                var x20=x00;var x21=x00;var x22=x00;var x23=x00;var x30=x00;var x31=x00;var x32=x00;var x33=x00;
                for (int k = 0; k < K; k++) {
                    int bk = btB + k*ldbt + j;
                    var b0=FloatVector.fromArray(sp,Bt,bk);var b1=FloatVector.fromArray(sp,Bt,bk+U);var b2=FloatVector.fromArray(sp,Bt,bk+2*U);var b3=FloatVector.fromArray(sp,Bt,bk+3*U);
                    var v=FloatVector.broadcast(sp,A[a0+k]);x00=v.fma(b0,x00);x01=v.fma(b1,x01);x02=v.fma(b2,x02);x03=v.fma(b3,x03);
                    v=FloatVector.broadcast(sp,A[a1+k]);x10=v.fma(b0,x10);x11=v.fma(b1,x11);x12=v.fma(b2,x12);x13=v.fma(b3,x13);
                    v=FloatVector.broadcast(sp,A[a2+k]);x20=v.fma(b0,x20);x21=v.fma(b1,x21);x22=v.fma(b2,x22);x23=v.fma(b3,x23);
                    v=FloatVector.broadcast(sp,A[a3+k]);x30=v.fma(b0,x30);x31=v.fma(b1,x31);x32=v.fma(b2,x32);x33=v.fma(b3,x33);
                }
                x00.intoArray(C,r0+j);x01.intoArray(C,r0+j+U);x02.intoArray(C,r0+j+2*U);x03.intoArray(C,r0+j+3*U);
                x10.intoArray(C,r1+j);x11.intoArray(C,r1+j+U);x12.intoArray(C,r1+j+2*U);x13.intoArray(C,r1+j+3*U);
                x20.intoArray(C,r2+j);x21.intoArray(C,r2+j+U);x22.intoArray(C,r2+j+2*U);x23.intoArray(C,r2+j+3*U);
                x30.intoArray(C,r3+j);x31.intoArray(C,r3+j+U);x32.intoArray(C,r3+j+2*U);x33.intoArray(C,r3+j+3*U);
            }
            for (; j < J; j++) for (int ii=0;ii<4;ii++){ float s=0; for(int k=0;k<K;k++) s+=A[aB+(i+ii)*lda+k]*Bt[btB+k*ldbt+j]; C[cB+(i+ii)*ldc+j]=s; }
        }
        for (; i < I; i++) for (int j=0;j<J;j++){ float s=0; for(int k=0;k<K;k++) s+=A[aB+i*lda+k]*Bt[btB+k*ldbt+j]; C[cB+i*ldc+j]=s; }
    }

    /** Contract-over-rows GEMM C[j,i] = sum_t U[uB+t*d+j]*K[kB+t*kld+i] (vectorize i, broadcast U[t,j]); 4x4 tile. */
    private static void utk(float[] U, int uB, int n, int d, float[] K, int kB, int kld, float[] C, int cB, int cld) {
        if (!FloatTensor.USE_VECTOR_API) {
            for (int j=0;j<d;j++) for (int i=0;i<d;i++){ float s=0; for(int t=0;t<n;t++) s+=U[uB+t*d+j]*K[kB+t*kld+i]; C[cB+j*cld+i]=s; }
            return;
        }
        VectorSpecies<Float> sp = FloatTensor.F_SPECIES; int U_=sp.length(), IV=4*U_, j = 0;
        for (; j + 4 <= d; j += 4) {
            int i = 0;
            for (; i + IV <= d; i += IV) {
                var c00=FloatVector.zero(sp);var c01=c00;var c02=c00;var c03=c00;var c10=c00;var c11=c00;var c12=c00;var c13=c00;
                var c20=c00;var c21=c00;var c22=c00;var c23=c00;var c30=c00;var c31=c00;var c32=c00;var c33=c00;
                for (int t = 0; t < n; t++) {
                    int kt=kB+t*kld+i, ut=uB+t*d+j;
                    var k0=FloatVector.fromArray(sp,K,kt);var k1=FloatVector.fromArray(sp,K,kt+U_);var k2=FloatVector.fromArray(sp,K,kt+2*U_);var k3=FloatVector.fromArray(sp,K,kt+3*U_);
                    var u=FloatVector.broadcast(sp,U[ut]);c00=u.fma(k0,c00);c01=u.fma(k1,c01);c02=u.fma(k2,c02);c03=u.fma(k3,c03);
                    u=FloatVector.broadcast(sp,U[ut+1]);c10=u.fma(k0,c10);c11=u.fma(k1,c11);c12=u.fma(k2,c12);c13=u.fma(k3,c13);
                    u=FloatVector.broadcast(sp,U[ut+2]);c20=u.fma(k0,c20);c21=u.fma(k1,c21);c22=u.fma(k2,c22);c23=u.fma(k3,c23);
                    u=FloatVector.broadcast(sp,U[ut+3]);c30=u.fma(k0,c30);c31=u.fma(k1,c31);c32=u.fma(k2,c32);c33=u.fma(k3,c33);
                }
                int r0=cB+j*cld+i,r1=r0+cld,r2=r1+cld,r3=r2+cld;
                c00.intoArray(C,r0);c01.intoArray(C,r0+U_);c02.intoArray(C,r0+2*U_);c03.intoArray(C,r0+3*U_);
                c10.intoArray(C,r1);c11.intoArray(C,r1+U_);c12.intoArray(C,r1+2*U_);c13.intoArray(C,r1+3*U_);
                c20.intoArray(C,r2);c21.intoArray(C,r2+U_);c22.intoArray(C,r2+2*U_);c23.intoArray(C,r2+3*U_);
                c30.intoArray(C,r3);c31.intoArray(C,r3+U_);c32.intoArray(C,r3+2*U_);c33.intoArray(C,r3+3*U_);
            }
            for (; i < d; i++) for (int jj=0;jj<4;jj++){ float s=0; for(int t=0;t<n;t++) s+=U[uB+t*d+j+jj]*K[kB+t*kld+i]; C[cB+(j+jj)*cld+i]=s; }
        }
    }

    /** Batched gated delta-net: GEMM projections + batched causal conv, then the CHUNKED parallel scan
     *  (intra-chunk recurrence collapsed into matmuls + a triangular solve; carry across chunks). Parallel
     *  over heads. Token-exact vs the sequential {@link #ssmForward}; math validated in bench/DeltaNetParity. */
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

        // 7. delta-net recurrence — Lever 3: CHUNKED gated DeltaNet. Within a chunk the per-row sequential
        // dependency collapses into matmuls + a triangular solve over the chunk; the carry state S flows
        // across chunks. This reuses the cache-resident S0 within a chunk (vs streaming the evolving state
        // every row), turning the bandwidth-bound scan into compute-dense, vectorized matmuls. Parallel over
        // heads (state per head independent). NUMERICALLY-STABLE factoring (matches llama.cpp): substitute
        // W_t = gamma_t U_t so the unstable V/gamma never appears (1/gamma overflows when gamma underflows,
        // which large gates from special tokens trigger -> NaN). All decays are gamma_t = prod_{s<=t} a (<=1)
        // or ratios gamma_t/gamma_r = prod_{s=r+1..t} a (<=1, r<=t), built as direct running products:
        //   W = beta(V - gamma KS0); solve W_t -= beta(k_t.k_r)(gamma_t/gamma_r)W_r;
        //   O = gamma QS0 + sum_{r<=t}(q_t.k_r)(gamma_t/gamma_r)W_r; S = gamma_L S0 + sum_t (gamma_L/gamma_t)W_t k_t^T.
        // Tiled: KS0^T, QS0^T via abMul(+S0T transpose); U^T K via utk; n^2 parts vdot/vaxpy. Validated float-exact.
        float[] output = state.ssmOutput;
        float[] S = state.ssmState[layer];
        float[] chunkM = state.ssmChunkM, chunkG = state.ssmChunkG, decay = state.ssmDecay,
                qs0 = state.ssmQS0, s0t = state.ssmS0T, utkC = state.ssmUtK;
        final int HV = fHV, NH = fDtRank, CH = DELTANET_CHUNK, LDA = NH * HV;
        Parallel.parallelFor(0, NH, h -> {
            int sBase = h*HV*HV, mBase = h*CH*HV, gBase = h*CH, dBase = h*CH*CH, qBase = h*CH*HV, tBase = h*HV*HV;
            for (int c0 = 0; c0 < seqLen; c0 += CH) {
                int n = Math.min(CH, seqLen - c0);
                int row0 = (c0*NH + h) * HV;                                  // chunk row 0 for head h (kArr/qArr/output)
                float acc = 1f;                                              // gamma_t = prod_{s<=t} a (<=1) and
                for (int t = 0; t < n; t++) {                                 // D[t,r] = prod_{s=r+1..t} a = a_t D[t-1,r]
                    float at = (float) Math.exp(gate[(c0+t)*NH + h]);
                    acc *= at; chunkG[gBase + t] = acc;
                    int dt = dBase + t*CH;
                    if (t > 0) { System.arraycopy(decay, dBase + (t-1)*CH, decay, dt, t); vscale(decay, dt, at, t); }
                    decay[dt + t] = 1f;
                }
                transpose(S, sBase, HV, HV, HV, s0t, tBase);                  // s0t[i,j] = S0[j,i]
                abMul(kArr, row0, LDA, s0t, tBase, HV, chunkM, mBase, HV, n, HV, HV);   // KS0[t,j] -> chunkM
                for (int t = 0; t < n; t++) {                                 // W = beta (V - gamma KS0)
                    int vRow = ((c0+t)*NH + h) * HV, mt = mBase + t*HV; float bt = beta[(c0+t)*NH + h], gt = chunkG[gBase + t];
                    for (int j = 0; j < HV; j++) chunkM[mt + j] = bt * (vArr[vRow + j] - gt*chunkM[mt + j]);
                }
                for (int t = 0; t < n; t++) {                                 // solve (decayed) in place
                    int kRow = ((c0+t)*NH + h) * HV, mt = mBase + t*HV, dt = dBase + t*CH; float bt = beta[(c0+t)*NH + h];
                    for (int r = 0; r < t; r++)
                        vaxpy(chunkM, mt, -bt * vdot(kArr, kRow, kArr, ((c0+r)*NH + h)*HV, HV) * decay[dt + r], chunkM, mBase + r*HV, HV);
                }
                abMul(qArr, row0, LDA, s0t, tBase, HV, qs0, qBase, HV, n, HV, HV);      // QS0[t,j] -> qs0
                for (int t = 0; t < n; t++) {                                 // O = gamma QS0 + tril decayed
                    int qRow = ((c0+t)*NH + h) * HV, oRow = qRow, qt = qBase + t*HV, dt = dBase + t*CH; float gt = chunkG[gBase + t];
                    for (int j = 0; j < HV; j++) output[oRow + j] = gt * qs0[qt + j];
                    for (int r = 0; r <= t; r++)
                        vaxpy(output, oRow, vdot(qArr, qRow, kArr, ((c0+r)*NH + h)*HV, HV) * decay[dt + r], chunkM, mBase + r*HV, HV);
                }
                int dEnd = dBase + (n-1)*CH;                                  // W_t -> (gamma_L/gamma_t) W_t
                for (int t = 0; t < n; t++) vscale(chunkM, mBase + t*HV, decay[dEnd + t], HV);
                utk(chunkM, mBase, n, HV, kArr, row0, LDA, utkC, tBase, HV);   // sum_t Wd[t,j] k_t[i] -> utkC[j,i]
                float gL = chunkG[gBase + n - 1];                             // S_L = gamma_L S0 + Wd^T K
                for (int j = 0; j < HV; j++) { int sr = sBase + j*HV, ur = tBase + j*HV;
                    for (int i = 0; i < HV; i++) S[sr + i] = gL * S[sr + i] + utkC[ur + i]; }
            }
        });

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
        Ffn.dense(w.ffnGate[layer], w.ffnUp[layer], w.ffnDown[layer], state.xb, state.ffnGate, state.ffnUp, state.xb,
                seqLen, dim, hiddenDim, Ffn.Act.SILU_GLU);
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
            selectTopK(state.moeRouterB, s * numExperts, numExperts, topK, topE, topP);
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
        // CSR grouping + gather + scatter is shared; the per-expert math (gated SiLU) is here.
        Moe.Routing r = state.moeRouting;
        r.seqLen = seqLen; r.topK = topK; r.numExperts = numExperts;
        Moe.dispatch(r, dim, state.moeInputB, state.moeGather, state.moeDownB, state.moeOutB, null,
                (e, n, gather, out) -> {
                    int gateUpOffset = e * gateUpStride;
                    w.moeExpertGate[layer].gemm(gather, dim, state.moeGateUpB, expertFFN, n, expertFFN, dim, gateUpOffset);
                    w.moeExpertUp[layer].gemm(gather, dim, state.moeUpB, expertFFN, n, expertFFN, dim, gateUpOffset);
                    Parallel.forRows(n, j -> state.moeGateUpB.siluMultiplyInPlace(j * expertFFN, state.moeUpB, j * expertFFN, expertFFN));
                    w.moeExpertDown[layer].gemm(state.moeGateUpB, expertFFN, out, dim, n, dim, expertFFN, e * downStride);
                });

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
                    state.moeOutB.saxpyInPlace(s * dim, state.moeSharedOutB, s * dim, dim, sharedScale);
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
        final FloatTensor x, xb, xb2, q, k, v, logits, ffnUp, ffnGate, ssmQkv, ssmTmp;
        final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch();
        final float[] attnGateArr, ssmZ, ssmConvOut, ssmQ, ssmK, ssmV, ssmQGroup, ssmKGroup, ssmGate, ssmBeta, ssmOutput, ssmSk, ssmD;
        final float[] ssmChunkM, ssmChunkG, ssmDecay, ssmQS0, ssmS0T, ssmUtK;   // per-head chunked-scan scratch (sliced by head)
        // Batched-attention scratch (chunk rows): queries deinterleaved from q, attention output.
        final FloatTensor attnQ, attnOut;
        final FloatTensor[] keyCache, valueCache, ssmConvState;
        final float[][] ssmState;   // delta-net state, raw heap array (hot scan; no segment accessor)
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
        final Moe.Routing moeRouting;
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
            this.ssmChunkM = new float[dtRank * DELTANET_CHUNK * headVDim];
            this.ssmChunkG = new float[dtRank * DELTANET_CHUNK];
            this.ssmDecay = new float[dtRank * DELTANET_CHUNK * DELTANET_CHUNK];   // per-head CxC decay ratios
            this.ssmQS0 = new float[dtRank * DELTANET_CHUNK * headVDim];
            this.ssmS0T = new float[dtRank * headVDim * headVDim];
            this.ssmUtK = new float[dtRank * headVDim * headVDim];

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
                this.moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeProbByExpert);
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
                this.moeRouting = null;
            }

            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            this.ssmConvState = new FloatTensor[config.numberOfLayers];
            this.ssmState = new float[config.numberOfLayers][];
            for (int l = 0; l < config.numberOfLayers; l++) {
                if (config.isFullAttention[l]) {
                    keyCache[l] = F16FloatTensor.allocate(config.contextLength * kvDim);
                    valueCache[l] = F16FloatTensor.allocate(config.contextLength * kvDim);
                } else {
                    ssmConvState[l] = F32FloatTensor.allocate((config.ssmConvKernel - 1) * convChannels);
                    ssmState[l] = new float[headVDim * headVDim * dtRank];
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
        LFMTokenizer tokenizer = new LFMTokenizer(gguf, JinjaRenderer::template);
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
