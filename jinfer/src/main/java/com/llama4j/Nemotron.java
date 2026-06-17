// Nemotron Cascade 2 ("nemotron_h_moe") support: a hybrid Mamba2-SSM + periodic full-attention +
// MoE transformer. Each block is exactly ONE of {SSM, attention, MoE-FFN}, chosen per-layer from the
// GGUF head_count_kv / feed_forward_length arrays, behind a single pre-norm. Kept entirely behind the
// Model seam; ported from ../nemotron3.java/Nemotron3.java and cross-checked against llama.cpp's
// nemotron-h graph. Single-token forward + batched prefill: the Mamba2 recurrence stays sequential
// within a chunk, but the per-token projections batch into GEMMs.
//
// Notable architecture specifics (differ from a Qwen-style MoE): NO RoPE and NO q/k norm in
// attention; MoE router is SIGMOID with an additive selection bias (ffn exp_probs_b), combine weights
// taken from the UNBIASED sigmoid, then normalized + scaled; all FFN/expert activations are
// squared-ReLU (no gate projection); a shared expert is always added.
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

final class Nemotron implements Model {

    private final Configuration configuration;
    private final LFMTokenizer tokenizer;
    private final Weights weights;

    Nemotron(Configuration configuration, LFMTokenizer tokenizer, Weights weights) {
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
    static final boolean SINGLE_TOKEN_PREFILL = System.getProperty("nemotron.singleTokenPrefill") != null;

    /** The Mamba2 selective-scan recurrence is sequential, but the per-token projections (in-proj,
     *  conv, Q/K/V, router, experts) batch into GEMMs; only the scan stays sequential within a chunk. */
    @Override
    public int batchCapacity() {
        return SINGLE_TOKEN_PREFILL ? 1 : Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
    }

    @Override
    public State createNewState() {
        State state = new State(configuration);
        // add_bos_token is false: -1 = "no prior token", so the engine's prefill feeds the rendered
        // prompt verbatim without prepending anything (no spurious leading token).
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
        if (configuration.eosTokenId >= 0) stops.add(configuration.eosTokenId);
        for (String name : new String[]{"<|im_end|>", "<|endoftext|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        return stops;
    }

    // === Math helpers ===

    // === Forward (single token) ===

    private void forward(State state, int token, int position) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;

        w.tokenEmbeddingTable.copyTo(token * dim, state.x, 0, dim);

        for (int l = 0; l < config.numberOfLayers; l++) {
            // Single pre-norm per block (attn_norm) feeds whichever mixer this block is.
            rmsnorm(state.xb, 0, state.x, 0, w.attnNorm[l], dim, eps);
            switch (config.layerTypes[l]) {
                case SSM -> ssmForward(state, l);
                case ATTENTION -> attentionForward(state, l, position);
                case MOE -> moeForward(state, l);
            }
            state.x.addInPlace(0, state.xb, 0, dim);   // residual; mixer left its output in xb
        }
        state.lastRowOffset = 0;
    }

    /** Mamba2 SSM mixer: in-proj -> split (z|xBC|dt) -> depthwise causal conv1d+SiLU on xBC ->
     *  selective scan with per-head dt/A and per-group B/C -> D skip -> SiLU(z) gate -> per-group
     *  gated RMSNorm -> out-proj. Reads {@code state.xb} (normed input), writes the result to it. */
    private void ssmForward(State state, int l) {
        Configuration c = configuration;
        Weights w = weights;
        int dim = c.embeddingLength;
        int dInner = c.ssmInnerSize;
        int dState = c.ssmStateSize;
        int nHead = c.ssmTimeStepRank;       // Mamba2 head count is overloaded onto time_step_rank
        int nGroup = c.ssmGroupCount;
        int dConv = c.ssmConvKernel;
        int headDim = dInner / nHead;
        int convCh = c.ssmConvChannels();
        int qSize = nGroup * dState;
        float eps = c.rmsNormEps;

        // 1. in-projection, split into z (gate) | xBC (conv input: x|B|C) | dt (per ssm head)
        w.ssmIn[l].matmul(state.xb, state.ssmInProj, c.ssmInProjSize(), dim);
        float[] z = state.ssmZ, xbc = state.ssmXbc, convOut = state.ssmConvOut, y = state.ssmY, dt = state.ssmDt;
        for (int i = 0; i < dInner; i++) z[i] = state.ssmInProj.getFloat(i);
        for (int i = 0; i < convCh; i++) xbc[i] = state.ssmInProj.getFloat(dInner + i);
        int dtOff = 2 * dInner + 2 * nGroup * dState;
        for (int h = 0; h < nHead; h++) dt[h] = Activations.softplus(state.ssmInProj.getFloat(dtOff + h) + w.ssmDtB[l].getFloat(h));

        // 2. depthwise causal conv1d over the dConv-wide window (state ring + current), bias, SiLU
        FloatTensor convState = state.ssmConvState[l];
        F32FloatTensor convW = w.ssmConv1d[l], convB = w.ssmConv1dB[l];
        for (int ch = 0; ch < convCh; ch++) {
            float sum = convB == null ? 0f : convB.getFloat(ch);
            int wOff = ch * dConv;
            for (int k = 0; k < dConv - 1; k++) {
                sum += convW.getFloat(wOff + k) * convState.getFloat(k * convCh + ch);
            }
            sum += convW.getFloat(wOff + dConv - 1) * xbc[ch];
            convOut[ch] = Activations.silu(sum);
        }
        // advance the conv ring (drop oldest, append current xBC)
        for (int k = 0; k < dConv - 2; k++) {
            for (int ch = 0; ch < convCh; ch++) {
                convState.setFloat(k * convCh + ch, convState.getFloat((k + 1) * convCh + ch));
            }
        }
        for (int ch = 0; ch < convCh; ch++) convState.setFloat((dConv - 2) * convCh + ch, xbc[ch]);

        // 3. selective scan: per head h, state h_new = h*exp(dt*A) + B*(x*dt); y = C.h + D*x; gate SiLU(z)
        FloatTensor ssmState = state.ssmState[l];
        for (int h = 0; h < nHead; h++) {
            int g = h / (nHead / nGroup);
            float dA = (float) Math.exp(dt[h] * w.ssmA[l].getFloat(h));
            float dScale = w.ssmD[l].getFloat(h);
            for (int ii = 0; ii < headDim; ii++) {
                int idx = h * headDim + ii;
                float xdt = convOut[idx] * dt[h];
                float sum = 0f;
                for (int i0 = 0; i0 < dState; i0++) {
                    int st = i0 + idx * dState;
                    int ig = i0 + g * dState;
                    float next = ssmState.getFloat(st) * dA + convOut[dInner + ig] * xdt;
                    ssmState.setFloat(st, next);
                    sum += next * convOut[dInner + qSize + ig];
                }
                y[idx] = (sum + convOut[idx] * dScale) * Activations.silu(z[idx]);
            }
        }

        // 4. per-group gated RMSNorm (gate already folded into y above) then out-projection
        int groupDim = dInner / nGroup;
        F32FloatTensor normW = w.ssmNorm[l];
        for (int g = 0; g < nGroup; g++) {
            int off = g * groupDim;
            float ss = 0f;
            for (int i = 0; i < groupDim; i++) ss += y[off + i] * y[off + i];
            float inv = (float) (1.0 / Math.sqrt(ss / groupDim + eps));
            for (int i = 0; i < groupDim; i++) {
                state.ssmTmp.setFloat(off + i, y[off + i] * inv * normW.getFloat(off + i));
            }
        }
        w.ssmOut[l].matmul(state.ssmTmp, state.xb, dim, dInner);
    }

    /** Full causal softmax attention, GQA. No RoPE, no q/k norm, no biases. Reads {@code state.xb}
     *  (normed input), writes the output projection back to it. */
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

        w.attnQ[layer].matmul(state.xb, state.q, queryDim, dim);
        w.attnK[layer].matmul(state.xb, state.k, kvDim, dim);
        w.attnV[layer].matmul(state.xb, state.v, kvDim, dim);
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

        w.attnOutput[layer].matmul(state.xb2, state.xb, dim, queryDim);
    }

    /** MoE-FFN: sigmoid router with additive selection bias, top-k by biased score but combined with
     *  the unbiased sigmoid weight (optionally normalized then scaled); squared-ReLU experts plus an
     *  always-on shared expert. Reads {@code state.xb} (normed input), writes the result back to it. */
    private void moeForward(State state, int layer) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int expertFF = config.expertFeedForwardLength;
        int numExperts = config.expertCount;
        int topK = Math.min(config.expertUsedCount, numExperts);

        // Router: per-expert sigmoid(logit). Selection uses sigmoid + exp_probs_b bias; the kept
        // combine weight is the UNBIASED sigmoid.
        w.ffnGateInp[layer].matmul(state.xb, state.moeRouter, numExperts, dim);
        for (int e = 0; e < numExperts; e++) state.moeRouter.setFloat(e, Activations.sigmoid(state.moeRouter.getFloat(e)));

        int[] topExperts = state.moeTopExperts;
        float[] topWeights = state.moeTopWeights;
        F32FloatTensor probsB = w.expProbsB[layer];
        for (int k = 0; k < topK; k++) {
            int best = -1;
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int e = 0; e < numExperts; e++) {
                float score = state.moeRouter.getFloat(e) + (probsB == null ? 0f : probsB.getFloat(e));
                boolean taken = false;
                for (int j = 0; j < k; j++) if (topExperts[j] == e) { taken = true; break; }
                if (!taken && score > bestScore) { bestScore = score; best = e; }
            }
            topExperts[k] = best;
            topWeights[k] = state.moeRouter.getFloat(best);   // unbiased combine weight
        }

        float weightSum = 0f;
        if (config.expertWeightsNorm) {
            for (int k = 0; k < topK; k++) weightSum += topWeights[k];
            weightSum = Math.max(weightSum, 6.103515625e-5f);   // 2^-14 floor
        }
        for (int k = 0; k < topK; k++) {
            float coeff = topWeights[k];
            if (config.expertWeightsNorm) coeff /= weightSum;
            coeff *= config.expertWeightsScale;
            topWeights[k] = coeff;
        }

        state.moeAccum.fillInPlace(0, dim, 0f);
        for (int k = 0; k < topK; k++) {
            int e = topExperts[k];
            int upOffset = e * expertFF * dim;
            int downOffset = e * dim * expertFF;
            w.ffnUpExps[layer].matmul(state.xb, state.moeHidden, expertFF, dim, upOffset);
            state.moeHidden.reluSqrInPlace(0, expertFF);
            w.ffnDownExps[layer].matmul(state.moeHidden, state.moeExpertOut, dim, expertFF, downOffset);
            state.moeAccum.saxpyInPlace(0, state.moeExpertOut, 0, dim, topWeights[k]);
        }

        // Shared expert (always active), squared-ReLU, coefficient 1.
        if (w.ffnUpShexp[layer] != null) {
            int sff = config.expertSharedFeedForwardLength;
            w.ffnUpShexp[layer].matmul(state.xb, state.moeSharedHidden, sff, dim);
            state.moeSharedHidden.reluSqrInPlace(0, sff);
            w.ffnDownShexp[layer].matmul(state.moeSharedHidden, state.moeExpertOut, dim, sff);
            state.moeAccum.addInPlace(0, state.moeExpertOut, 0, dim);
        }

        state.moeAccum.copyTo(0, state.xb, 0, dim);
    }

    // === Batched forward (prompt processing): seqLen tokens in one pass ===

    /** Processes {@code seqLen} tokens at positions {@code [startPos, startPos+seqLen)} in one pass:
     *  the per-token projections (in-proj, conv, Q/K/V, router, experts) become GEMMs; the Mamba2 scan
     *  stays sequential within the chunk (state carries forward). Single pre-norm per block dispatches
     *  to the batched mixer. Leaves the post-final-layer residual in {@code x}; {@link #computeLogits}
     *  finalizes the last row. Token-exact (greedy) vs the single-token {@link #forward}. */
    void forwardBatch(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
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
            switch (config.layerTypes[l]) {
                case SSM -> ssmForwardBatch(state, l, seqLen);
                case ATTENTION -> attentionForwardBatch(state, l, startPos, seqLen);
                case MOE -> moeForwardBatch(state, l, seqLen);
            }
            state.x.addInPlace(0, state.xb, 0, seqLen * dim);
        }
        state.lastRowOffset = (seqLen - 1) * dim;
    }

    /** Batched attention: Q/K/V GEMMs, K/V written to the contiguous cache, causal flash attention,
     *  output projection. No RoPE / q-k norm / biases (mirrors {@link #attentionForward}). */
    private void attentionForwardBatch(State state, int layer, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int headSize = config.headSize;
        int kvDim = config.kvDim();
        int queryDim = config.queryDim();
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads;

        w.attnQ[layer].gemm(state.xb, dim, state.attnQ, queryDim, seqLen, queryDim, dim);
        w.attnK[layer].gemm(state.xb, dim, state.k, kvDim, seqLen, kvDim, dim);
        w.attnV[layer].gemm(state.xb, dim, state.v, kvDim, seqLen, kvDim, dim);

        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        for (int s = 0; s < seqLen; s++) {
            state.k.copyTo(s * kvDim, keyCache, (startPos + s) * kvDim, kvDim);
            state.v.copyTo(s * kvDim, valueCache, (startPos + s) * kvDim, kvDim);
        }

        FlashAttention.causalPrefill((F32FloatTensor) state.attnQ, (F32FloatTensor) state.attnOut,
                keyCache, valueCache, configuration.numberOfHeads, startPos, seqLen, headSize, kvDim, queryDim, kvMul);
        w.attnOutput[layer].gemm(state.attnOut, queryDim, state.xb, dim, seqLen, dim, queryDim);
    }

    /** Batched Mamba2 SSM: in-proj GEMM, batched causal conv1d (parallel over channels, reading the
     *  conv ring for the chunk prefix), then the selective scan SEQUENTIAL over rows (state carries
     *  forward) parallel over heads, per-row gated RMSNorm, out-proj GEMM. Token-exact vs the
     *  single-token {@link #ssmForward}. */
    private void ssmForwardBatch(State state, int l, int seqLen) {
        Configuration c = configuration;
        Weights w = weights;
        int dim = c.embeddingLength;
        int dInner = c.ssmInnerSize;
        int dState = c.ssmStateSize;
        int nHead = c.ssmTimeStepRank;
        int nGroup = c.ssmGroupCount;
        int dConv = c.ssmConvKernel;
        int headDim = dInner / nHead;
        int convCh = c.ssmConvChannels();
        int qSize = nGroup * dState;
        int inProjSize = c.ssmInProjSize();
        float eps = c.rmsNormEps;

        // 1. in-projection GEMM, then split each row into z | xBC | dt.
        w.ssmIn[l].gemm(state.xb, dim, state.ssmInProj, inProjSize, seqLen, inProjSize, dim);
        float[] z = state.ssmZ, xbc = state.ssmXbc, convOut = state.ssmConvOut, y = state.ssmY, dt = state.ssmDt;
        int dtOff = 2 * dInner + 2 * nGroup * dState;
        int fInProj = inProjSize, fDInner = dInner, fConvCh = convCh, fNHead = nHead;
        F32FloatTensor dtB = w.ssmDtB[l];
        Parallel.forRows(seqLen, s -> {
            int p = s * fInProj;
            for (int i = 0; i < fDInner; i++) z[s * fDInner + i] = state.ssmInProj.getFloat(p + i);
            for (int i = 0; i < fConvCh; i++) xbc[s * fConvCh + i] = state.ssmInProj.getFloat(p + fDInner + i);
            for (int h = 0; h < fNHead; h++) dt[s * fNHead + h] = Activations.softplus(state.ssmInProj.getFloat(p + dtOff + h) + dtB.getFloat(h));
        });

        // 2. batched depthwise causal conv1d + SiLU; row s, channel ch reads taps s-(dConv-1)+k from
        // the chunk or, for negative positions, the conv ring (last dConv-1 inputs of the prior chunk).
        FloatTensor convState = state.ssmConvState[l];
        F32FloatTensor convW = w.ssmConv1d[l], convBias = w.ssmConv1dB[l];
        int fK = dConv, fHist = dConv - 1;
        Parallel.parallelFor(0, convCh, ch -> {
            int wOff = ch * fK;
            for (int s = 0; s < seqLen; s++) {
                float sum = convBias == null ? 0f : convBias.getFloat(ch);
                for (int k = 0; k < fK; k++) {
                    int pos = s - fHist + k;
                    float in = pos < 0 ? convState.getFloat((pos + fHist) * fConvCh + ch) : xbc[pos * fConvCh + ch];
                    sum += convW.getFloat(wOff + k) * in;
                }
                convOut[s * fConvCh + ch] = Activations.silu(sum);
            }
        });
        // roll the conv ring: keep the last dConv-1 xBC rows of this chunk.
        Parallel.parallelFor(0, convCh, ch -> {
            for (int k = 0; k < fHist; k++) {
                int pos = seqLen - fHist + k;
                float v = pos < 0 ? convState.getFloat((pos + fHist) * fConvCh + ch) : xbc[pos * fConvCh + ch];
                convState.setFloat(k * fConvCh + ch, v);
            }
        });

        // 3. selective scan: SEQUENTIAL over rows (state carries forward), parallel over heads.
        FloatTensor ssmState = state.ssmState[l];
        int fHeadDim = headDim, fDState = dState, fNGroup = nGroup, fQSize = qSize;
        F32FloatTensor ssmA = w.ssmA[l], ssmD = w.ssmD[l];
        for (int s = 0; s < seqLen; s++) {
            int cBase = s * fConvCh, base = s * fDInner, dtBase = s * fNHead;
            Parallel.parallelFor(0, fNHead, h -> {
                int g = h / (fNHead / fNGroup);
                float dtH = dt[dtBase + h];
                float dA = (float) Math.exp(dtH * ssmA.getFloat(h));
                float dScale = ssmD.getFloat(h);
                for (int ii = 0; ii < fHeadDim; ii++) {
                    int idx = h * fHeadDim + ii;
                    float xdt = convOut[cBase + idx] * dtH;
                    float sum = 0f;
                    for (int i0 = 0; i0 < fDState; i0++) {
                        int st = i0 + idx * fDState;
                        int ig = i0 + g * fDState;
                        float next = ssmState.getFloat(st) * dA + convOut[cBase + fDInner + ig] * xdt;
                        ssmState.setFloat(st, next);
                        sum += next * convOut[cBase + fDInner + fQSize + ig];
                    }
                    y[base + idx] = (sum + convOut[cBase + idx] * dScale) * Activations.silu(z[base + idx]);
                }
            });
        }

        // 4. per-group gated RMSNorm (gate already folded into y) then out-projection GEMM.
        int groupDim = dInner / nGroup;
        F32FloatTensor normW = w.ssmNorm[l];
        int fGroupDim = groupDim;
        Parallel.forRows(seqLen, s -> {
            for (int g = 0; g < fNGroup; g++) {
                int off = s * fDInner + g * fGroupDim;
                float ss = 0f;
                for (int i = 0; i < fGroupDim; i++) ss += y[off + i] * y[off + i];
                float inv = (float) (1.0 / Math.sqrt(ss / fGroupDim + eps));
                for (int i = 0; i < fGroupDim; i++) {
                    state.ssmTmp.setFloat(off + i, y[off + i] * inv * normW.getFloat(g * fGroupDim + i));
                }
            }
        });
        w.ssmOut[l].gemm(state.ssmTmp, dInner, state.xb, dim, seqLen, dim, dInner);
    }

    /** Batched MoE: router GEMM + per-row sigmoid/top-k/normalize, CSR gather-by-expert squared-ReLU
     *  GEMMs, plus the batched shared expert. Sums routed (weighted) + shared into {@code xb}.
     *  Token-exact vs the single-token {@link #moeForward}. */
    private void moeForwardBatch(State state, int layer, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int expertFF = config.expertFeedForwardLength;
        int numExperts = config.expertCount;
        int topK = Math.min(config.expertUsedCount, numExperts);

        // Snapshot the normed input (CSR gather scrambles row order; router + shared read it too).
        state.xb.copyTo(0, state.moeInputB, 0, seqLen * dim);

        w.ffnGateInp[layer].gemm(state.moeInputB, dim, state.moeRouterB, numExperts, seqLen, numExperts, dim);
        int[] counts = state.moeExpertCounts;
        java.util.Arrays.fill(counts, 0);
        F32FloatTensor probsB = w.expProbsB[layer];
        int[] topE = state.moeTopExperts;
        float[] topP = state.moeTopWeights;
        for (int s = 0; s < seqLen; s++) {
            int rb = s * numExperts;
            for (int e = 0; e < numExperts; e++) state.moeRouterB.setFloat(rb + e, Activations.sigmoid(state.moeRouterB.getFloat(rb + e)));
            // top-k by (sigmoid + bias); kept combine weight is the UNBIASED sigmoid.
            for (int k = 0; k < topK; k++) {
                int best = -1;
                float bestScore = Float.NEGATIVE_INFINITY;
                for (int e = 0; e < numExperts; e++) {
                    float score = state.moeRouterB.getFloat(rb + e) + (probsB == null ? 0f : probsB.getFloat(e));
                    boolean taken = false;
                    for (int j = 0; j < k; j++) if (topE[j] == e) { taken = true; break; }
                    if (!taken && score > bestScore) { bestScore = score; best = e; }
                }
                topE[k] = best;
                topP[k] = state.moeRouterB.getFloat(rb + best);
            }
            float weightSum = 0f;
            if (config.expertWeightsNorm) {
                for (int k = 0; k < topK; k++) weightSum += topP[k];
                weightSum = Math.max(weightSum, 6.103515625e-5f);
            }
            for (int k = 0; k < topK; k++) {
                float coeff = topP[k];
                if (config.expertWeightsNorm) coeff /= weightSum;
                coeff *= config.expertWeightsScale;
                state.moeRowTopE[s * topK + k] = topE[k];
                state.moeRowTopP[s * topK + k] = coeff;
                counts[topE[k]]++;
            }
        }
        int[] off = state.moeExpertOffsets;
        off[0] = 0;
        for (int e = 0; e < numExperts; e++) off[e + 1] = off[e] + counts[e];
        int[] cursor = state.moeCursor;
        System.arraycopy(off, 0, cursor, 0, numExperts);
        for (int s = 0; s < seqLen; s++) {
            for (int k = 0; k < topK; k++) {
                int e = state.moeRowTopE[s * topK + k];
                int pos = cursor[e]++;
                state.moeRowByExpert[pos] = s;
                state.moeProbByExpert[pos] = state.moeRowTopP[s * topK + k];
            }
        }

        // Experts (grouped): one up/down GEMM per expert over its rows; squared-ReLU; scatter-add.
        state.moeOutB.fillInPlace(0, seqLen * dim, 0f);
        int fDim = dim, fEFF = expertFF;
        for (int e = 0; e < numExperts; e++) {
            int start = off[e], n = off[e + 1] - start;
            if (n == 0) continue;
            Parallel.forRows(n, j -> state.moeInputB.copyTo(state.moeRowByExpert[start + j] * fDim, state.moeGather, j * fDim, fDim));
            int upOffset = e * expertFF * dim;
            int downOffset = e * dim * expertFF;
            w.ffnUpExps[layer].gemm(state.moeGather, dim, state.moeUpB, expertFF, n, expertFF, dim, upOffset);
            Parallel.forRows(n, j -> state.moeUpB.reluSqrInPlace(j * fEFF, fEFF));
            w.ffnDownExps[layer].gemm(state.moeUpB, expertFF, state.moeDownB, dim, n, dim, expertFF, downOffset);
            Parallel.forRows(n, j -> state.moeOutB.saxpyInPlace(state.moeRowByExpert[start + j] * fDim, state.moeDownB, j * fDim, fDim,
                    state.moeProbByExpert[start + j]));
        }

        // Shared expert (batched), squared-ReLU, added with coefficient 1.
        if (w.ffnUpShexp[layer] != null) {
            int sff = config.expertSharedFeedForwardLength;
            int fSFF = sff;
            w.ffnUpShexp[layer].gemm(state.moeInputB, dim, state.moeSharedUpB, sff, seqLen, sff, dim);
            Parallel.forRows(seqLen, s -> state.moeSharedUpB.reluSqrInPlace(s * fSFF, fSFF));
            w.ffnDownShexp[layer].gemm(state.moeSharedUpB, sff, state.moeSharedOutB, dim, seqLen, dim, sff);
            state.moeOutB.addInPlace(0, state.moeSharedOutB, 0, seqLen * dim);
        }

        state.moeOutB.copyTo(0, state.xb, 0, seqLen * dim);
    }

    // === Configuration ===

    enum LayerType { SSM, ATTENTION, MOE }

    record Configuration(int embeddingLength, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                         int headSize, int vocabularySize, int contextLength, float rmsNormEps,
                         LayerType[] layerTypes,
                         int ssmInnerSize, int ssmGroupCount, int ssmTimeStepRank, int ssmStateSize, int ssmConvKernel,
                         int expertCount, int expertUsedCount, int expertFeedForwardLength,
                         int expertSharedFeedForwardLength, boolean expertWeightsNorm, float expertWeightsScale,
                         int bosTokenId, int eosTokenId) {
        int queryDim() { return numberOfHeads * headSize; }
        int kvDim() { return numberOfKeyValueHeads * headSize; }
        int ssmConvChannels() { return ssmInnerSize + 2 * ssmGroupCount * ssmStateSize; }
        int ssmInProjSize() { return 2 * ssmInnerSize + 2 * ssmGroupCount * ssmStateSize + ssmTimeStepRank; }
    }

    // === Weights (per-layer arrays; entries are null for layers of a different type) ===

    record Weights(FloatTensor tokenEmbeddingTable, F32FloatTensor outputNorm, FloatTensor outputWeight,
                   F32FloatTensor[] attnNorm,
                   FloatTensor[] attnQ, FloatTensor[] attnK, FloatTensor[] attnV, FloatTensor[] attnOutput,
                   FloatTensor[] ssmIn, F32FloatTensor[] ssmConv1d, F32FloatTensor[] ssmConv1dB,
                   F32FloatTensor[] ssmA, F32FloatTensor[] ssmD, F32FloatTensor[] ssmDtB, F32FloatTensor[] ssmNorm,
                   FloatTensor[] ssmOut,
                   FloatTensor[] ffnGateInp, F32FloatTensor[] expProbsB, FloatTensor[] ffnUpExps, FloatTensor[] ffnDownExps,
                   FloatTensor[] ffnUpShexp, FloatTensor[] ffnDownShexp) {
    }

    // === State ===

    static final class State implements InferenceState {
        // Capacity c rows: the residual stream, projections and SSM scratch hold one row per chunk
        // token; the single-token reference forward uses row 0. q/att/xb2 and the single-token MoE
        // buffers stay single-row; the batched attention (attnQ/attnOut) and grouped-MoE (moe*B + CSR)
        // buffers are chunk-wide. Per-layer KV / conv-ring / ssm-state caches carry across chunks.
        final int capacity;
        final FloatTensor x, xb, xb2, q, k, v, att, logits;
        final FloatTensor ssmInProj, ssmTmp;
        final float[] ssmZ, ssmXbc, ssmConvOut, ssmY, ssmDt;
        final FloatTensor attnQ, attnOut;     // batched attention (chunk rows)
        final FloatTensor moeRouter, moeHidden, moeExpertOut, moeAccum, moeSharedHidden;   // single-token MoE
        final int[] moeTopExperts;
        final float[] moeTopWeights;
        // Batched grouped-MoE scratch (chunk-wide; CSR grouping of rows by routed expert).
        final FloatTensor moeInputB, moeRouterB, moeOutB, moeGather, moeUpB, moeDownB, moeSharedUpB, moeSharedOutB;
        final int[] moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeRowTopE;
        final float[] moeProbByExpert, moeRowTopP;
        final FloatTensor[] keyCache, valueCache, ssmConvState, ssmState;
        int latestToken;
        boolean logitsValid;
        int lastRowOffset;

        State(Configuration config) {
            int c = Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
            this.capacity = c;
            int dim = config.embeddingLength;
            int queryDim = config.queryDim();
            int kvDim = config.kvDim();
            int dInner = config.ssmInnerSize;
            int convCh = config.ssmConvChannels();

            this.x = F32FloatTensor.allocate(c * dim);
            this.xb = F32FloatTensor.allocate(c * dim);
            this.xb2 = F32FloatTensor.allocate(queryDim);
            this.q = F32FloatTensor.allocate(queryDim);
            this.k = F32FloatTensor.allocate(c * kvDim);
            this.v = F32FloatTensor.allocate(c * kvDim);
            this.att = F32FloatTensor.allocate(config.numberOfHeads * config.contextLength);
            this.logits = F32FloatTensor.allocate(config.vocabularySize);
            this.attnQ = F32FloatTensor.allocate(c * queryDim);
            this.attnOut = F32FloatTensor.allocate(c * queryDim);

            this.ssmInProj = F32FloatTensor.allocate(c * config.ssmInProjSize());
            this.ssmTmp = F32FloatTensor.allocate(c * dInner);
            this.ssmZ = new float[c * dInner];
            this.ssmXbc = new float[c * convCh];
            this.ssmConvOut = new float[c * convCh];
            this.ssmY = new float[c * dInner];
            this.ssmDt = new float[c * config.ssmTimeStepRank];

            int e = Math.max(1, config.expertCount);
            int eff = Math.max(1, config.expertFeedForwardLength);
            int sff = Math.max(1, config.expertSharedFeedForwardLength);
            int tk = Math.max(1, config.expertUsedCount);
            this.moeRouter = F32FloatTensor.allocate(e);
            this.moeHidden = F32FloatTensor.allocate(eff);
            this.moeExpertOut = F32FloatTensor.allocate(dim);
            this.moeAccum = F32FloatTensor.allocate(dim);
            this.moeSharedHidden = F32FloatTensor.allocate(sff);
            this.moeTopExperts = new int[tk];
            this.moeTopWeights = new float[tk];
            this.moeInputB = F32FloatTensor.allocate(c * dim);
            this.moeRouterB = F32FloatTensor.allocate(c * e);
            this.moeOutB = F32FloatTensor.allocate(c * dim);
            this.moeGather = F32FloatTensor.allocate(c * dim);
            this.moeUpB = F32FloatTensor.allocate(c * eff);
            this.moeDownB = F32FloatTensor.allocate(c * dim);
            this.moeSharedUpB = F32FloatTensor.allocate(c * sff);
            this.moeSharedOutB = F32FloatTensor.allocate(c * dim);
            this.moeExpertCounts = new int[e];
            this.moeExpertOffsets = new int[e + 1];
            this.moeCursor = new int[e];
            this.moeRowByExpert = new int[c * tk];
            this.moeRowTopE = new int[c * tk];
            this.moeProbByExpert = new float[c * tk];
            this.moeRowTopP = new float[c * tk];

            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            this.ssmConvState = new FloatTensor[config.numberOfLayers];
            this.ssmState = new FloatTensor[config.numberOfLayers];
            for (int l = 0; l < config.numberOfLayers; l++) {
                switch (config.layerTypes[l]) {
                    case ATTENTION -> {
                        keyCache[l] = F32FloatTensor.allocate(config.contextLength * kvDim);
                        valueCache[l] = F32FloatTensor.allocate(config.contextLength * kvDim);
                    }
                    case SSM -> {
                        ssmConvState[l] = F32FloatTensor.allocate((config.ssmConvKernel - 1) * convCh);
                        ssmState[l] = F32FloatTensor.allocate(dInner * config.ssmStateSize);
                    }
                    case MOE -> { }
                }
            }
        }

        @Override public int latestToken() { return latestToken; }

        @Override public void latestToken(int token) { this.latestToken = token; }
    }

    // === Loading ===

    static Nemotron loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load Nemotron model")) {
            try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
                return loadModel(fileChannel, gguf, contextLength, true);
            }
        }
    }

    static Nemotron loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        LFMTokenizer tokenizer = new LFMTokenizer(gguf);
        String arch = gguf.getString("general.architecture");

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) {
            contextLength = modelContextLength;
        }
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int headSize = gguf.getValueOrDefault(int.class, arch + ".attention.key_length", embeddingLength / Math.max(1, numberOfHeads));
        float rmsNormEps = gguf.getValueOrDefault(float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);

        // Per-layer type: SSM when both head_count_kv and feed_forward_length are 0; attention when
        // head_count_kv > 0; MoE-FFN when feed_forward_length > 0.
        int[] headCountKv = gguf.getValue(int[].class, arch + ".attention.head_count_kv");
        int[] feedForward = gguf.getValue(int[].class, arch + ".feed_forward_length");
        LayerType[] layerTypes = new LayerType[numberOfLayers];
        int kvHeads = 0;
        for (int i = 0; i < numberOfLayers; i++) {
            if (feedForward[i] > 0) {
                layerTypes[i] = LayerType.MOE;
            } else if (headCountKv[i] > 0) {
                layerTypes[i] = LayerType.ATTENTION;
                kvHeads = headCountKv[i];
            } else {
                layerTypes[i] = LayerType.SSM;
            }
        }

        int ssmInnerSize = gguf.getValueOrDefault(int.class, arch + ".ssm.inner_size", 0);
        int ssmGroupCount = gguf.getValueOrDefault(int.class, arch + ".ssm.group_count", 0);
        int ssmTimeStepRank = gguf.getValueOrDefault(int.class, arch + ".ssm.time_step_rank", 0);
        int ssmStateSize = gguf.getValueOrDefault(int.class, arch + ".ssm.state_size", 0);
        int ssmConvKernel = gguf.getValueOrDefault(int.class, arch + ".ssm.conv_kernel", 0);

        int expertCount = gguf.getValueOrDefault(int.class, arch + ".expert_count", 0);
        int expertUsedCount = gguf.getValueOrDefault(int.class, arch + ".expert_used_count", 0);
        int expertFeedForwardLength = gguf.getValueOrDefault(int.class, arch + ".expert_feed_forward_length", 0);
        int expertSharedFeedForwardLength = gguf.getValueOrDefault(int.class, arch + ".expert_shared_feed_forward_length", 0);
        boolean expertWeightsNorm = gguf.getValueOrDefault(boolean.class, arch + ".expert_weights_norm", false);
        float expertWeightsScale = gguf.getValueOrDefault(float.class, arch + ".expert_weights_scale", 1.0f);

        int bosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.bos_token_id", 0);
        int eosTokenId = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);

        Configuration config = new Configuration(embeddingLength, numberOfLayers, numberOfHeads, kvHeads,
                headSize, tokenizer.vocabularySize(), contextLength, rmsNormEps, layerTypes,
                ssmInnerSize, ssmGroupCount, ssmTimeStepRank, ssmStateSize, ssmConvKernel,
                expertCount, expertUsedCount, expertFeedForwardLength, expertSharedFeedForwardLength,
                expertWeightsNorm, expertWeightsScale, bosTokenId, eosTokenId);

        if (!loadWeightsFlag) {
            return new Nemotron(config, tokenizer, null);
        }
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        return new Nemotron(config, tokenizer, loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
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
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ssm_in.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_conv1d.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_conv1d.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_a")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_d")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_dt.bias")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ssm_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ssm_out.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate_inp.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".exp_probs_b.bias")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up_exps.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down_exps.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up_shexp.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down_shexp.weight")));
    }
}
