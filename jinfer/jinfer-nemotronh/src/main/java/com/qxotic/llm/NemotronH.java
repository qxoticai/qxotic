// Nemotron-H ("nemotron_h" / "nemotron_h_moe", Nemotron Cascade 2) against the com.qxotic.jinfer model
// API: a faithful port of the production jinfer Nemotron forward. A hybrid where each block is exactly
// ONE of {Mamba2-SSM mixer, full-attention, MoE-FFN}, behind a single pre-norm. Architecture specifics:
// NO RoPE and NO q/k norm in attention; the MoE router is SIGMOID with an additive selection bias
// (exp_probs_b) while the combine weight is the UNBIASED sigmoid (optionally normalized + scaled);
// all FFN/expert activations are squared-ReLU (no gate projection); a shared expert is always added.
//
// Batched prefill + single-token decode: the per-token projections (SSM in/conv/out, attention Q/K/V/O,
// MoE router/experts) batch into GEMMs over the chunk, while only the Mamba2 recurrence stays sequential
// (one parallelFor over heads, rows sequential inside, conv-ring + SSM state carried in State). Decode
// (seqLen==1) takes the single-token cores. Token-exact vs the production. Uses public kernels (gemm,
// gemm-with-offset, causalPrefill/flashDecode, Moe.dispatch) plus the shared scalars (silu/sigmoid/
// softplus/reluSqr) from LLM.
package com.qxotic.llm;

import com.qxotic.format.gguf.GGUF;

import com.qxotic.jinfer.*;

import static com.qxotic.jinfer.Norms.rmsnorm;
import static com.qxotic.llm.LLM.reluSqr;
import static com.qxotic.llm.LLM.sigmoid;
import static com.qxotic.llm.LLM.silu;
import static com.qxotic.llm.LLM.softplus;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class NemotronH implements LanguageModel<NemotronH.Configuration, NemotronH.Weights, NemotronH.State> {

    private final Configuration configuration;
    private final LFMTokenizer tokenizer;
    private final Weights weights;

    NemotronH(Configuration configuration, LFMTokenizer tokenizer, Weights weights) {
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
    public void ingest(State s, Batch batch) {
        int n = batch.count();
        if (n > s.batchCapacity) throw new IllegalArgumentException("batch " + n + " exceeds batchCapacity " + s.batchCapacity);
        int from = s.position();
        if (from + n > s.contextCapacity) {
            throw new IllegalArgumentException("ingest of " + n + " at position " + from + " exceeds contextCapacity " + s.contextCapacity);
        }
        switch (batch.input()) {
            case Batch.Input.Tokens t -> forward(s, t.ids(), 0, from, n);   // one batched pass (decode cores at n==1)
            case Batch.Input.Sequences seq ->
                throw new UnsupportedOperationException("Nemotron-H is generative: packed sequences (batched embedding) not supported");
            case Batch.Input.Embeddings e ->
                throw new UnsupportedOperationException("Nemotron-H is text-only: embedding input is not supported");
        }
        s.lastChunkLen = n;
        s.outputCount = batch.outputs() == Batch.Outputs.ALL ? n : 1;
        s.position = from + n;
    }

    @Override
    public FloatTensor logits(State s, int output) {
        int dim = configuration.embeddingLength;
        int row = s.lastChunkLen - s.outputCount + output;
        rmsnorm(s.xb, 0, s.x, (long) row * dim, weights.outputNorm, dim, configuration.rmsNormEps);
        weights.outputWeight.matmul(s.xb, s.logits, configuration.vocabularySize, dim);
        return s.logits;
    }

    @Override
    public State fork(State s) {
        State f = new State(configuration, s.contextCapacity, s.batchCapacity);
        for (int l = 0; l < configuration.numberOfLayers; l++) {
            if (s.keyCache[l] != null) {
                int len = s.position * configuration.kvDim();
                s.keyCache[l].copyTo(0, f.keyCache[l], 0, len);
                s.valueCache[l].copyTo(0, f.valueCache[l], 0, len);
            }
            if (s.ssmConvState[l] != null) {
                int len = (int) s.ssmConvState[l].size();
                s.ssmConvState[l].copyTo(0, f.ssmConvState[l], 0, len);
                System.arraycopy(s.ssmState[l], 0, f.ssmState[l], 0, s.ssmState[l].length);
            }
        }
        f.position = s.position;
        return f;
    }

    /** The eos / turn-delimiter ids that terminate generation (convenience for callers/tests). */
    public Set<Integer> stopTokens() {
        Set<Integer> stops = new HashSet<>();
        if (configuration.eosTokenId >= 0) stops.add(configuration.eosTokenId);
        for (String name : new String[]{"<|im_end|>", "<|endoftext|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        return stops;
    }

    // === Forward (single token at `position`, residual written to row `row` of state.x) ===


    /** Batched forward over the chunk: embed all rows, then per block a batched pre-norm feeds whichever
     *  mixer this layer is. seqLen==1 takes the single-token cores (decode); seqLen>1 the batched cores
     *  (prefill) — projections become GEMMs, only the Mamba2 recurrence stays sequential (per head). */
    void forward(State s, int[] ids, int tokenOffset, int startPos, int seqLen) {
        Configuration c = configuration;
        int dim = c.embeddingLength;
        float eps = c.rmsNormEps;
        for (int i = 0; i < seqLen; i++) {
            weights.tokenEmbeddings.copyTo((long) ids[tokenOffset + i] * dim, s.x, (long) i * dim, dim);
        }
        for (int l = 0; l < c.numberOfLayers; l++) {
            F32FloatTensor attNorm = weights.layers[l].attnNorm();
            Parallel.forRows(seqLen, row -> rmsnorm(s.xb, (long) row * dim, s.x, (long) row * dim, attNorm, dim, eps));
            switch (c.layerTypes[l]) {
                case SSM       -> { if (seqLen == 1) ssmForward(s, l); else ssmForwardBatch(s, l, seqLen); }
                case ATTENTION -> attention(s, l, startPos, seqLen);
                case MOE       -> { if (seqLen == 1) moeForward(s, l); else moeForwardBatch(s, l, seqLen); }
            }
            s.x.addInPlace(0, s.xb, 0, seqLen * dim);   // residual; mixer left its output in xb
            if (LLM.TRACE) LLM.traceSum("l" + l + "-" + c.layerTypes[l], s.x, seqLen * dim);
        }
    }

    /** Mamba2 SSM mixer: in-proj → split (z|xBC|dt) → depthwise causal conv1d+SiLU → selective scan
     *  (per-head dt/A, per-group B/C, D skip, SiLU(z) gate) → per-group gated RMSNorm → out-proj.
     *  Reads state.xb (normed), writes the result back to state.xb. */
    private void ssmForward(State state, int l) {
        Configuration c = configuration;
        SsmWeights w = weights.layers[l].ssm();
        int dim = c.embeddingLength;
        int dInner = c.ssmInnerSize, dState = c.ssmStateSize, nHead = c.ssmTimeStepRank, nGroup = c.ssmGroupCount, dConv = c.ssmConvKernel;
        int headDim = dInner / nHead, convCh = c.ssmConvChannels(), qSize = nGroup * dState;
        float eps = c.rmsNormEps;

        // 1. in-projection, split into z (gate) | xBC (conv input) | dt (per ssm head)
        w.inProj().matmul(state.xb, state.ssmInProj, c.ssmInProjSize(), dim);
        float[] z = state.ssmZ, xbc = state.ssmXbc, convOut = state.ssmConvOut, y = state.ssmY, dt = state.ssmDt;
        for (int i = 0; i < dInner; i++) z[i] = state.ssmInProj.getFloat(i);
        for (int i = 0; i < convCh; i++) xbc[i] = state.ssmInProj.getFloat(dInner + i);
        int dtOff = 2 * dInner + 2 * nGroup * dState;
        for (int h = 0; h < nHead; h++) dt[h] = softplus(state.ssmInProj.getFloat(dtOff + h) + w.dtBias().getFloat(h));

        // 2. depthwise causal conv1d over the dConv window (ring state + current), bias, SiLU
        FloatTensor convState = state.ssmConvState[l];
        F32FloatTensor convW = w.conv1d(), convB = w.conv1dBias();
        for (int ch = 0; ch < convCh; ch++) {
            float sum = convB == null ? 0f : convB.getFloat(ch);
            int wOff = ch * dConv;
            for (int k = 0; k < dConv - 1; k++) sum += convW.getFloat(wOff + k) * convState.getFloat(k * convCh + ch);
            sum += convW.getFloat(wOff + dConv - 1) * xbc[ch];
            convOut[ch] = silu(sum);
        }
        for (int k = 0; k < dConv - 2; k++)
            for (int ch = 0; ch < convCh; ch++) convState.setFloat(k * convCh + ch, convState.getFloat((k + 1) * convCh + ch));
        for (int ch = 0; ch < convCh; ch++) convState.setFloat((dConv - 2) * convCh + ch, xbc[ch]);

        // 3. selective scan: per head h_new = h*exp(dt*A) + B*(x*dt); y = C.h + D*x; gate SiLU(z)
        float[] S = state.ssmState[l];
        for (int h = 0; h < nHead; h++) {
            int g = h / (nHead / nGroup);
            float dA = (float) Math.exp(dt[h] * w.a().getFloat(h));
            float dScale = w.d().getFloat(h);
            for (int ii = 0; ii < headDim; ii++) {
                int idx = h * headDim + ii;
                float xdt = convOut[idx] * dt[h];
                int stBase = idx * dState, bOff = dInner + g * dState, cOff = dInner + qSize + g * dState;
                float sum = 0f;
                for (int i0 = 0; i0 < dState; i0++) {
                    float next = S[stBase + i0] * dA + convOut[bOff + i0] * xdt;
                    S[stBase + i0] = next;
                    sum += next * convOut[cOff + i0];
                }
                y[idx] = (sum + convOut[idx] * dScale) * silu(z[idx]);
            }
        }

        // 4. per-group gated RMSNorm (gate already folded into y) then out-projection
        int groupDim = dInner / nGroup;
        F32FloatTensor normW = w.norm();
        for (int g = 0; g < nGroup; g++) {
            int off = g * groupDim;
            float ss = 0f;
            for (int i = 0; i < groupDim; i++) ss += y[off + i] * y[off + i];
            float inv = (float) (1.0 / Math.sqrt(ss / groupDim + eps));
            for (int i = 0; i < groupDim; i++) state.ssmTmp.setFloat(off + i, y[off + i] * inv * normW.getFloat(off + i));
        }
        w.outProj().matmul(state.ssmTmp, state.xb, dim, dInner);
    }

    /** Full causal GQA attention — NO RoPE / q-k norm / biases: Q/K/V projections, K/V appended to the
     *  cache at `position`, sink-free flash decode, output projection written back to state.xb. */
    private void attention(State state, int l, int startPos, int seqLen) {
        Configuration c = configuration;
        AttentionWeights w = weights.layers[l].attention();
        int dim = c.embeddingLength, headSize = c.headSize, heads = c.numberOfHeads;
        int kvDim = c.kvDim(), queryDim = c.queryDim(), kvMul = heads / c.numberOfKeyValueHeads;

        w.wq().gemm(state.xb, dim, state.attnQ, queryDim, seqLen, queryDim, dim);
        w.wk().gemm(state.xb, dim, state.k, kvDim, seqLen, kvDim, dim);
        w.wv().gemm(state.xb, dim, state.v, kvDim, seqLen, kvDim, dim);

        FloatTensor keyCache = state.keyCache[l], valueCache = state.valueCache[l];
        for (int s = 0; s < seqLen; s++) {   // commit this chunk's K/V to the contiguous cache
            state.k.copyTo((long) s * kvDim, keyCache, (long) (startPos + s) * kvDim, kvDim);
            state.v.copyTo((long) s * kvDim, valueCache, (long) (startPos + s) * kvDim, kvDim);
        }

        if (seqLen == 1) {
            float attScale = 1.0f / (float) Math.sqrt(headSize);
            FlashAttention.flashDecode((F32FloatTensor) state.attnQ, (F32FloatTensor) state.attnOut,
                    keyCache, valueCache, null, null, heads, startPos, 0, headSize, kvDim, kvMul, attScale, 0, null, state.decodeScratch);
        } else {
            FlashAttention.causalPrefill((F32FloatTensor) state.attnQ, (F32FloatTensor) state.attnOut,
                    keyCache, valueCache, heads, startPos, seqLen, headSize, kvDim, queryDim, kvMul);
        }
        w.wo().gemm(state.attnOut, queryDim, state.xb, dim, seqLen, dim, queryDim);
    }

    /** MoE-FFN: sigmoid router with additive selection bias (top-k by biased score, combine weight is the
     *  UNBIASED sigmoid, optionally normalized then scaled); squared-ReLU experts (no gate) plus an
     *  always-on shared expert. Reads state.xb (normed), writes the result back to state.xb. */
    private void moeForward(State state, int l) {
        Configuration c = configuration;
        MoeFfnWeights w = weights.layers[l].moe();
        int dim = c.embeddingLength, expertFF = c.expertFeedForwardLength, numExperts = c.expertCount;
        int topK = Math.min(c.expertUsedCount, numExperts);

        w.router().matmul(state.xb, state.moeRouter, numExperts, dim);
        for (int e = 0; e < numExperts; e++) state.moeRouter.setFloat(e, sigmoid(state.moeRouter.getFloat(e)));

        int[] topE = state.moeTopExperts;
        float[] topP = state.moeTopWeights;
        selectTopK(state.moeRouter, 0, w.expProbsB(), numExperts, topK, topE, topP);

        float weightSum = 0f;
        if (c.expertWeightsNorm) {
            for (int k = 0; k < topK; k++) weightSum += topP[k];
            weightSum = Math.max(weightSum, 6.103515625e-5f);   // 2^-14 floor
        }
        for (int k = 0; k < topK; k++) {
            float coeff = topP[k];
            if (c.expertWeightsNorm) coeff /= weightSum;
            coeff *= c.expertWeightsScale;
            topP[k] = coeff;
        }

        for (int i = 0; i < dim; i++) state.moeAccum.setFloat(i, 0f);
        for (int k = 0; k < topK; k++) {
            int e = topE[k];
            w.upExps().gemm(state.xb, dim, state.moeHidden, expertFF, 1, expertFF, dim, (long) e * expertFF * dim);
            reluSqr(state.moeHidden, 0, expertFF);
            w.downExps().gemm(state.moeHidden, expertFF, state.moeExpertOut, dim, 1, dim, expertFF, (long) e * dim * expertFF);
            float coeff = topP[k];
            for (int i = 0; i < dim; i++) state.moeAccum.setFloat(i, state.moeAccum.getFloat(i) + state.moeExpertOut.getFloat(i) * coeff);
        }

        if (w.upShexp() != null) {   // shared expert (always active), squared-ReLU, coefficient 1
            int sff = c.expertSharedFeedForwardLength;
            w.upShexp().matmul(state.xb, state.moeSharedHidden, sff, dim);
            reluSqr(state.moeSharedHidden, 0, sff);
            w.downShexp().matmul(state.moeSharedHidden, state.moeExpertOut, dim, sff);
            for (int i = 0; i < dim; i++) state.moeAccum.setFloat(i, state.moeAccum.getFloat(i) + state.moeExpertOut.getFloat(i));
        }

        state.moeAccum.copyTo(0, state.xb, 0, dim);
    }

    /** Top-k experts by (sigmoid router + probsB bias), keeping the UNBIASED router value as the combine
     *  weight. Non-destructive "not already taken" scan, so the router tensor is left intact. */
    private static void selectTopK(FloatTensor router, int base, F32FloatTensor probsB, int numExperts,
                                   int topK, int[] topE, float[] topP) {
        for (int k = 0; k < topK; k++) {
            int best = -1;
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int e = 0; e < numExperts; e++) {
                float score = router.getFloat(base + e) + (probsB == null ? 0f : probsB.getFloat(e));
                boolean taken = false;
                for (int j = 0; j < k; j++) if (topE[j] == e) { taken = true; break; }
                if (!taken && score > bestScore) { bestScore = score; best = e; }
            }
            topE[k] = best;
            topP[k] = router.getFloat(base + best);
        }
    }

    // === Batched mixers (prefill): projections become GEMMs over the chunk; only the Mamba2 recurrence
    //     stays sequential (one parallelFor over heads, rows sequential inside). Token-exact vs decode. ===

    /** Batched Mamba2 SSM: in-proj GEMM, batched causal conv1d (parallel over channels, reads the ring for
     *  negative positions), selective scan parallel-over-heads + sequential-rows (state carries), per-group
     *  gated RMSNorm, out-proj GEMM. Mirrors {@link #ssmForward} row-for-row. */
    private void ssmForwardBatch(State state, int l, int seqLen) {
        Configuration c = configuration;
        SsmWeights w = weights.layers[l].ssm();
        int dim = c.embeddingLength;
        int dInner = c.ssmInnerSize, dState = c.ssmStateSize, nHead = c.ssmTimeStepRank, nGroup = c.ssmGroupCount, dConv = c.ssmConvKernel;
        int headDim = dInner / nHead, convCh = c.ssmConvChannels(), qSize = nGroup * dState, inProjSize = c.ssmInProjSize();
        float eps = c.rmsNormEps;

        // 1. in-projection GEMM; split each row into z | xBC | dt.
        w.inProj().gemm(state.xb, dim, state.ssmInProj, inProjSize, seqLen, inProjSize, dim);
        float[] z = state.ssmZ, xbc = state.ssmXbc, convOut = state.ssmConvOut, y = state.ssmY, dt = state.ssmDt;
        int dtOff = 2 * dInner + 2 * nGroup * dState;
        int fInProj = inProjSize, fDInner = dInner, fConvCh = convCh, fNHead = nHead, fSeq = seqLen;
        F32FloatTensor dtB = w.dtBias();
        Parallel.forRows(seqLen, s -> {
            int p = s * fInProj;
            for (int i = 0; i < fDInner; i++) z[s * fDInner + i] = state.ssmInProj.getFloat(p + i);
            for (int i = 0; i < fConvCh; i++) xbc[s * fConvCh + i] = state.ssmInProj.getFloat(p + fDInner + i);
            for (int h = 0; h < fNHead; h++) dt[s * fNHead + h] = softplus(state.ssmInProj.getFloat(p + dtOff + h) + dtB.getFloat(h));
        });

        // 2. batched depthwise causal conv1d + SiLU; row s reads taps s-(dConv-1)+k from chunk or conv ring.
        FloatTensor convState = state.ssmConvState[l];
        F32FloatTensor convW = w.conv1d(), convBias = w.conv1dBias();
        int fK = dConv, fHist = dConv - 1;
        Parallel.parallelFor(0, convCh, ch -> {
            int wOff = ch * fK;
            for (int s = 0; s < fSeq; s++) {
                float sum = convBias == null ? 0f : convBias.getFloat(ch);
                for (int k = 0; k < fK; k++) {
                    int pos = s - fHist + k;
                    float in = pos < 0 ? convState.getFloat((pos + fHist) * fConvCh + ch) : xbc[pos * fConvCh + ch];
                    sum += convW.getFloat(wOff + k) * in;
                }
                convOut[s * fConvCh + ch] = silu(sum);
            }
        });
        Parallel.parallelFor(0, convCh, ch -> {   // roll the conv ring: keep the last dConv-1 xBC rows
            for (int k = 0; k < fHist; k++) {
                int pos = fSeq - fHist + k;
                float v = pos < 0 ? convState.getFloat((pos + fHist) * fConvCh + ch) : xbc[pos * fConvCh + ch];
                convState.setFloat(k * fConvCh + ch, v);
            }
        });

        // 3. selective scan: parallel over heads (independent state), sequential over rows inside.
        float[] S = state.ssmState[l];
        int fHeadDim = headDim, fDState = dState, fNGroup = nGroup, fQSize = qSize;
        F32FloatTensor ssmA = w.a(), ssmD = w.d();
        Parallel.parallelFor(0, fNHead, h -> {
            int g = h / (fNHead / fNGroup);
            float A_h = ssmA.getFloat(h), D_h = ssmD.getFloat(h);
            for (int s = 0; s < fSeq; s++) {
                int cBase = s * fConvCh, base = s * fDInner, dtBase = s * fNHead;
                float dtH = dt[dtBase + h];
                float dA = (float) Math.exp(dtH * A_h);
                for (int ii = 0; ii < fHeadDim; ii++) {
                    int idx = h * fHeadDim + ii;
                    float xdt = convOut[cBase + idx] * dtH;
                    int stBase = idx * fDState, bOff = cBase + fDInner + g * fDState, cOff = cBase + fDInner + fQSize + g * fDState;
                    float sum = 0f;
                    for (int i0 = 0; i0 < fDState; i0++) {
                        float next = S[stBase + i0] * dA + convOut[bOff + i0] * xdt;
                        S[stBase + i0] = next;
                        sum += next * convOut[cOff + i0];
                    }
                    y[base + idx] = (sum + convOut[cBase + idx] * D_h) * silu(z[base + idx]);
                }
            }
        });

        // 4. per-group gated RMSNorm (gate folded into y) then out-projection GEMM.
        int groupDim = dInner / nGroup;
        F32FloatTensor normW = w.norm();
        int fGroupDim = groupDim;
        Parallel.forRows(seqLen, s -> {
            for (int g = 0; g < fNGroup; g++) {
                int off = s * fDInner + g * fGroupDim;
                float ss = 0f;
                for (int i = 0; i < fGroupDim; i++) ss += y[off + i] * y[off + i];
                float inv = (float) (1.0 / Math.sqrt(ss / fGroupDim + eps));
                for (int i = 0; i < fGroupDim; i++) state.ssmTmp.setFloat(off + i, y[off + i] * inv * normW.getFloat(g * fGroupDim + i));
            }
        });
        w.outProj().gemm(state.ssmTmp, dInner, state.xb, dim, seqLen, dim, dInner);
    }

    /** Batched MoE: router GEMM + per-row sigmoid/top-k/normalize, CSR gather-by-expert squared-ReLU GEMMs
     *  via the shared {@link Moe#dispatch}, plus the batched shared expert. Mirrors {@link #moeForward}. */
    private void moeForwardBatch(State state, int l, int seqLen) {
        Configuration c = configuration;
        MoeFfnWeights w = weights.layers[l].moe();
        int dim = c.embeddingLength, expertFF = c.expertFeedForwardLength, numExperts = c.expertCount;
        int topK = Math.min(c.expertUsedCount, numExperts);

        state.xb.copyTo(0, state.moeInputB, 0, seqLen * dim);   // snapshot (CSR scrambles rows; router/shared read it)
        w.router().gemm(state.moeInputB, dim, state.moeRouterB, numExperts, seqLen, numExperts, dim);

        int[] counts = state.moeExpertCounts;
        F32FloatTensor probsB = w.expProbsB();
        int fTopK = topK, fNumExperts = numExperts;
        boolean fNorm = c.expertWeightsNorm; float fScale = c.expertWeightsScale;
        Parallel.forRows(seqLen, s -> {   // per-row routing is independent (selectTopK is non-destructive)
            int rb = s * fNumExperts;
            for (int e = 0; e < fNumExperts; e++) state.moeRouterB.setFloat(rb + e, sigmoid(state.moeRouterB.getFloat(rb + e)));
            int[] te = new int[fTopK];
            float[] tp = new float[fTopK];
            selectTopK(state.moeRouterB, rb, probsB, fNumExperts, fTopK, te, tp);   // top-k by sigmoid+bias; combine = unbiased
            float weightSum = 0f;
            if (fNorm) {
                for (int k = 0; k < fTopK; k++) weightSum += tp[k];
                weightSum = Math.max(weightSum, 6.103515625e-5f);
            }
            for (int k = 0; k < fTopK; k++) {
                float coeff = tp[k];
                if (fNorm) coeff /= weightSum;
                coeff *= fScale;
                state.moeRowTopE[s * fTopK + k] = te[k];
                state.moeRowTopP[s * fTopK + k] = coeff;
            }
        });
        java.util.Arrays.fill(counts, 0);   // counts sequential (race-free) after the parallel per-row routing
        for (int s = 0; s < seqLen; s++) for (int k = 0; k < topK; k++) counts[state.moeRowTopE[s * topK + k]]++;
        Moe.Routing r = state.moeRouting;
        r.seqLen = seqLen; r.topK = topK; r.numExperts = numExperts;
        Moe.dispatch(r, dim, state.moeInputB, state.moeGather, state.moeDownB, state.moeOutB, null,
                (e, n, gather, out) -> {
                    w.upExps().gemm(gather, dim, state.moeUpB, expertFF, n, expertFF, dim, (long) e * expertFF * dim);
                    Parallel.forRows(n, j -> reluSqr(state.moeUpB, j * expertFF, expertFF));
                    w.downExps().gemm(state.moeUpB, expertFF, out, dim, n, dim, expertFF, (long) e * dim * expertFF);
                });

        if (w.upShexp() != null) {   // shared expert (batched), squared-ReLU, coefficient 1
            int sff = c.expertSharedFeedForwardLength, fSFF = sff;
            w.upShexp().gemm(state.moeInputB, dim, state.moeSharedUpB, sff, seqLen, sff, dim);
            Parallel.forRows(seqLen, s -> reluSqr(state.moeSharedUpB, s * fSFF, fSFF));
            w.downShexp().gemm(state.moeSharedUpB, sff, state.moeSharedOutB, dim, seqLen, dim, sff);
            state.moeOutB.addInPlace(0, state.moeSharedOutB, 0, seqLen * dim);
        }
        state.moeOutB.copyTo(0, state.xb, 0, seqLen * dim);
    }

    // === Configuration ===

    public enum LayerType { SSM, ATTENTION, MOE }

    public record Configuration(int embeddingLength, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                                int headSize, int vocabularySize, int contextLength, float rmsNormEps,
                                LayerType[] layerTypes,
                                int ssmInnerSize, int ssmGroupCount, int ssmTimeStepRank, int ssmStateSize, int ssmConvKernel,
                                int expertCount, int expertUsedCount, int expertFeedForwardLength,
                                int expertSharedFeedForwardLength, boolean expertWeightsNorm, float expertWeightsScale,
                                int bosTokenId, int eosTokenId) implements Config {
        int queryDim() { return numberOfHeads * headSize; }
        int kvDim() { return numberOfKeyValueHeads * headSize; }
        int ssmConvChannels() { return ssmInnerSize + 2 * ssmGroupCount * ssmStateSize; }
        int ssmInProjSize() { return 2 * ssmInnerSize + 2 * ssmGroupCount * ssmStateSize + ssmTimeStepRank; }
    }

    // === Weights (LayerWeights[] record-of-records; exactly one sub-record set per layer) ===

    public record AttentionWeights(FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo) {}

    public record SsmWeights(FloatTensor inProj, F32FloatTensor conv1d, F32FloatTensor conv1dBias,
                             F32FloatTensor a, F32FloatTensor d, F32FloatTensor dtBias, F32FloatTensor norm,
                             FloatTensor outProj) {}

    public record MoeFfnWeights(FloatTensor router, F32FloatTensor expProbsB, FloatTensor upExps, FloatTensor downExps,
                                FloatTensor upShexp, FloatTensor downShexp) {}

    public record LayerWeights(F32FloatTensor attnNorm, AttentionWeights attention, SsmWeights ssm, MoeFfnWeights moe) {}

    public record Weights(FloatTensor tokenEmbeddings, F32FloatTensor outputNorm, FloatTensor outputWeight,
                          LayerWeights[] layers) {}

    // === State ===

    public static final class State implements RuntimeState {
        final int contextCapacity, batchCapacity;
        int position, outputCount, lastChunkLen;
        final FloatTensor x;                                   // batchCapacity rows: residual per token
        final FloatTensor xb, k, v, attnQ, attnOut, logits;    // chunk-wide scratch
        final FloatTensor ssmInProj, ssmTmp;
        final float[] ssmZ, ssmXbc, ssmConvOut, ssmY, ssmDt;
        final FloatTensor moeRouter, moeHidden, moeExpertOut, moeAccum, moeSharedHidden;   // single-token decode scratch
        final int[] moeTopExperts;
        final float[] moeTopWeights;
        // batched MoE (prefill): CSR routing + chunk-wide expert/shared buffers
        final FloatTensor moeInputB, moeRouterB, moeGather, moeDownB, moeOutB, moeUpB, moeSharedUpB, moeSharedOutB;
        final int[] moeRowTopE, moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert;
        final float[] moeRowTopP, moeProbByExpert;
        final Moe.Routing moeRouting;
        final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch();
        final FloatTensor[] keyCache, valueCache, ssmConvState;   // attention / SSM layers respectively
        final float[][] ssmState;                                // Mamba2 recurrent state, raw heap array

        State(Configuration config, int contextCapacity, int batchCapacity) {
            this.contextCapacity = contextCapacity;
            this.batchCapacity = Math.max(1, batchCapacity);
            int cap = this.batchCapacity;
            int dim = config.embeddingLength;
            int queryDim = config.queryDim(), kvDim = config.kvDim();
            int dInner = config.ssmInnerSize, convCh = config.ssmConvChannels();

            this.x = FloatTensor.allocateF32(cap * dim);
            this.xb = FloatTensor.allocateF32(cap * dim);
            this.k = FloatTensor.allocateF32(cap * kvDim);
            this.v = FloatTensor.allocateF32(cap * kvDim);
            this.attnQ = FloatTensor.allocateF32(cap * queryDim);
            this.attnOut = FloatTensor.allocateF32(cap * queryDim);
            this.logits = FloatTensor.allocateF32(config.vocabularySize);
            this.ssmInProj = FloatTensor.allocateF32(cap * config.ssmInProjSize());
            this.ssmTmp = FloatTensor.allocateF32(cap * Math.max(1, dInner));
            this.ssmZ = new float[cap * Math.max(1, dInner)];
            this.ssmXbc = new float[cap * Math.max(1, convCh)];
            this.ssmConvOut = new float[cap * Math.max(1, convCh)];
            this.ssmY = new float[cap * Math.max(1, dInner)];
            this.ssmDt = new float[cap * Math.max(1, config.ssmTimeStepRank)];

            int e = Math.max(1, config.expertCount), eff = Math.max(1, config.expertFeedForwardLength);
            int sff = Math.max(1, config.expertSharedFeedForwardLength), tk = Math.max(1, config.expertUsedCount);
            this.moeRouter = FloatTensor.allocateF32(e);
            this.moeHidden = FloatTensor.allocateF32(eff);
            this.moeExpertOut = FloatTensor.allocateF32(dim);
            this.moeAccum = FloatTensor.allocateF32(dim);
            this.moeSharedHidden = FloatTensor.allocateF32(sff);
            this.moeTopExperts = new int[tk];
            this.moeTopWeights = new float[tk];
            this.moeInputB = FloatTensor.allocateF32(cap * dim);
            this.moeRouterB = FloatTensor.allocateF32(cap * e);
            this.moeGather = FloatTensor.allocateF32(cap * dim);
            this.moeDownB = FloatTensor.allocateF32(cap * dim);
            this.moeOutB = FloatTensor.allocateF32(cap * dim);
            this.moeUpB = FloatTensor.allocateF32(cap * eff);
            this.moeSharedUpB = FloatTensor.allocateF32(cap * sff);
            this.moeSharedOutB = FloatTensor.allocateF32(cap * dim);
            this.moeRowTopE = new int[cap * tk];
            this.moeRowTopP = new float[cap * tk];
            this.moeExpertCounts = new int[e];
            this.moeExpertOffsets = new int[e + 1];
            this.moeCursor = new int[e];
            this.moeRowByExpert = new int[cap * tk];
            this.moeProbByExpert = new float[cap * tk];
            this.moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeProbByExpert);

            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            this.ssmConvState = new FloatTensor[config.numberOfLayers];
            this.ssmState = new float[config.numberOfLayers][];
            for (int l = 0; l < config.numberOfLayers; l++) {
                switch (config.layerTypes[l]) {
                    case ATTENTION -> {
                        keyCache[l] = FloatTensor.allocateF16(contextCapacity, kvDim);
                        valueCache[l] = FloatTensor.allocateF16(contextCapacity, kvDim);
                    }
                    case SSM -> {
                        ssmConvState[l] = FloatTensor.allocateF32((config.ssmConvKernel - 1) * convCh);
                        ssmState[l] = new float[dInner * config.ssmStateSize];
                    }
                    case MOE -> { }
                }
            }
        }

        @Override public int contextCapacity() { return contextCapacity; }
        @Override public int batchCapacity()   { return batchCapacity; }
        @Override public int position()         { return position; }
        @Override public int outputCount()      { return outputCount; }
    }

    // === Loading ===

    public static NemotronH loadModel(Path ggufPath, int contextLength) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, contextLength);
        }
    }

    public static NemotronH loadModel(FileChannel fileChannel, GGUF gguf, int contextLength) throws IOException {
        LFMTokenizer tokenizer = new LFMTokenizer(gguf);
        String arch = gguf.getString("general.architecture");

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) contextLength = modelContextLength;
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int headSize = gguf.getValueOrDefault(int.class, arch + ".attention.key_length", embeddingLength / Math.max(1, numberOfHeads));
        float rmsNormEps = gguf.getValueOrDefault(float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);

        // Per-layer type: MoE when feed_forward_length>0; attention when head_count_kv>0; else SSM.
        int[] headCountKv = gguf.getValue(int[].class, arch + ".attention.head_count_kv");
        int[] feedForward = gguf.getValue(int[].class, arch + ".feed_forward_length");
        LayerType[] layerTypes = new LayerType[numberOfLayers];
        int kvHeads = 0;
        for (int i = 0; i < numberOfLayers; i++) {
            if (feedForward[i] > 0) layerTypes[i] = LayerType.MOE;
            else if (headCountKv[i] > 0) { layerTypes[i] = LayerType.ATTENTION; kvHeads = headCountKv[i]; }
            else layerTypes[i] = LayerType.SSM;
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

        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        return new NemotronH(config, tokenizer, loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
        int n = config.numberOfLayers;
        FloatTensor tokenEmbeddings = ModelLoader.loadQuantized(tensors.get("token_embd.weight"));
        FloatTensor outputWeight = tensors.containsKey("output.weight")
                ? ModelLoader.loadQuantized(tensors.get("output.weight")) : tokenEmbeddings;
        F32FloatTensor outputNorm = ModelLoader.toF32Tensor(tensors.get("output_norm.weight"));

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            F32FloatTensor attnNorm = ModelLoader.toF32Tensor(tensors.get(p + "attn_norm.weight"));
            AttentionWeights att = null;
            SsmWeights ssm = null;
            MoeFfnWeights moe = null;
            switch (config.layerTypes[i]) {
                case ATTENTION -> att = new AttentionWeights(
                        ModelLoader.loadQuantized(tensors.get(p + "attn_q.weight")),
                        ModelLoader.loadQuantized(tensors.get(p + "attn_k.weight")),
                        ModelLoader.loadQuantized(tensors.get(p + "attn_v.weight")),
                        ModelLoader.loadQuantized(tensors.get(p + "attn_output.weight")));
                case SSM -> ssm = new SsmWeights(
                        ModelLoader.loadQuantized(tensors.get(p + "ssm_in.weight")),
                        ModelLoader.toF32Tensor(tensors.get(p + "ssm_conv1d.weight")),
                        f32OrNull(tensors, p + "ssm_conv1d.bias"),
                        ModelLoader.toF32Tensor(tensors.get(p + "ssm_a")),
                        ModelLoader.toF32Tensor(tensors.get(p + "ssm_d")),
                        ModelLoader.toF32Tensor(tensors.get(p + "ssm_dt.bias")),
                        ModelLoader.toF32Tensor(tensors.get(p + "ssm_norm.weight")),
                        ModelLoader.loadQuantized(tensors.get(p + "ssm_out.weight")));
                case MOE -> moe = new MoeFfnWeights(
                        ModelLoader.loadQuantized(tensors.get(p + "ffn_gate_inp.weight")),
                        f32OrNull(tensors, p + "exp_probs_b.bias"),
                        ModelLoader.loadQuantized(tensors.get(p + "ffn_up_exps.weight")),
                        ModelLoader.loadQuantized(tensors.get(p + "ffn_down_exps.weight")),
                        quantOrNull(tensors, p + "ffn_up_shexp.weight"),
                        quantOrNull(tensors, p + "ffn_down_shexp.weight"));
            }
            layers[i] = new LayerWeights(attnNorm, att, ssm, moe);
        }
        return new Weights(tokenEmbeddings, outputNorm, outputWeight, layers);
    }

    private static F32FloatTensor f32OrNull(Map<String, GGMLTensorEntry> t, String name) {
        GGMLTensorEntry e = t.get(name);
        return e == null ? null : ModelLoader.toF32Tensor(e);
    }

    private static FloatTensor quantOrNull(Map<String, GGMLTensorEntry> t, String name) {
        GGMLTensorEntry e = t.get(name);
        return e == null ? null : ModelLoader.loadQuantized(e);
    }
}
