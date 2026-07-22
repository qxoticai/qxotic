// Qwen3.5 ("qwen35" / "qwen35moe") against the com.qxotic.jinfer.models model API: a faithful port
// of the
// production com.qxotic.jinfer.Qwen35 forward. Qwen3.5 is a HYBRID gated-delta-net
// (linear-attention)
// + periodic full-attention transformer, dense or MoE: layers are SSM (gated delta-net) by default,
// every full_attention_interval-th layer is full softmax attention with QK-norm + a query/output
// gate.
//
// This port batches prompt prefill: dense FFN via ffnForwardBatch, MoE via moeForwardBatch (CSR
// gather-by-expert GEMMs), token-exact vs the single-token reference forward (behind
// -Djinfer.singleTokenPrefill). Only the gated delta-net recurrence + conv ring stay sequential
// over
// the chunk's rows, carried forward in the State exactly as in a streaming decode. This
// keeps the port to public jinfer kernels only (matmul / flashDecode / gemm-with-offset /
// Activations.siluMultiply / RoPE.precomputeFreqsCis) and the shared scalar helpers
// (Activations.sigmoid/silu/softplus) + RoPE.applyInterleaved. Text-only -> implements only
// LanguageModel.
package com.qxotic.jinfer.models.qwen35;

import static com.qxotic.jinfer.Activations.sigmoid;
import static com.qxotic.jinfer.Activations.silu;
import static com.qxotic.jinfer.Activations.softplus;
import static com.qxotic.jinfer.Norms.rmsnorm;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.*;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class Qwen35
        implements LanguageModel<Qwen35.Configuration, Qwen35.Weights, Qwen35.State> {

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final String chatTemplateSource;
    private final byte[] modelSeed;
    private final Weights weights;

    Qwen35(
            Configuration configuration,
            Tokenizer tokenizer,
            String chatTemplateSource,
            byte[] modelSeed,
            Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.chatTemplateSource = chatTemplateSource;
        this.modelSeed = modelSeed;
        this.weights = weights;
    }

    // === com.qxotic.jinfer model API seam ===

    @Override
    public Configuration config() {
        return configuration;
    }

    @Override
    public Weights weights() {
        return weights;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public State newState(int contextCapacity, int batchCapacity) {
        State state = new State(configuration, contextCapacity, batchCapacity);
        return state;
    }

    @Override
    public void ingest(State s, Batch batch) {
        int n = batch.count();
        if (n > s.batchCapacity)
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + s.batchCapacity);
        int from = s.position();
        if (from + n > s.contextCapacity) {
            throw new IllegalArgumentException(
                    "ingest of "
                            + n
                            + " at position "
                            + from
                            + " exceeds contextCapacity "
                            + s.contextCapacity);
        }
        switch (batch.input()) {
            case Batch.Input.Tokens t -> {
                int[] ids = t.ids();
                // Prefill batches the projection GEMMs over the chunk (dense FFN via
                // ffnForwardBatch, MoE
                // via moeForwardBatch); only the delta-net recurrence + conv ring stay sequential
                // over rows.
                if (n == 1)
                    Parallel.onDecodePool(
                            () -> {
                                forward(s, ids, 0, from, 1);
                                return null;
                            });
                else forward(s, ids, 0, from, n);
            }
            case Batch.Input.Sequences seq ->
                    throw new UnsupportedOperationException(
                            "Qwen3.5 is generative: packed sequences (batched embedding) not"
                                    + " supported");
            case Batch.Input.Embeddings e ->
                    throw new UnsupportedOperationException(
                            "Qwen3.5 is text-only: embedding input is not supported");
        }
        s.advance(
                n,
                Batch.Outputs
                        .LAST); // streamed MoE keeps only the last row's residual (LAST semantics)
    }

    @Override
    public FloatTensor logits(State s, int output) {
        if (output != 0)
            throw new UnsupportedOperationException(
                    "Qwen3.5 port keeps only the last row (LAST); ALL outputs unsupported");
        int dim = configuration.embeddingLength;
        return Parallel.onDecodePool(
                () -> {
                    rmsnorm(
                            s.xb,
                            0,
                            s.x,
                            s.lastRowOffset,
                            weights.outputNorm,
                            dim,
                            configuration.rmsNormEps);
                    weights.outputWeight.matmul(s.xb, s.logits, configuration.vocabularySize, dim);
                    return s.logits;
                });
    }

    private com.qxotic.jinfer.chat.TurnTemplate turnTemplate; // memoized: stateless, model-lifetime

    /**
     * This model bundled with the three text facts its GGUF carries - what an
     * architecture-dispatching loader hands to a caller that does not know the family.
     */
    public LoadedModel<Qwen35.State> loaded() {
        return new LoadedModel<>(this, tokenizer(), chatTemplateSource, stopTokens(), modelSeed);
    }

    /** The chat-layer binding: token-level facts plus this model's chat framing. */
    public com.qxotic.jinfer.chat.ChatModel<Qwen35.State> chatModel() {
        return new com.qxotic.jinfer.chat.ChatModel<>(
                loaded(), turnTemplate().map(t -> (com.qxotic.jinfer.chat.ChatTemplate) t));
    }

    public java.util.Optional<com.qxotic.jinfer.chat.TurnTemplate> turnTemplate() {
        if (turnTemplate == null) turnTemplate = new Qwen35TurnTemplate(tokenizer());
        return java.util.Optional.of(turnTemplate);
    }

    @Override
    public java.util.Optional<com.qxotic.jinfer.cache.StateCodec<Qwen35.State>> stateCodec() {
        // The gated-delta-net S matrices are a LARGE true recurrence (MBs per SSM layer) - neither
        // per-position rows nor a small residue, so this family offers no block caching; live
        // sessions (SessionPool) still give append-only reuse.
        return java.util.Optional.empty();
    }

    /** The turn-delimiter / eos ids that terminate generation (convenience for callers/tests). */
    public Set<Integer> stopTokens() {
        Set<Integer> stops = new HashSet<>();
        for (String name : new String[] {"<|im_end|>", "<|endoftext|>"}) {
            SpecialTokens.find(tokenizer, name).ifPresent(stops::add);
        }
        return stops;
    }

    // === Forward (single-token reference; ingest streams the prompt one token at a time) ===

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
            F32FloatTensor attNormW = w.attnNorm[l], postW = w.postAttentionNorm[l];
            Parallel.forRows(
                    seqLen,
                    s ->
                            rmsnorm(
                                    state.xb,
                                    (long) s * fDim,
                                    state.x,
                                    (long) s * fDim,
                                    attNormW,
                                    fDim,
                                    eps));

            if (config.isFullAttention[l]) {
                if (seqLen == 1) attentionForward(state, l, startPos);
                else attentionForwardBatch(state, l, startPos, seqLen);
            } else {
                if (seqLen == 1) ssmForward(state, l);
                else ssmForwardBatch(state, l, seqLen);
            }

            // sublayer residual, then post-attention norm acts as the pre-FFN norm
            state.xb.addInPlace(0, state.x, 0, seqLen * dim);
            state.xb.copyTo(0, state.x, 0, seqLen * dim);
            Parallel.forRows(
                    seqLen,
                    s ->
                            rmsnorm(
                                    state.xb,
                                    (long) s * fDim,
                                    state.xb,
                                    (long) s * fDim,
                                    postW,
                                    fDim,
                                    eps));

            if (config.isMoE()) {
                if (seqLen == 1) moeForward(state, l);
                else moeForwardBatch(state, l, seqLen);
            } else {
                if (seqLen == 1) ffnForward(state, l);
                else ffnForwardBatch(state, l, seqLen);
            }
            state.x.addInPlace(0, state.xb, 0, seqLen * dim);
            if (Trace.ENABLED) Trace.sum("l_out-" + l, state.x, seqLen * dim);
        }
        state.lastRowOffset = (seqLen - 1) * dim;
    }

    // === Batched prefill cores: projection GEMMs over the chunk; the delta-net recurrence + conv
    // stay
    //     sequential over rows (state carries forward). Token-exact vs the single-token cores. ===

    /**
     * Batched full attention: GEMM Q/K/V, per-row QK-norm + RoPE, KV to cache, causal flash
     * attention, sigmoid(gate)-gated output projection. Mirrors {@link #attentionForward}.
     */
    private void attentionForwardBatch(State state, int layer, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength, headSize = config.headSize, heads = config.numberOfHeads;
        int kvHeads = config.numberOfKeyValueHeads,
                kvDim = config.kvDim(),
                queryDim = config.queryDim();
        int kvMul = heads / kvHeads;
        float eps = config.rmsNormEps;

        w.attnQ[layer].gemm(state.xb, dim, state.q, 2 * queryDim, seqLen, 2 * queryDim, dim);
        w.attnK[layer].gemm(state.xb, dim, state.k, kvDim, seqLen, kvDim, dim);
        w.attnV[layer].gemm(state.xb, dim, state.v, kvDim, seqLen, kvDim, dim);

        int fHeadSz = headSize,
                fHeads = heads,
                fKvHeads = kvHeads,
                fQDim = queryDim,
                fKvDim = kvDim,
                fStart = startPos;
        F32FloatTensor qNormW = w.attnQNorm[layer], kNormW = w.attnKNorm[layer];
        float[] gateArr = state.attnGateArr;
        Parallel.forRows(
                seqLen,
                s -> {
                    int qBase = s * 2 * fQDim, qDst = s * fQDim;
                    for (int h = 0; h < fHeads; h++) {
                        int base = qBase + h * 2 * fHeadSz;
                        for (int d = 0; d < fHeadSz; d++) {
                            gateArr[qDst + h * fHeadSz + d] = state.q.getFloat(base + fHeadSz + d);
                            state.attnQ.setFloat(
                                    qDst + h * fHeadSz + d, state.q.getFloat(base + d));
                        }
                    }
                    for (int h = 0; h < fHeads; h++)
                        rmsnorm(
                                state.attnQ,
                                qDst + h * fHeadSz,
                                state.attnQ,
                                qDst + h * fHeadSz,
                                qNormW,
                                fHeadSz,
                                eps);
                    int kBase = s * fKvDim;
                    for (int h = 0; h < fKvHeads; h++)
                        rmsnorm(
                                state.k,
                                kBase + h * fHeadSz,
                                state.k,
                                kBase + h * fHeadSz,
                                kNormW,
                                fHeadSz,
                                eps);
                    if (w.ropeHalf > 0) {
                        for (int h = 0; h < fHeads; h++)
                            RoPE.applyInterleaved(
                                    state.attnQ,
                                    qDst + h * fHeadSz,
                                    fStart + s,
                                    w.ropeCr,
                                    w.ropeCi,
                                    w.ropeHalf);
                        for (int h = 0; h < fKvHeads; h++)
                            RoPE.applyInterleaved(
                                    state.k,
                                    kBase + h * fHeadSz,
                                    fStart + s,
                                    w.ropeCr,
                                    w.ropeCi,
                                    w.ropeHalf);
                    }
                });

        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        for (int s = 0; s < seqLen; s++) {
            state.k.copyTo((long) s * kvDim, keyCache, (long) (startPos + s) * kvDim, kvDim);
            state.v.copyTo((long) s * kvDim, valueCache, (long) (startPos + s) * kvDim, kvDim);
        }
        FlashAttention.causalPrefill(
                (F32FloatTensor) state.attnQ,
                (F32FloatTensor) state.attnOut,
                keyCache,
                valueCache,
                heads,
                startPos,
                seqLen,
                headSize,
                kvDim,
                queryDim,
                kvMul);

        int total = seqLen * queryDim;
        for (int i = 0; i < total; i++)
            state.attnOut.setFloat(i, state.attnOut.getFloat(i) * sigmoid(gateArr[i]));
        w.attnOutput[layer].gemm(state.attnOut, queryDim, state.xb, dim, seqLen, dim, queryDim);
    }

    /**
     * Batched gated delta-net: the QKV/gate/alpha/beta/out projections become GEMMs over the chunk;
     * the conv ring + per-head recurrence run sequentially over rows (state carries). Mirrors
     * {@link #ssmForward}.
     */
    private void ssmForwardBatch(State state, int layer, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength,
                dInner = config.ssmInnerSize,
                nGroup = config.ssmGroupCount;
        int dtRank = config.ssmTimeStepRank,
                dState = config.ssmStateSize,
                convKernel = config.ssmConvKernel;
        int headVDim = config.headVDim(), convChannels = config.convChannels();
        int kOff = dState * nGroup, vOff = 2 * dState * nGroup;
        float eps = config.rmsNormEps, scale = (float) (1.0 / Math.sqrt(headVDim));

        // batched projections
        w.attnQkv[layer].gemm(state.xb, dim, state.ssmQkv, convChannels, seqLen, convChannels, dim);
        w.attnGate[layer].gemm(state.xb, dim, state.gateProj, dInner, seqLen, dInner, dim);
        w.ssmAlpha[layer].gemm(state.xb, dim, state.alphaProj, dtRank, seqLen, dtRank, dim);
        w.ssmBeta[layer].gemm(state.xb, dim, state.betaProj, dtRank, seqLen, dtRank, dim);

        FloatTensor convState = state.ssmConvState[layer], qkv = state.ssmQkv;
        F32FloatTensor convWeight = w.ssmConv1d[layer];
        float[] convOut = state.ssmConvOut, qGroup = state.ssmQGroup, kGroup = state.ssmKGroup;
        float[] qArr = state.ssmQ, kArr = state.ssmK, vArr = state.ssmV;
        float[] gate = state.ssmGate,
                beta = state.ssmBeta,
                output = state.ssmOutput,
                sk = state.ssmSk,
                dd = state.ssmD;
        float[] S = state.ssmState[layer];
        int fK = convKernel, fCC = convChannels, fHist = convKernel - 1, fSeq = seqLen;
        int fHV = headVDim, fNG = nGroup, fDt = dtRank, fKOff = kOff, fVOff = vOff;
        float fScale = scale, fEps = eps;

        // batched causal conv (parallel over channels; each reads ring history + chunk rows), then
        // roll ring
        Parallel.parallelFor(
                0,
                convChannels,
                c -> {
                    int wOff = c * fK;
                    for (int s = 0; s < fSeq; s++) {
                        float sum = 0;
                        for (int k = 0; k < fK; k++) {
                            int pos = s - fHist + k;
                            float in =
                                    pos < 0
                                            ? convState.getFloat((pos + fHist) * fCC + c)
                                            : qkv.getFloat(pos * fCC + c);
                            sum += convWeight.getFloat(wOff + k) * in;
                        }
                        convOut[s * fCC + c] = silu(sum);
                    }
                });
        Parallel.parallelFor(
                0,
                convChannels,
                c -> {
                    for (int k = 0; k < fHist; k++) {
                        int pos = fSeq - fHist + k;
                        float v =
                                pos < 0
                                        ? convState.getFloat((pos + fHist) * fCC + c)
                                        : qkv.getFloat(pos * fCC + c);
                        convState.setFloat(k * fCC + c, v);
                    }
                });

        // batched per-group L2-norm of Q,K (Q folds 1/sqrt(HV)) + tile nGroup -> dtRank (parallel
        // over rows)
        Parallel.forRows(
                seqLen,
                s -> {
                    int cBase = s * fCC, gBase = s * fNG * fHV, dBase = s * fDt * fHV;
                    for (int h = 0; h < fNG; h++) {
                        float qNormSq = 0, kNormSq = 0;
                        int hOff = h * fHV;
                        for (int d = 0; d < fHV; d++) {
                            float qv = convOut[cBase + hOff + d],
                                    kv = convOut[cBase + fKOff + hOff + d];
                            qNormSq += qv * qv;
                            kNormSq += kv * kv;
                        }
                        float qInv = (float) (1.0 / Math.sqrt(qNormSq + fEps)) * fScale,
                                kInv = (float) (1.0 / Math.sqrt(kNormSq + fEps));
                        for (int d = 0; d < fHV; d++) {
                            qGroup[gBase + hOff + d] = convOut[cBase + hOff + d] * qInv;
                            kGroup[gBase + hOff + d] = convOut[cBase + fKOff + hOff + d] * kInv;
                        }
                    }
                    for (int h = 0; h < fDt; h++) {
                        int dstOff = dBase + h * fHV,
                                srcOff = gBase + (h % fNG) * fHV,
                                vSrc = cBase + fVOff + h * fHV;
                        for (int d = 0; d < fHV; d++) {
                            qArr[dstOff + d] = qGroup[srcOff + d];
                            kArr[dstOff + d] = kGroup[srcOff + d];
                            vArr[dstOff + d] = convOut[vSrc + d];
                        }
                    }
                });

        // batched gate/beta: gate = softplus(alpha@x + dt_bias)*A ; beta = sigmoid(beta@x)
        for (int i = 0; i < seqLen * fDt; i++)
            gate[i] =
                    softplus(state.alphaProj.getFloat(i) + w.ssmDtBias[layer].getFloat(i % fDt))
                            * w.ssmA[layer].getFloat(i % fDt);
        for (int i = 0; i < seqLen * fDt; i++) beta[i] = sigmoid(state.betaProj.getFloat(i));

        // delta-net recurrence: ONE parallelFor over heads; each head runs all rows sequentially
        // (state carries)
        Parallel.parallelFor(
                0,
                dtRank,
                h -> {
                    int stateBase = h * fHV * fHV, skOff = h * fHV;
                    for (int s = 0; s < fSeq; s++) {
                        int base = s * fDt * fHV + h * fHV, gOff = s * fDt + h;
                        float expGate = (float) Math.exp(gate[gOff]), betaH = beta[gOff];
                        VectorMath.scale(S, stateBase, expGate, fHV * fHV);
                        for (int j = 0; j < fHV; j++)
                            sk[skOff + j] = VectorMath.dot(S, stateBase + j * fHV, kArr, base, fHV);
                        for (int i = 0; i < fHV; i++)
                            dd[skOff + i] = (vArr[base + i] - sk[skOff + i]) * betaH;
                        for (int j = 0; j < fHV; j++)
                            VectorMath.axpy(S, stateBase + j * fHV, dd[skOff + j], kArr, base, fHV);
                        for (int j = 0; j < fHV; j++)
                            output[base + j] =
                                    VectorMath.dot(S, stateBase + j * fHV, qArr, base, fHV);
                    }
                });

        // batched SiLU(z)-gated RMSNorm -> ssmTmp (z = gateProj), then out projection
        F32FloatTensor ssmNormW = w.ssmNorm[layer];
        FloatTensor gateProj = state.gateProj;
        int fDInner = dInner;
        Parallel.forRows(
                seqLen,
                s -> {
                    int dBase = s * fDt * fHV, zBase = s * fDInner;
                    for (int h = 0; h < fDt; h++) {
                        int headOff = dBase + h * fHV, oOff = zBase + h * fHV;
                        float ss = 0;
                        for (int d = 0; d < fHV; d++) {
                            float val = output[headOff + d];
                            ss += val * val;
                        }
                        float invRms = (float) (1.0 / Math.sqrt(ss / fHV + fEps));
                        for (int d = 0; d < fHV; d++) {
                            float normed = output[headOff + d] * invRms * ssmNormW.getFloat(d);
                            state.ssmTmp.setFloat(
                                    oOff + d, normed * silu(gateProj.getFloat(oOff + d)));
                        }
                    }
                });
        w.ssmOut[layer].gemm(state.ssmTmp, dInner, state.xb, dim, seqLen, dim, dInner);
    }

    /**
     * Batched dense SwiGLU FFN: gate/up/down become GEMMs over the chunk. Mirrors {@link
     * #ffnForward}.
     */
    private void ffnForwardBatch(State state, int layer, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength, hiddenDim = config.hiddenDim;
        w.ffnGate[layer].gemm(state.xb, dim, state.ffnGate, hiddenDim, seqLen, hiddenDim, dim);
        w.ffnUp[layer].gemm(state.xb, dim, state.ffnUp, hiddenDim, seqLen, hiddenDim, dim);
        Activations.siluMultiply(state.ffnGate, 0, state.ffnUp, 0, seqLen * hiddenDim);
        w.ffnDown[layer].gemm(state.ffnGate, hiddenDim, state.xb, dim, seqLen, dim, hiddenDim);
    }

    /**
     * Full softmax attention with QK-norm, fused query/output gate (attn_q -> [q | gate]), GQA and
     * RoPE; output gated by sigmoid(gate).
     */
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
            rmsnorm(
                    state.q,
                    h * headSize,
                    state.q,
                    h * headSize,
                    w.attnQNorm[layer],
                    headSize,
                    eps);
        }
        w.attnK[layer].matmul(state.xb, state.k, kvDim, dim);
        w.attnV[layer].matmul(state.xb, state.v, kvDim, dim);
        for (int h = 0; h < kvHeads; h++) {
            rmsnorm(
                    state.k,
                    h * headSize,
                    state.k,
                    h * headSize,
                    w.attnKNorm[layer],
                    headSize,
                    eps);
        }
        if (w.ropeHalf > 0) {
            for (int h = 0; h < heads; h++)
                RoPE.applyInterleaved(
                        state.q, h * headSize, position, w.ropeCr, w.ropeCi, w.ropeHalf);
            for (int h = 0; h < kvHeads; h++)
                RoPE.applyInterleaved(
                        state.k, h * headSize, position, w.ropeCr, w.ropeCi, w.ropeHalf);
        }
        state.k.copyTo(0, state.keyCache[layer], position * kvDim, kvDim);
        state.v.copyTo(0, state.valueCache[layer], position * kvDim, kvDim);

        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        float attScale = 1.0f / (float) Math.sqrt(headSize);
        FlashAttention.flashDecode(
                (F32FloatTensor) state.q,
                (F32FloatTensor) state.xb2,
                keyCache,
                valueCache,
                null,
                null,
                heads,
                position,
                0,
                headSize,
                kvDim,
                kvMul,
                attScale,
                0,
                null,
                state.decodeScratch);

        for (int i = 0; i < queryDim; i++) {
            state.xb2.setFloat(i, state.xb2.getFloat(i) * sigmoid(gateArr[i]));
        }
        w.attnOutput[layer].matmul(state.xb2, state.xb, dim, queryDim);
    }

    /**
     * Gated delta-net (linear-attention) layer: depthwise causal conv -> SiLU -> per-group L2-norm
     * of Q/K -> tile to value heads -> delta-net recurrence over a [headVDim,headVDim] state ->
     * SiLU(z)-gated RMSNorm -> output projection. (Production single-token reference recurrence.)
     */
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
        Parallel.parallelFor(
                0,
                convChannels,
                c -> {
                    float sum = 0;
                    int wOff = c * convKernel;
                    for (int k = 0; k < convKernel - 1; k++) {
                        sum +=
                                convWeight.getFloat(wOff + k)
                                        * convState.getFloat(k * convChannels + c);
                    }
                    sum += convWeight.getFloat(wOff + (convKernel - 1)) * qkv.getFloat(c);
                    convOut[c] = silu(sum);
                });
        // update conv ring (per channel: shift left, append current qkv as newest)
        Parallel.parallelFor(
                0,
                convChannels,
                c -> {
                    for (int k = 0; k < convKernel - 2; k++) {
                        convState.setFloat(
                                k * convChannels + c,
                                convState.getFloat((k + 1) * convChannels + c));
                    }
                    convState.setFloat((convKernel - 2) * convChannels + c, qkv.getFloat(c));
                });

        // 4. split + per-group L2-norm of Q,K (Q folds in 1/sqrt(headVDim)); 5. tile nGroup ->
        // dtRank
        float scale = (float) (1.0 / Math.sqrt(headVDim));
        float[] qGroup = state.ssmQGroup, kGroup = state.ssmKGroup;
        Parallel.parallelFor(
                0,
                nGroup,
                h -> {
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
        Parallel.parallelFor(
                0,
                dtRank,
                h -> {
                    int dstOff = h * headVDim,
                            srcOff = (h % nGroup) * headVDim,
                            vSrc = vOff + h * headVDim;
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
            gate[h] =
                    softplus(state.ssmTmp.getFloat(h) + w.ssmDtBias[layer].getFloat(h))
                            * w.ssmA[layer].getFloat(h);
        }
        w.ssmBeta[layer].matmul(state.xb, state.ssmTmp, dtRank, dim);
        float[] beta = state.ssmBeta;
        for (int h = 0; h < dtRank; h++) {
            beta[h] = sigmoid(state.ssmTmp.getFloat(h));
        }

        // 7. delta-net recurrence per head; state element (i,j,h) at h*HV^2 + j*HV + i
        float[] output = state.ssmOutput;
        float[] S = state.ssmState[layer];
        float[] sk = state.ssmSk, d = state.ssmD;
        Parallel.parallelFor(
                0,
                dtRank,
                h -> {
                    float expGate = (float) Math.exp(gate[h]);
                    float betaH = beta[h];
                    int stateBase = h * headVDim * headVDim;
                    int headOff = h * headVDim;
                    for (int idx = stateBase; idx < stateBase + headVDim * headVDim; idx++)
                        S[idx] *= expGate;
                    for (int j = 0; j < headVDim; j++) {
                        float sum = 0;
                        int row = stateBase + j * headVDim;
                        for (int i = 0; i < headVDim; i++) sum += S[row + i] * kArr[headOff + i];
                        sk[headOff + j] = sum;
                    }
                    for (int i = 0; i < headVDim; i++)
                        d[headOff + i] = (vArr[headOff + i] - sk[headOff + i]) * betaH;
                    for (int j = 0; j < headVDim; j++) {
                        float dj = d[headOff + j];
                        int row = stateBase + j * headVDim;
                        for (int i = 0; i < headVDim; i++) S[row + i] += dj * kArr[headOff + i];
                    }
                    for (int j = 0; j < headVDim; j++) {
                        float sum = 0;
                        int row = stateBase + j * headVDim;
                        for (int i = 0; i < headVDim; i++) sum += S[row + i] * qArr[headOff + i];
                        output[headOff + j] = sum;
                    }
                });

        // 8. SiLU(z)-gated RMSNorm per head, 9. output projection
        Parallel.parallelFor(
                0,
                dtRank,
                h -> {
                    int headOff = h * headVDim;
                    float ss = 0;
                    for (int dd = 0; dd < headVDim; dd++) {
                        float val = output[headOff + dd];
                        ss += val * val;
                    }
                    float invRms = (float) (1.0 / Math.sqrt(ss / headVDim + eps));
                    for (int dd = 0; dd < headVDim; dd++) {
                        float normed =
                                output[headOff + dd] * invRms * w.ssmNorm[layer].getFloat(dd);
                        state.ssmTmp.setFloat(headOff + dd, normed * silu(z[headOff + dd]));
                    }
                });
        w.ssmOut[layer].matmul(state.ssmTmp, state.xb, dim, dInner);
    }

    /** Dense SwiGLU FFN, inlined here (each port owns its dense FFN). */
    private void ffnForward(State state, int layer) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int hiddenDim = config.hiddenDim;
        w.ffnGate[layer].matmul(state.xb, state.ffnGate, hiddenDim, dim);
        w.ffnUp[layer].matmul(state.xb, state.ffnUp, hiddenDim, dim);
        Activations.siluMultiply(state.ffnGate, 0, state.ffnUp, 0, hiddenDim);
        w.ffnDown[layer].matmul(state.ffnGate, state.xb, dim, hiddenDim);
    }

    /**
     * Insertion-sort top-k of the softmaxed router probs into {@code topE}/{@code topP}, descending
     * and stable so ties keep the lower expert index. Unfilled slots are left as {@code -1}/-INF.
     */
    private static void selectTopK(
            FloatTensor probs, int probBase, int numExperts, int topK, int[] topE, float[] topP) {
        for (int i = 0; i < topK; i++) {
            topE[i] = -1;
            topP[i] = Float.NEGATIVE_INFINITY;
        }
        for (int e = 0; e < numExperts; e++) {
            float prob = probs.getFloat(probBase + e);
            int insertPos = -1;
            for (int k = 0; k < topK; k++) {
                if (prob > topP[k]) {
                    insertPos = k;
                    break;
                }
            }
            if (insertPos >= 0) {
                for (int k = topK - 1; k > insertPos; k--) {
                    topP[k] = topP[k - 1];
                    topE[k] = topE[k - 1];
                }
                topP[insertPos] = prob;
                topE[insertPos] = e;
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
            // gemm(seqLen=1, thisOffset) is the public form of the package-private
            // matmul-with-offset.
            w.moeExpertGate[layer].gemm(
                    state.xb, dim, state.moeGateResult, expertFFN, 1, expertFFN, dim, gateUpOffset);
            w.moeExpertUp[layer].gemm(
                    state.xb, dim, state.moeUpResult, expertFFN, 1, expertFFN, dim, gateUpOffset);
            Activations.siluMultiply(state.moeGateResult, 0, state.moeUpResult, 0, expertFFN);
            w.moeExpertDown[layer].gemm(
                    state.moeGateResult,
                    expertFFN,
                    state.moeExpertOut,
                    dim,
                    1,
                    dim,
                    expertFFN,
                    downOffset);
            moeOutput.saxpyInPlace(0, state.moeExpertOut, 0, dim, weight);
        }

        if (config.expertSharedFeedForwardLength > 0 && w.moeSharedGate[layer] != null) {
            int sharedFFN = config.expertSharedFeedForwardLength;
            w.moeSharedGate[layer].matmul(state.xb, state.moeSharedGate, sharedFFN, dim);
            w.moeSharedUp[layer].matmul(state.xb, state.moeSharedUp, sharedFFN, dim);
            Activations.siluMultiply(state.moeSharedGate, 0, state.moeSharedUp, 0, sharedFFN);
            w.moeSharedDown[layer].matmul(state.moeSharedGate, state.moeSharedOut, dim, sharedFFN);
            float sharedScale = 1.0f;
            if (w.moeSharedInputGate[layer] != null) {
                w.moeSharedInputGate[layer].matmul(state.xb, state.moeSharedInputGate, 1, dim);
                sharedScale = sigmoid(state.moeSharedInputGate.getFloat(0));
            }
            moeOutput.saxpyInPlace(0, state.moeSharedOut, 0, dim, sharedScale);
        }

        moeOutput.copyTo(0, state.xb, 0, dim);
    }

    /**
     * Batched top-k MoE + shared expert (prompt prefill): router GEMM, per-row
     * softmax+top-k+renorm, CSR gather-by-expert GEMMs, plus the batched shared expert with its
     * sigmoid input gate. Sums experts + shared*sharedScale into {@code xb}. Token-exact vs the
     * single-token {@link #moeForward}.
     */
    private void moeForwardBatch(State state, int layer, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int expertFFN = config.expertFeedForwardLength;
        int numExperts = config.expertCount;
        int topK = Math.min(config.expertUsedCount, numExperts);
        int gateUpStride = expertFFN * dim;
        int downStride = dim * expertFFN;

        // The pre-FFN normed input is already in state.xb; snapshot it as the per-row
        // router/expert/shared
        // input (the CSR gather scrambles row order).
        state.xb.copyTo(0, state.moeInputB, 0, seqLen * dim);

        // router: softmax over all experts, top-k, renormalize top-k (per row)
        w.moeRouter[layer].gemm(
                state.moeInputB, dim, state.moeRouterB, numExperts, seqLen, numExperts, dim);
        int[] counts = state.moeExpertCounts;
        java.util.Arrays.fill(counts, 0);
        for (int s = 0; s < seqLen; s++) {
            state.moeRouterB.softmaxInPlace(s * numExperts, numExperts);
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

        // CSR grouping + gather + scatter is the shared kernel; the per-expert gated SiLU is here.
        Moe.Routing r = state.moeRouting;
        r.seqLen = seqLen;
        r.topK = topK;
        r.numExperts = numExperts;
        Moe.dispatch(
                r,
                dim,
                state.moeInputB,
                state.moeGather,
                state.moeDownB,
                state.moeOutB,
                null,
                (e, n, gather, out) -> {
                    int gateUpOffset = e * gateUpStride;
                    w.moeExpertGate[layer].gemm(
                            gather,
                            dim,
                            state.moeGateUpB,
                            expertFFN,
                            n,
                            expertFFN,
                            dim,
                            gateUpOffset);
                    w.moeExpertUp[layer].gemm(
                            gather, dim, state.moeUpB, expertFFN, n, expertFFN, dim, gateUpOffset);
                    Parallel.forRows(
                            n,
                            j ->
                                    Activations.siluMultiply(
                                            state.moeGateUpB,
                                            j * expertFFN,
                                            state.moeUpB,
                                            j * expertFFN,
                                            expertFFN));
                    w.moeExpertDown[layer].gemm(
                            state.moeGateUpB,
                            expertFFN,
                            out,
                            dim,
                            n,
                            dim,
                            expertFFN,
                            e * downStride);
                });

        // shared expert (batched) + sigmoid input gate, added per row
        if (config.expertSharedFeedForwardLength > 0 && w.moeSharedGate[layer] != null) {
            int sharedFFN = config.expertSharedFeedForwardLength;
            w.moeSharedGate[layer].gemm(
                    state.moeInputB, dim, state.moeSharedGateB, sharedFFN, seqLen, sharedFFN, dim);
            w.moeSharedUp[layer].gemm(
                    state.moeInputB, dim, state.moeSharedUpB, sharedFFN, seqLen, sharedFFN, dim);
            Parallel.forRows(
                    seqLen,
                    s ->
                            Activations.siluMultiply(
                                    state.moeSharedGateB,
                                    s * sharedFFN,
                                    state.moeSharedUpB,
                                    s * sharedFFN,
                                    sharedFFN));
            w.moeSharedDown[layer].gemm(
                    state.moeSharedGateB,
                    sharedFFN,
                    state.moeSharedOutB,
                    dim,
                    seqLen,
                    dim,
                    sharedFFN);
            if (w.moeSharedInputGate[layer] != null) {
                w.moeSharedInputGate[layer].gemm(
                        state.moeInputB, dim, state.moeSharedInputGateB, 1, seqLen, 1, dim);
                Parallel.forRows(
                        seqLen,
                        s -> {
                            float sharedScale = sigmoid(state.moeSharedInputGateB.getFloat(s));
                            state.moeOutB.saxpyInPlace(
                                    s * dim, state.moeSharedOutB, s * dim, dim, sharedScale);
                        });
            } else {
                state.moeOutB.addInPlace(0, state.moeSharedOutB, 0, seqLen * dim);
            }
        }

        state.moeOutB.copyTo(0, state.xb, 0, seqLen * dim);
    }

    // === Configuration ===

    public static final class Configuration implements Config {
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
        final int hiddenDim;
        final boolean[] isFullAttention;
        final int ssmInnerSize;
        final int ssmGroupCount;
        final int ssmTimeStepRank;
        final int ssmStateSize;
        final int ssmConvKernel;
        final int expertCount;
        final int expertUsedCount;
        final int expertFeedForwardLength;
        final int expertSharedFeedForwardLength;

        Configuration(
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
                boolean[] isFullAttention,
                int ssmInnerSize,
                int ssmGroupCount,
                int ssmTimeStepRank,
                int ssmStateSize,
                int ssmConvKernel,
                int expertCount,
                int expertUsedCount,
                int expertFeedForwardLength,
                int expertSharedFeedForwardLength) {
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

        @Override
        public int vocabularySize() {
            return vocabularySize;
        }

        @Override
        public int contextLength() {
            return contextLength;
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

    // === Weights (production parallel-array layout, kept verbatim: the gated-delta-net math
    // indexes
    //     w.xxx[layer] across intricate kernels, so re-grouping into LayerWeights[] would risk the
    //     token-exactness this port is verified against; a follow-up refactor can use the compare
    // harness). ===

    public static final class Weights {
        final FloatTensor tokenEmbeddingTable;
        final F32FloatTensor outputNorm;
        final FloatTensor outputWeight;
        final F32FloatTensor[] attnNorm, postAttentionNorm;
        final FloatTensor[] attnQ, attnK, attnV, attnOutput;
        final F32FloatTensor[] attnQNorm, attnKNorm;
        final FloatTensor[] attnQkv, attnGate, ssmAlpha, ssmBeta, ssmOut;
        final F32FloatTensor[] ssmConv1d, ssmA, ssmDtBias, ssmNorm;
        final FloatTensor[] ffnGate, ffnUp, ffnDown;
        final FloatTensor[] moeRouter, moeExpertGate, moeExpertUp, moeExpertDown;
        final FloatTensor[] moeSharedGate, moeSharedUp, moeSharedDown, moeSharedInputGate;
        final float[] ropeCr, ropeCi;
        final int ropeHalf;

        Weights(
                FloatTensor tokenEmbeddingTable,
                F32FloatTensor outputNorm,
                FloatTensor outputWeight,
                F32FloatTensor[] attnNorm,
                F32FloatTensor[] postAttentionNorm,
                FloatTensor[] attnQ,
                FloatTensor[] attnK,
                FloatTensor[] attnV,
                FloatTensor[] attnOutput,
                F32FloatTensor[] attnQNorm,
                F32FloatTensor[] attnKNorm,
                FloatTensor[] attnQkv,
                FloatTensor[] attnGate,
                FloatTensor[] ssmAlpha,
                FloatTensor[] ssmBeta,
                FloatTensor[] ssmOut,
                F32FloatTensor[] ssmConv1d,
                F32FloatTensor[] ssmA,
                F32FloatTensor[] ssmDtBias,
                F32FloatTensor[] ssmNorm,
                FloatTensor[] ffnGate,
                FloatTensor[] ffnUp,
                FloatTensor[] ffnDown,
                FloatTensor[] moeRouter,
                FloatTensor[] moeExpertGate,
                FloatTensor[] moeExpertUp,
                FloatTensor[] moeExpertDown,
                FloatTensor[] moeSharedGate,
                FloatTensor[] moeSharedUp,
                FloatTensor[] moeSharedDown,
                FloatTensor[] moeSharedInputGate,
                float[] ropeCr,
                float[] ropeCi,
                int ropeHalf) {
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

    // === State (single-token scratch; KV cache + conv ring + delta-net state carry across ingests)
    // ===

    public static final class State extends com.qxotic.jinfer.BaseState {
        final int contextCapacity, batchCapacity;
        int lastRowOffset;

        final FloatTensor x, xb, xb2, q, k, v, logits, ffnUp, ffnGate, ssmQkv, ssmTmp;
        final FloatTensor attnQ,
                attnOut,
                gateProj,
                alphaProj,
                betaProj; // batched-prefill scratch (chunk rows)
        final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch();
        final float[] attnGateArr,
                ssmZ,
                ssmConvOut,
                ssmQ,
                ssmK,
                ssmV,
                ssmQGroup,
                ssmKGroup,
                ssmGate,
                ssmBeta,
                ssmOutput,
                ssmSk,
                ssmD;
        final FloatTensor[] keyCache, valueCache, ssmConvState;
        final float[][] ssmState;
        final FloatTensor moeRouterLogits,
                moeOutput,
                moeExpertOut,
                moeGateResult,
                moeUpResult,
                moeSharedGate,
                moeSharedUp,
                moeSharedOut,
                moeSharedInputGate;
        final int[] moeTopExperts;
        final float[] moeTopWeights;
        // batched grouped-MoE prefill scratch (CSR gather-by-expert; sized to batchCapacity rows)
        final FloatTensor moeInputB,
                moeRouterB,
                moeOutB,
                moeGather,
                moeGateUpB,
                moeUpB,
                moeDownB,
                moeSharedGateB,
                moeSharedUpB,
                moeSharedOutB,
                moeSharedInputGateB;
        final int[] moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeRowTopE;
        final float[] moeProbByExpert, moeRowTopP;
        final Moe.Routing moeRouting;

        State(Configuration config, int contextCapacity, int batchCapacity) {
            if (contextCapacity > config.contextLength) {
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds model contextLength "
                                + config.contextLength);
            }
            this.contextCapacity = contextCapacity;
            this.batchCapacity = Math.max(1, batchCapacity);
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

            int B = this.batchCapacity; // prompt chunks process B rows at once (batched GEMMs)
            this.x = FloatTensor.allocateF32(B * dim);
            this.xb = FloatTensor.allocateF32(B * dim);
            this.xb2 = FloatTensor.allocateF32(xb2Size); // single-token decode only
            this.q = FloatTensor.allocateF32(B * 2 * queryDim);
            this.k = FloatTensor.allocateF32(B * kvDim);
            this.v = FloatTensor.allocateF32(B * kvDim);
            this.logits = FloatTensor.allocateF32(config.vocabularySize);
            this.ffnUp = hiddenDim > 0 ? FloatTensor.allocateF32(B * hiddenDim) : null;
            this.ffnGate = hiddenDim > 0 ? FloatTensor.allocateF32(B * hiddenDim) : null;
            this.attnGateArr = new float[B * queryDim];
            this.attnQ = FloatTensor.allocateF32(B * queryDim);
            this.attnOut = FloatTensor.allocateF32(B * queryDim);

            this.ssmQkv = FloatTensor.allocateF32(B * convChannels);
            this.ssmTmp = FloatTensor.allocateF32(B * dInner);
            this.gateProj = FloatTensor.allocateF32(B * dInner);
            this.alphaProj = FloatTensor.allocateF32(B * dtRank);
            this.betaProj = FloatTensor.allocateF32(B * dtRank);
            this.ssmZ = new float[B * dInner]; // all chunk rows (batched conv/norm/scan)
            this.ssmConvOut = new float[B * convChannels];
            this.ssmQ = new float[B * dtRank * headVDim];
            this.ssmK = new float[B * dtRank * headVDim];
            this.ssmV = new float[B * dtRank * headVDim];
            this.ssmQGroup = new float[B * nGroup * headVDim];
            this.ssmKGroup = new float[B * nGroup * headVDim];
            this.ssmGate = new float[B * dtRank];
            this.ssmBeta = new float[B * dtRank];
            this.ssmOutput = new float[B * dtRank * headVDim];
            this.ssmSk = new float[dtRank * headVDim]; // per-head scratch (reused across rows)
            this.ssmD = new float[dtRank * headVDim];

            if (config.isMoE()) {
                int e = config.expertCount, eff = config.expertFeedForwardLength;
                int sff = Math.max(1, config.expertSharedFeedForwardLength);
                int tk = Math.max(1, config.expertUsedCount);
                this.moeRouterLogits = FloatTensor.allocateF32(e);
                this.moeOutput = FloatTensor.allocateF32(dim);
                this.moeExpertOut = FloatTensor.allocateF32(dim);
                this.moeGateResult = FloatTensor.allocateF32(eff);
                this.moeUpResult = FloatTensor.allocateF32(eff);
                this.moeSharedGate = FloatTensor.allocateF32(sff);
                this.moeSharedUp = FloatTensor.allocateF32(sff);
                this.moeSharedOut = FloatTensor.allocateF32(dim);
                this.moeSharedInputGate = FloatTensor.allocateF32(1);
                this.moeTopExperts = new int[tk];
                this.moeTopWeights = new float[tk];
                int c = this.batchCapacity;
                this.moeInputB = FloatTensor.allocateF32(c * dim);
                this.moeRouterB = FloatTensor.allocateF32(c * e);
                this.moeOutB = FloatTensor.allocateF32(c * dim);
                this.moeGather = FloatTensor.allocateF32(c * dim);
                this.moeGateUpB = FloatTensor.allocateF32(c * eff);
                this.moeUpB = FloatTensor.allocateF32(c * eff);
                this.moeDownB = FloatTensor.allocateF32(c * dim);
                this.moeSharedGateB = FloatTensor.allocateF32(c * sff);
                this.moeSharedUpB = FloatTensor.allocateF32(c * sff);
                this.moeSharedOutB = FloatTensor.allocateF32(c * dim);
                this.moeSharedInputGateB = FloatTensor.allocateF32(c);
                this.moeExpertCounts = new int[e];
                this.moeExpertOffsets = new int[e + 1];
                this.moeCursor = new int[e];
                this.moeRowByExpert = new int[c * tk];
                this.moeRowTopE = new int[c * tk];
                this.moeProbByExpert = new float[c * tk];
                this.moeRowTopP = new float[c * tk];
                this.moeRouting =
                        new Moe.Routing(
                                moeRowTopE,
                                moeRowTopP,
                                moeExpertCounts,
                                moeExpertOffsets,
                                moeCursor,
                                moeRowByExpert,
                                moeProbByExpert);
            } else {
                this.moeRouterLogits =
                        this.moeOutput =
                                this.moeExpertOut = this.moeGateResult = this.moeUpResult = null;
                this.moeSharedGate =
                        this.moeSharedUp = this.moeSharedOut = this.moeSharedInputGate = null;
                this.moeTopExperts = null;
                this.moeTopWeights = null;
                this.moeInputB =
                        this.moeRouterB =
                                this.moeOutB =
                                        this.moeGather = this.moeGateUpB = this.moeUpB = null;
                this.moeDownB =
                        this.moeSharedGateB =
                                this.moeSharedUpB =
                                        this.moeSharedOutB = this.moeSharedInputGateB = null;
                this.moeExpertCounts =
                        this.moeExpertOffsets =
                                this.moeCursor = this.moeRowByExpert = this.moeRowTopE = null;
                this.moeProbByExpert = this.moeRowTopP = null;
                this.moeRouting = null;
            }

            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            this.ssmConvState = new FloatTensor[config.numberOfLayers];
            this.ssmState = new float[config.numberOfLayers][];
            for (int l = 0; l < config.numberOfLayers; l++) {
                if (config.isFullAttention[l]) {
                    keyCache[l] = FloatTensor.allocateF16(contextCapacity * kvDim);
                    valueCache[l] = FloatTensor.allocateF16(contextCapacity * kvDim);
                } else {
                    ssmConvState[l] =
                            FloatTensor.allocateF32((config.ssmConvKernel - 1) * convChannels);
                    ssmState[l] = new float[headVDim * headVDim * dtRank];
                }
            }
        }

        @Override
        public int contextCapacity() {
            return contextCapacity;
        }

        @Override
        public int batchCapacity() {
            return batchCapacity;
        }
    }

    // === Loading ===

    public static Qwen35 loadModel(Path ggufPath, int contextLength) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, contextLength, true);
        }
    }

    public static Qwen35 loadModel(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag)
            throws IOException {
        byte[] seed = com.qxotic.jinfer.cache.PromptCache.modelSeed(fileChannel);
        Tokenizer tokenizer = Tokenizers.fromGGUF(gguf);
        String arch = gguf.getString("general.architecture");

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) {
            contextLength = modelContextLength;
        }
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int numberOfKeyValueHeads = gguf.getValue(int.class, arch + ".attention.head_count_kv");
        int headSize =
                gguf.getValueOrDefault(
                        int.class, arch + ".attention.key_length", embeddingLength / numberOfHeads);
        float rmsNormEps =
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 1000000f);
        int ropeDimensionCount =
                gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);
        int fullAttentionInterval =
                gguf.getValueOrDefault(int.class, arch + ".full_attention_interval", 4);
        int hiddenDim = gguf.getValueOrDefault(int.class, arch + ".feed_forward_length", 0);

        int ssmInnerSize = gguf.getValueOrDefault(int.class, arch + ".ssm.inner_size", 0);
        int ssmGroupCount = gguf.getValueOrDefault(int.class, arch + ".ssm.group_count", 0);
        int ssmTimeStepRank = gguf.getValueOrDefault(int.class, arch + ".ssm.time_step_rank", 0);
        int ssmStateSize = gguf.getValueOrDefault(int.class, arch + ".ssm.state_size", 0);
        int ssmConvKernel = gguf.getValueOrDefault(int.class, arch + ".ssm.conv_kernel", 0);

        int expertCount = gguf.getValueOrDefault(int.class, arch + ".expert_count", 0);
        int expertUsedCount = gguf.getValueOrDefault(int.class, arch + ".expert_used_count", 0);
        int expertFeedForwardLength =
                gguf.getValueOrDefault(int.class, arch + ".expert_feed_forward_length", 0);
        int expertSharedFeedForwardLength =
                gguf.getValueOrDefault(int.class, arch + ".expert_shared_feed_forward_length", 0);

        boolean[] isFullAttention = new boolean[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            isFullAttention[i] = (i + 1) % fullAttentionInterval == 0;
        }

        Configuration config =
                new Configuration(
                        embeddingLength,
                        numberOfLayers,
                        numberOfHeads,
                        numberOfKeyValueHeads,
                        headSize,
                        tokenizer.vocabulary().size(),
                        contextLength,
                        rmsNormEps,
                        ropeTheta,
                        ropeDimensionCount,
                        hiddenDim,
                        isFullAttention,
                        ssmInnerSize,
                        ssmGroupCount,
                        ssmTimeStepRank,
                        ssmStateSize,
                        ssmConvKernel,
                        expertCount,
                        expertUsedCount,
                        expertFeedForwardLength,
                        expertSharedFeedForwardLength);

        if (!loadWeightsFlag) {
            return new Qwen35(config, tokenizer, Tokenizers.chatTemplateSource(gguf), seed, null);
        }
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        return new Qwen35(
                config,
                tokenizer,
                Tokenizers.chatTemplateSource(gguf),
                seed,
                loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
        int n = config.numberOfLayers;
        FloatTensor tokenEmbeddingTable =
                ModelLoader.loadQuantized(tensors.get("token_embd.weight"));
        FloatTensor outputWeight =
                tensors.containsKey("output.weight")
                        ? ModelLoader.loadQuantized(tensors.get("output.weight"))
                        : tokenEmbeddingTable;

        int ropeDim = Math.max(0, Math.min(config.ropeDimensionCount, config.headSize) & ~1);
        Pair<float[], float[]> rope =
                ropeDim > 0
                        ? RoPE.precomputeFreqsCis(config.contextLength, ropeDim, config.ropeTheta)
                        : null;

        return new Weights(
                tokenEmbeddingTable,
                ModelLoader.toF32Tensor(tensors.get("output_norm.weight")),
                outputWeight,
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_norm.weight")),
                ModelLoader.f32Array(
                        n, i -> tensors.get("blk." + i + ".post_attention_norm.weight")),
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
                ModelLoader.quantArray(
                        n,
                        i ->
                                ModelLoader.firstPresent(
                                        tensors,
                                        "blk." + i + ".ffn_gate_inp.weight",
                                        "blk." + i + ".ffn_router.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate_exps.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up_exps.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down_exps.weight")),
                ModelLoader.quantArray(
                        n,
                        i ->
                                ModelLoader.firstPresent(
                                        tensors,
                                        "blk." + i + ".ffn_gate_shexp.weight",
                                        "blk." + i + ".ffn_shared_expert_gate.weight")),
                ModelLoader.quantArray(
                        n,
                        i ->
                                ModelLoader.firstPresent(
                                        tensors,
                                        "blk." + i + ".ffn_up_shexp.weight",
                                        "blk." + i + ".ffn_shared_expert_up.weight")),
                ModelLoader.quantArray(
                        n,
                        i ->
                                ModelLoader.firstPresent(
                                        tensors,
                                        "blk." + i + ".ffn_down_shexp.weight",
                                        "blk." + i + ".ffn_shared_expert_down.weight")),
                ModelLoader.quantArray(
                        n,
                        i ->
                                ModelLoader.firstPresent(
                                        tensors,
                                        "blk." + i + ".ffn_gate_inp_shexp.weight",
                                        "blk." + i + ".ffn_shared_expert_gate_inp.weight")),
                rope != null ? rope.first() : null,
                rope != null ? rope.second() : null,
                ropeDim / 2);
    }
}
