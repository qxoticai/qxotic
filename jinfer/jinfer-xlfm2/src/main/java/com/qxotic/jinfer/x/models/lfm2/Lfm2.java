// LFM2.5 (Liquid Foundation Model 2.5) against the MemoryView boundary: a call-site-by-call-site
// port of jinfer-lfm2's Lfm2 (cycle 1 of the FloatTensor migration). Each layer is EITHER GQA
// attention (kv-heads > 0) OR a gated short-convolution mixer (kv-heads == 0); the FFN is EITHER
// dense SwiGLU OR top-k MoE. Text-only. Weights/state/KV are MemoryView<MemorySegment>; kernels
// are the x statics; gemm/gemv entry shims resolve to x.MatMul.mm here (the old virtuals').
package com.qxotic.jinfer.x.models.lfm2;

import static com.qxotic.jinfer.x.Segments.readFloat;
import static com.qxotic.jinfer.x.Segments.writeFloat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.Activations;
import com.qxotic.jinfer.x.Convert;
import com.qxotic.jinfer.x.FlashAttention;
import com.qxotic.jinfer.x.MatMul;
import com.qxotic.jinfer.x.Norms;
import com.qxotic.jinfer.x.Ops;
import com.qxotic.jinfer.x.Parallel;
import com.qxotic.jinfer.x.RoPE;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.Views.Raw;
import com.qxotic.jinfer.x.boundary.BaseState;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Config;
import com.qxotic.jinfer.x.boundary.EmbeddingModel;
import com.qxotic.jinfer.x.boundary.LanguageModel;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Moe;
import com.qxotic.jinfer.x.kernels.Trace;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

public final class Lfm2
        implements LanguageModel<Lfm2.Configuration, Lfm2.Weights, Lfm2.State>,
                EmbeddingModel<Lfm2.Configuration, Lfm2.Weights, Lfm2.State> {

    /** llama.cpp's pooling_type enum value for CLS - pool the sequence's FIRST row (its BOS). */
    static final int POOLING_CLS = 2;

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    Lfm2(Configuration configuration, Tokenizer tokenizer, Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
    }

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
    public State newState(int contextCapacity, int batchCapacity, Arena arena) {
        return new State(configuration, contextCapacity, batchCapacity, arena);
    }

    @Override
    public void forward(State s, Batch batch) {
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
                if (n == 1)
                    Parallel.onDecodePool(
                            () -> {
                                forward(s, ids, 0, from, n);
                                return null;
                            });
                else forward(s, ids, 0, from, n);
            }
            case Batch.Input.Sequences seq -> {
                if (configuration.causalAttention)
                    throw new UnsupportedOperationException(
                            "this LFM2.5 checkpoint is generative: batched embedding needs the"
                                    + " embedding checkpoint (LFM2.5-Embedding, attention.causal ="
                                    + " false)");
                forwardSegmented(s, seq.tokens().ids(), seq.seqLen(), n);
            }
        }
        s.advance(n, batch.outputs());
    }

    @Override
    public MemoryView<?> head(State s, int output) {
        int dim = configuration.embeddingLength;
        int row = s.lastChunkLen - s.outputCount + output;
        return Parallel.onDecodePool(
                () -> {
                    Norms.rmsnorm(
                            s.xb,
                            0,
                            s.residual,
                            (long) row * dim,
                            weights.finalNorm,
                            dim,
                            configuration.rmsNormEps);
                    gemv(weights.wcls, s.xb, s.logits, configuration.vocabularySize, dim);
                    Activations.softcap(
                            s.logits,
                            0,
                            configuration.vocabularySize,
                            configuration.logitSoftcapping);
                    return s.logits;
                });
    }

    // === Forward ===

    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        // ONCE for the batch: an angle never depends on the layer
        RoPE.fill(
                state.ropeCos,
                state.ropeSin,
                startPos,
                seqLen,
                configuration.headSizeFull / 2,
                weights.rope());
        embedTokens(state, tokens, tokenOffset, seqLen);
        for (int l = 0; l < configuration.numberOfLayers; l++) layer(state, l, startPos, seqLen);
        commitKv(state, startPos, seqLen);
    }

    /** Token-embedding lookup into the residual stream (no scaling, unlike Gemma4). */
    private void embedTokens(State state, int[] tokens, int tokenOffset, int seqLen) {
        Views.checkAlive(weights.tokenEmbeddings, "tokenEmbeddings"); // fail-fast on freed weights
        int dim = configuration.embeddingLength;
        MemoryView<MemorySegment> emb = weights.tokenEmbeddings;
        DataType dt = emb.dataType();
        for (int s = 0; s < seqLen; s++) {
            long srcOff = (long) tokens[tokenOffset + s] * dim;
            long dstOff = (long) s * dim;
            if (dt == DataType.Q8_0) {
                Convert.dequantQ8_0(emb, srcOff, state.residual, dstOff, dim);
            } else if (dt == DataType.FP16) {
                Convert.f16ToF32(emb, srcOff, state.residual, dstOff, dim);
            } else if (dt == DataType.FP32) {
                Convert.copyF32(emb, srcOff, state.residual, dstOff, dim);
            } else {
                throw new UnsupportedOperationException("embedding dtype " + dt);
            }
        }
    }

    /** One block: short-conv mixer OR attention, then the FFN, in place on the residual. */
    private void layer(State state, int l, int startPos, int seqLen) {
        if (configuration.isRecurrentLayer(l)) shortConvMixer(state, l, seqLen);
        else attention(state, l, startPos, seqLen);
        feedForward(state, l, seqLen);
        if (Trace.ENABLED)
            Trace.sum("l_out-" + l, state.residual, seqLen * configuration.embeddingLength);
    }

    // --- short-conv mixer (recurrent layer) ---

    /** Pre-norm -> in-proj (B|C_gate|x) -> causal FIR scan -> out-proj, added to the residual. */
    private void shortConvMixer(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength;
        float eps = configuration.rmsNormEps;
        ShortConvWeights sc = weights.layers[l].shortConv();
        MemoryView<MemorySegment> preNorm =
                weights.layers[l].attnNorm(); // conv layers use attn_norm as the mixer pre-norm
        Parallel.forRows(
                seqLen,
                s ->
                        Norms.rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                preNorm,
                                dim,
                                eps));
        gemm(sc.inProj(), state.xb, dim, state.shortConvTmp, 3 * dim, seqLen, 3 * dim, dim);
        shortConvScan(state, l, seqLen);
        gemm(sc.outProj(), state.xb2, dim, state.shortConvOut, dim, seqLen, dim, dim);
        Ops.addInPlace(state.residual, 0, state.shortConvOut, 0, seqLen * dim);
    }

    /**
     * Causal short-convolution as a dConv-tap FIR over bx = B∘x rows (scalar; ported from the
     * production {@code Llama.shortConvScan}). For each channel: {@code out[s] = C_gate[s] *
     * (Σ_{k<hist} state[k]·kernel[k] + bx[s]·kernel[hist])}, where {@code state} holds the previous
     * {@code hist=dConv-1} bx values; bx is materialized in place over the B block of shortConvTmp
     * and the newest bx rolls into shortConvState.
     */
    private void shortConvScan(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength;
        int dConv = configuration.shortConvLCache, hist = dConv - 1;
        Raw kernel =
                Views.rawF32(
                        weights.layers[l].shortConv().kernel(),
                        "kernel"); // per channel: dConv taps at c*dConv + k
        Raw convState = Views.rawF32(state.shortConvState[l], "shortConvState");
        Raw tmp = Views.rawF32(state.shortConvTmp, "shortConvTmp");
        Raw out = Views.rawF32(state.xb2, "xb2");
        for (int s = 0; s < seqLen; s++) {
            int tmpOff = s * 3 * dim, outOff = s * dim;
            for (int c = 0; c < dim; c++) {
                float b = readFloat(tmp.vseg(), tmp.vbase() + (long) (tmpOff + c) * Float.BYTES);
                float cg =
                        readFloat(
                                tmp.vseg(), tmp.vbase() + (long) (tmpOff + dim + c) * Float.BYTES);
                float xv =
                        readFloat(
                                tmp.vseg(),
                                tmp.vbase() + (long) (tmpOff + 2 * dim + c) * Float.BYTES);
                float bx = b * xv;
                writeFloat(tmp.vseg(), tmp.vbase() + (long) (tmpOff + c) * Float.BYTES, bx);
                int kBase = c * dConv;
                float sum = 0f;
                for (int k = 0; k < hist; k++)
                    sum +=
                            readFloat(
                                            convState.vseg(),
                                            convState.vbase() + ((long) k * dim + c) * Float.BYTES)
                                    * readFloat(
                                            kernel.vseg(),
                                            kernel.vbase() + (long) (kBase + k) * Float.BYTES);
                sum +=
                        bx
                                * readFloat(
                                        kernel.vseg(),
                                        kernel.vbase() + (long) (kBase + dConv - 1) * Float.BYTES);
                writeFloat(out.vseg(), out.vbase() + (long) (outOff + c) * Float.BYTES, cg * sum);
                for (int k = 0; k < hist - 1; k++)
                    writeFloat(
                            convState.vseg(),
                            convState.vbase() + ((long) k * dim + c) * Float.BYTES,
                            readFloat(
                                    convState.vseg(),
                                    convState.vbase() + ((long) (k + 1) * dim + c) * Float.BYTES));
                if (hist > 0)
                    writeFloat(
                            convState.vseg(),
                            convState.vbase() + ((long) (hist - 1) * dim + c) * Float.BYTES,
                            bx);
            }
        }
    }

    // --- attention (GQA) ---

    /**
     * Pre-norm GQA: per-head Q/K RMS-norm + NeoX RoPE (no V-norm), full causal attention with
     * {@code scale = 1/sqrt(headSize)}, output projection, optional post-norm, added to the
     * residual.
     */
    private void attention(State state, int l, int startPos, int seqLen) {
        Configuration config = configuration;
        int headSize = config.headSizeFull;
        int queryDim = config.queryDim(), kvDim = config.kvDim(l);
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeadsPerLayer[l];
        attentionProject(state, l, seqLen);
        MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];
        float scale = 1.0f / (float) Math.sqrt(headSize);
        if (seqLen > 1) {
            FlashAttention.slidingWindowPrefill(
                    state.query,
                    state.xbK,
                    state.keyCache[l],
                    state.valueCache[l],
                    bK,
                    bV,
                    config.numberOfHeads,
                    startPos,
                    seqLen,
                    headSize,
                    kvDim,
                    queryDim,
                    kvDim,
                    kvMul,
                    scale,
                    0,
                    0,
                    null);
        } else {
            FlashAttention.flashDecode(
                    state.query,
                    state.xbK,
                    state.keyCache[l],
                    state.valueCache[l],
                    bK,
                    bV,
                    config.numberOfHeads,
                    startPos,
                    0,
                    headSize,
                    kvDim,
                    kvMul,
                    scale,
                    0,
                    null,
                    state.decodeScratch);
        }

        attentionFinish(state, l, seqLen);
    }

    /**
     * The shared head of both attention paths: pre-norm, Q/K/V projections into {@code
     * query}/{@code batchK}/{@code batchV}, per-head QK RMS-norm + NeoX RoPE (positions are
     * whatever {@code ropeCos}/{@code ropeSin} were filled with - a range causally, per-sequence
     * restarts bidirectionally).
     */
    private void attentionProject(State state, int l, int seqLen) {
        Configuration config = configuration;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;
        int headSize = config.headSizeFull, halfHead = headSize / 2;
        int queryDim = config.queryDim(), kvDim = config.kvDim(l);
        int nKvHeads = config.numberOfKeyValueHeadsPerLayer[l];
        AttentionWeights attn = weights.layers[l].attention();

        MemoryView<MemorySegment> attNormW = weights.layers[l].attnNorm();
        Parallel.forRows(
                seqLen,
                s ->
                        Norms.rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                attNormW,
                                dim,
                                eps));
        gemm(attn.wq(), state.xb, dim, state.query, queryDim, seqLen, queryDim, dim);
        headNormRope(
                state.query,
                queryDim,
                config.numberOfHeads,
                headSize,
                halfHead,
                attn.qNorm(),
                seqLen,
                state.ropeCos,
                state.ropeSin);
        MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];
        gemm(attn.wk(), state.xb, dim, bK, kvDim, seqLen, kvDim, dim);
        if (attn.wv() != null) gemm(attn.wv(), state.xb, dim, bV, kvDim, seqLen, kvDim, dim);
        else Convert.copyF32(bK, 0, bV, 0, (long) seqLen * kvDim);
        headNormRope(
                bK,
                kvDim,
                nKvHeads,
                headSize,
                halfHead,
                attn.kNorm(),
                seqLen,
                state.ropeCos,
                state.ropeSin);
    }

    /** The shared tail: output projection, optional post-norm, added to the residual. */
    private void attentionFinish(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength;
        float eps = configuration.rmsNormEps;
        AttentionWeights attn = weights.layers[l].attention();
        gemm(
                attn.wo(),
                state.xbK,
                configuration.queryDim(),
                state.xb2,
                dim,
                seqLen,
                dim,
                configuration.queryDim());
        MemoryView<MemorySegment> postAttW = weights.layers[l].postAttnNorm();
        if (postAttW != null)
            Parallel.forRows(
                    seqLen,
                    s ->
                            Norms.rmsnorm(
                                    state.xb2,
                                    (long) s * dim,
                                    state.xb2,
                                    (long) s * dim,
                                    postAttW,
                                    dim,
                                    eps));
        Ops.addInPlace(state.residual, 0, state.xb2, 0, seqLen * dim);
    }

    /** Per-head RMS-norm then NeoX RoPE over each row (shared by Q and K). */
    private void headNormRope(
            MemoryView<MemorySegment> t,
            int rowStride,
            int nHeads,
            int headSize,
            int halfHead,
            MemoryView<MemorySegment> normW,
            int seqLen,
            MemoryView<MemorySegment> cos,
            MemoryView<MemorySegment> sin) {
        float eps = configuration.rmsNormEps;
        Parallel.forRows(
                seqLen,
                s -> {
                    for (int h = 0; h < nHeads; h++) {
                        long off = (long) s * rowStride + (long) h * headSize;
                        Norms.rmsnorm(t, off, t, off, normW, headSize, eps);
                    }
                    for (int h = 0; h < nHeads; h++) {
                        RoPE.applyNeox(
                                t,
                                (long) s * rowStride + (long) h * headSize,
                                s,
                                cos,
                                sin,
                                halfHead);
                    }
                });
    }

    // --- FFN ---

    /** Pre-norm FFN added to the residual: dense SiLU-GLU, or top-k MoE when the layer routes. */
    private void feedForward(State state, int l, int seqLen) {
        Configuration config = configuration;
        if (weights.layers[l].moe() != null) {
            moeFeedForward(state, l, seqLen);
            return;
        }
        int dim = config.embeddingLength, hiddenDim = config.feedForwardLength[l];
        float eps = config.rmsNormEps;
        DenseFfnWeights ffn = weights.layers[l].dense();
        MemoryView<MemorySegment> ffnNormW = weights.layers[l].ffnNorm(),
                postFfwW = weights.layers[l].postFfnNorm();
        Parallel.forRows(
                seqLen,
                s ->
                        Norms.rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                ffnNormW,
                                dim,
                                eps));
        gemm(ffn.gate(), state.xb, dim, state.hb, hiddenDim, seqLen, hiddenDim, dim);
        gemm(ffn.up(), state.xb, dim, state.hb2, hiddenDim, seqLen, hiddenDim, dim);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.siluMultiply(
                                state.hb, s * hiddenDim, state.hb2, s * hiddenDim, hiddenDim));
        gemm(ffn.down(), state.hb, hiddenDim, state.xb, dim, seqLen, dim, hiddenDim);
        if (postFfwW != null)
            Parallel.forRows(
                    seqLen,
                    s ->
                            Norms.rmsnorm(
                                    state.xb,
                                    (long) s * dim,
                                    state.xb,
                                    (long) s * dim,
                                    postFfwW,
                                    dim,
                                    eps));
        Ops.addInPlace(state.residual, 0, state.xb, 0, seqLen * dim);
    }

    /**
     * Top-k MoE FFN (LFM-style): no shared MLP, no expert pre/post norms. Router → optional {@code
     * exp_probs_b} bias → softmax|sigmoid → top-k → normalize the k weights → per-expert (separate)
     * gate/up/SiLU/down, prob-weighted into the residual via the shared CSR {@link Moe#dispatch}.
     */
    private void moeFeedForward(State state, int l, int seqLen) {
        Configuration config = configuration;
        int dim = config.embeddingLength, expertFF = config.expertFeedForwardLength;
        int nExperts = config.expertCount, topK = config.expertUsedCount;
        float eps = config.rmsNormEps;
        MoeFfnWeights moe = weights.layers[l].moe();
        MemoryView<MemorySegment> ffnNormW = weights.layers[l].ffnNorm(),
                postFfnNorm = weights.layers[l].postFfnNorm();

        // pre-norm into xb, then route on it
        Parallel.forRows(
                seqLen,
                s ->
                        Norms.rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                ffnNormW,
                                dim,
                                eps));
        gemm(moe.router(), state.xb, dim, state.moeRouterB, nExperts, seqLen, nExperts, dim);

        Raw routerB = Views.rawF32(state.moeRouterB, "moeRouterB");
        Raw expBias =
                moe.expProbsBias() != null
                        ? Views.rawF32(moe.expProbsBias(), "expProbsBias")
                        : null;
        int[] counts = state.moeExpertCounts;
        Arrays.fill(counts, 0);
        for (int s = 0; s < seqLen; s++) {
            long ro = (long) s * nExperts;
            if (expBias != null) {
                for (int i = 0; i < nExperts; i++)
                    writeFloat(
                            routerB.vseg(),
                            routerB.vbase() + (ro + i) * Float.BYTES,
                            readFloat(routerB.vseg(), routerB.vbase() + (ro + i) * Float.BYTES)
                                    + readFloat(
                                            expBias.vseg(),
                                            expBias.vbase() + (long) i * Float.BYTES));
            }
            if (config.expertGatingFunc == 2)
                Ops.mapInPlace(
                        state.moeRouterB, ro, nExperts, v -> (float) (1.0 / (1.0 + Math.exp(-v))));
            else Ops.softmaxInPlace(state.moeRouterB, ro, nExperts);
            for (int ki = 0; ki < topK; ki++) {
                int best = 0;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int ei = 0; ei < nExperts; ei++) {
                    float v = readFloat(routerB.vseg(), routerB.vbase() + (ro + ei) * Float.BYTES);
                    if (v > bestVal) {
                        bestVal = v;
                        best = ei;
                    }
                }
                state.moeRowTopE[s * topK + ki] = best;
                state.moeRowTopP[s * topK + ki] = bestVal;
                writeFloat(
                        routerB.vseg(),
                        routerB.vbase() + (ro + best) * Float.BYTES,
                        Float.NEGATIVE_INFINITY);
                counts[best]++;
            }
            float sum = 0f; // normalize the k routed weights
            for (int ki = 0; ki < topK; ki++) sum += state.moeRowTopP[s * topK + ki];
            for (int ki = 0; ki < topK; ki++) state.moeRowTopP[s * topK + ki] /= sum;
        }

        Moe.Routing r = state.moeRouting;
        r.seqLen = seqLen;
        r.topK = topK;
        r.numExperts = nExperts;
        Moe.dispatch(
                r,
                dim,
                state.xb,
                state.moeGather,
                state.moeDownB,
                state.moeOutB,
                null,
                (e, n, gather, out) -> {
                    gemm(
                            moe.gateExps(),
                            (long) e * expertFF * dim,
                            gather,
                            dim,
                            state.hb,
                            expertFF,
                            n,
                            expertFF,
                            dim);
                    gemm(
                            moe.upExps(),
                            (long) e * expertFF * dim,
                            gather,
                            dim,
                            state.hb2,
                            expertFF,
                            n,
                            expertFF,
                            dim);
                    Parallel.forRows(
                            n,
                            j ->
                                    Activations.siluMultiply(
                                            state.hb,
                                            j * expertFF,
                                            state.hb2,
                                            j * expertFF,
                                            expertFF));
                    gemm(
                            moe.downExps(),
                            (long) e * dim * expertFF,
                            state.hb,
                            expertFF,
                            out,
                            dim,
                            n,
                            dim,
                            expertFF);
                });

        Parallel.forRows(
                seqLen,
                s -> {
                    if (postFfnNorm != null)
                        Norms.rmsnorm(
                                state.moeOutB,
                                (long) s * dim,
                                state.moeOutB,
                                (long) s * dim,
                                postFfnNorm,
                                dim,
                                eps);
                    Ops.addInPlace(
                            state.residual, (long) s * dim, state.moeOutB, (long) s * dim, dim);
                });
    }

    /** Write the chunk's K/V into the (linear) cache for attention layers. */
    private void commitKv(State state, int startPos, int seqLen) {
        for (int l = 0; l < configuration.numberOfLayers; l++) {
            if (state.keyCache[l] == null) continue; // recurrent layer
            int kvDim = configuration.kvDim(l);
            for (int s = 0; s < seqLen; s++) {
                long kvPos = startPos + s;
                Convert.f32ToF16(
                        state.batchK[l], (long) s * kvDim, state.keyCache[l], kvPos * kvDim, kvDim);
                Convert.f32ToF16(
                        state.batchV[l],
                        (long) s * kvDim,
                        state.valueCache[l],
                        kvPos * kvDim,
                        kvDim);
            }
        }
    }

    // === Bidirectional embedding (the LFM2.5-Embedding checkpoints) ===

    /**
     * Segmented NON-causal forward for packed sequences: RoPE positions restart per sequence, every
     * token attends to its WHOLE sequence (and never a neighbour), the short-conv history zeroes at
     * each sequence start. Each sequence is entirely inside this chunk - {@link #embed} groups on
     * sequence boundaries - so no KV cache is read or written.
     */
    private void forwardSegmented(State state, int[] tokens, int[] seqLen, int n) {
        int[] posOf = new int[n];
        int[] segRow0 = new int[seqLen.length];
        int at = 0;
        for (int g = 0; g < seqLen.length; g++) {
            segRow0[g] = at;
            for (int p = 0; p < seqLen[g]; p++) posOf[at++] = p;
        }
        if (at != n)
            throw new IllegalArgumentException("seqLen sums to " + at + ", batch has " + n);
        RoPE.fill(
                state.ropeCos,
                state.ropeSin,
                posOf,
                n,
                configuration.headSizeFull / 2,
                weights.rope());
        embedTokens(state, tokens, 0, n);
        for (int l = 0; l < configuration.numberOfLayers; l++) {
            if (configuration.isRecurrentLayer(l))
                shortConvMixerCentered(state, l, n, segRow0, seqLen);
            else attentionBidirectional(state, l, n, segRow0, seqLen);
            feedForward(state, l, n);
        }
    }

    /**
     * The NON-causal short-conv mixer: llama.cpp pads symmetrically for a CENTERED window when
     * {@code attention.causal=false} ("causal prepends the state, non-causal pads symmetrically"),
     * so with {@code dConv} taps the window is {@code [s-pad .. s-pad+dConv-1]}, {@code pad =
     * (dConv-1)/2}, zeros beyond the sequence's own edges. Same pre-norm/in-proj/out-proj as {@link
     * #shortConvMixer}; only the scan differs, and no rolling state is read or written.
     */
    private void shortConvMixerCentered(
            State state, int l, int seqLen, int[] segRow0, int[] segLen) {
        int dim = configuration.embeddingLength;
        float eps = configuration.rmsNormEps;
        ShortConvWeights sc = weights.layers[l].shortConv();
        MemoryView<MemorySegment> preNorm = weights.layers[l].attnNorm();
        Parallel.forRows(
                seqLen,
                s ->
                        Norms.rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                preNorm,
                                dim,
                                eps));
        gemm(sc.inProj(), state.xb, dim, state.shortConvTmp, 3 * dim, seqLen, 3 * dim, dim);

        int dConv = configuration.shortConvLCache, pad = (dConv - 1) / 2;
        Raw kernel = Views.rawF32(sc.kernel(), "kernel"); // per channel: dConv taps at c*dConv + k
        Raw tmp = Views.rawF32(state.shortConvTmp, "shortConvTmp");
        Raw out = Views.rawF32(state.xb2, "xb2");
        // materialize bx = B*x in place over the B block first: the centered window reads
        // NEIGHBOUR rows, so every bx must exist before any output row is computed
        for (int s = 0; s < seqLen; s++) {
            int tmpOff = s * 3 * dim;
            for (int c = 0; c < dim; c++) {
                writeFloat(
                        tmp.vseg(),
                        tmp.vbase() + (long) (tmpOff + c) * Float.BYTES,
                        readFloat(tmp.vseg(), tmp.vbase() + (long) (tmpOff + c) * Float.BYTES)
                                * readFloat(
                                        tmp.vseg(),
                                        tmp.vbase() + (long) (tmpOff + 2 * dim + c) * Float.BYTES));
            }
        }
        for (int g = 0; g < segRow0.length; g++) {
            int r0 = segRow0[g], rEnd = r0 + segLen[g];
            for (int s = r0; s < rEnd; s++) {
                int tmpOff = s * 3 * dim, outOff = s * dim;
                for (int c = 0; c < dim; c++) {
                    float cg =
                            readFloat(
                                    tmp.vseg(),
                                    tmp.vbase() + (long) (tmpOff + dim + c) * Float.BYTES);
                    int kBase = c * dConv;
                    float sum = 0f;
                    for (int k = 0; k < dConv; k++) {
                        int row = s - pad + k; // zero beyond this sequence's own edges
                        if (row >= r0 && row < rEnd) {
                            sum +=
                                    readFloat(
                                                    tmp.vseg(),
                                                    tmp.vbase()
                                                            + ((long) row * 3 * dim + c)
                                                                    * Float.BYTES)
                                            * readFloat(
                                                    kernel.vseg(),
                                                    kernel.vbase()
                                                            + (long) (kBase + k) * Float.BYTES);
                        }
                    }
                    writeFloat(
                            out.vseg(), out.vbase() + (long) (outOff + c) * Float.BYTES, cg * sum);
                }
            }
        }
        gemm(sc.outProj(), state.xb2, dim, state.shortConvOut, dim, seqLen, dim, dim);
        Ops.addInPlace(state.residual, 0, state.shortConvOut, 0, seqLen * dim);
    }

    /**
     * As {@link #attention} up to the flash call, then one NON-causal prefill per sequence over its
     * own rows (gathered to scratch, like Qwen3's segmented attention). No cache writes: nothing
     * ever reads them - each sequence is complete in this chunk.
     */
    private void attentionBidirectional(
            State state, int l, int seqLen, int[] segRow0, int[] segLen) {
        Configuration config = configuration;
        int headSize = config.headSizeFull;
        int queryDim = config.queryDim(), kvDim = config.kvDim(l);
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeadsPerLayer[l];
        attentionProject(state, l, seqLen);
        MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];
        float scale = 1.0f / (float) Math.sqrt(headSize);
        EmbedScratch es = state.embedScratch(config);
        for (int g = 0; g < segRow0.length; g++) {
            int r0 = segRow0[g], sl = segLen[g];
            Convert.copyF32(state.query, (long) r0 * queryDim, es.segQ, 0, (long) sl * queryDim);
            Convert.copyF32(bK, (long) r0 * kvDim, es.segK, 0, (long) sl * kvDim);
            Convert.copyF32(bV, (long) r0 * kvDim, es.segV, 0, (long) sl * kvDim);
            FlashAttention.bidirectionalPrefill(
                    es.segQ,
                    es.segOut,
                    es.segK,
                    es.segV,
                    config.numberOfHeads,
                    sl,
                    headSize,
                    kvDim,
                    queryDim,
                    kvMul,
                    scale);
            Convert.copyF32(es.segOut, 0, state.xbK, (long) r0 * queryDim, (long) sl * queryDim);
        }
        attentionFinish(state, l, seqLen);
    }

    /**
     * The sentence embedding: final-norm the pooled row, L2-normalize - CLS pooling reads the
     * sequence's FIRST retained row (its BOS). {@code index} addresses retained rows exactly as
     * {@code logits}' output does. The returned view is a REUSED per-state buffer.
     */
    @Override
    public MemoryView<?> pool(State s, int index) {
        int dim = configuration.embeddingLength;
        int row = s.lastChunkLen - s.outputCount + index;
        MemoryView<MemorySegment> out = s.embedScratch(configuration).embOut;
        Norms.rmsnorm(
                out,
                0,
                s.residual,
                (long) row * dim,
                weights.finalNorm,
                dim,
                configuration.rmsNormEps);
        float inv = l2Inv(out, dim);
        Ops.mapInPlace(out, 0, dim, v -> v * inv);
        return out;
    }

    /** {@code 1/||t[0..n)||}, or 0 for a zero vector - the shared L2-normalization factor. */
    private static float l2Inv(MemoryView<MemorySegment> t, int n) {
        float ss = Norms.sumOfSquares(t, 0, n);
        return ss > 0 ? (float) (1.0 / Math.sqrt(ss)) : 0f;
    }

    /**
     * The ColBERT per-token read for one retained row: final-norm, {@code dense_2} projection to
     * {@code embeddingLengthOut}, L2-normalized - what llama.cpp's {@code build_dense_out} does to
     * {@code t_embd}, plus the client-side normalize the reference stack applies before MaxSim.
     * {@code out} is the caller's buffer. (The ColBERT face class itself is not part of this
     * slice.)
     */
    void colbertRow(State s, int row, float[] out) {
        int dim = configuration.embeddingLength;
        int outDim = configuration.embeddingLengthOut;
        MemoryView<MemorySegment> normed = s.embedScratch(configuration).embOut;
        Norms.rmsnorm(
                normed,
                0,
                s.residual,
                (long) row * dim,
                weights.finalNorm,
                dim,
                configuration.rmsNormEps);
        MemoryView<MemorySegment> projected = s.embedScratch(configuration).colbertOut;
        gemv(weights.dense2(), normed, projected, outDim, dim);
        float inv = l2Inv(projected, outDim);
        Raw p = Views.rawF32(projected, "colbertOut");
        for (int i = 0; i < outDim; i++)
            out[i] = readFloat(p.vseg(), p.vbase() + (long) i * Float.BYTES) * inv;
    }

    /**
     * Bidirectional embedding overrides the generic chunk-streaming default: a sequence attends to
     * ALL of its tokens, so it must be forwarded WHOLE - {@link #forEachSequence} re-cuts groups on
     * sequence boundaries. Emits each sequence's CLS (first-row) embedding, in input order.
     */
    @Override
    public void embed(State state, Batch.Input.Sequences seqs, Consumer<MemoryView<?>> sink) {
        if (!configuration.isEmbedder())
            throw new UnsupportedOperationException(
                    "this LFM2.5 checkpoint is not an embedder - load LFM2.5-Embedding"
                            + " (pooling_type=CLS, attention.causal=false)");
        int[] len = seqs.seqLen();
        int[] ids = seqs.tokens().ids();
        int[][] sequences = new int[len.length][];
        for (int i = 0, at = 0; i < len.length; at += len[i], i++) {
            sequences[i] = Arrays.copyOfRange(ids, at, at + len[i]);
        }
        // CLS: the sequence's FIRST row (its BOS)
        forEachSequence(
                state, sequences, (index, rowStart) -> sink.accept(embedding(state, rowStart)));
    }

    /** Visits each sequence right after its group's forward, while its rows are still retained. */
    interface SequenceVisitor {
        void sequence(int index, int rowStart);
    }

    /**
     * The bidirectional ingest loop both retrieval faces share (CLS embedding, ColBERT): packs
     * whole sequences greedily into groups capped by {@code min(batch, context)} - a sequence must
     * be forwarded WHOLE, so one over the cap refuses by name - resets the state per group (groups
     * are independent: positions and conv history restart), ingests with ALL outputs, and hands
     * each sequence to {@code visitor} with its first retained row. Returns the total token count.
     * Claims the state once across all groups; the per-ingest claims nest inside.
     */
    int forEachSequence(State state, int[][] seqs, SequenceVisitor visitor) {
        int cap = Math.min(state.batchCapacity, state.contextCapacity);
        int total = 0, seq = 0;
        state.enter();
        try {
            while (seq < seqs.length) {
                int end = seq, tokens = 0;
                while (end < seqs.length && tokens + seqs[end].length <= cap) {
                    tokens += seqs[end].length;
                    end++;
                }
                if (end == seq)
                    throw new IllegalArgumentException(
                            "sequence "
                                    + seq
                                    + " is "
                                    + seqs[seq].length
                                    + " tokens and bidirectional attention forwards a sequence"
                                    + " whole (the cap here is "
                                    + cap
                                    + ") - raise -Djinfer.batchCapacity/contextLength above it,"
                                    + " or chunk the text smaller");
                total += tokens;
                state.reset();
                ingest(state, Batch.pack(Arrays.copyOfRange(seqs, seq, end)));
                for (int i = seq, row = 0; i < end; row += seqs[i].length, i++) {
                    visitor.sequence(i, row);
                }
                seq = end;
            }
        } finally {
            state.exit();
        }
        return total;
    }

    // === gemm/gemv entry shims (the old FloatTensor virtuals, resolved to MatMul.mm) ===

    /** {@code c = w · aᵀ} per row: the old {@code w.gemm(a, aStride, c, cStride, n, m, k)}. */
    private static void gemm(
            MemoryView<MemorySegment> w,
            MemoryView<MemorySegment> a,
            int aStride,
            MemoryView<MemorySegment> c,
            int cStride,
            int n,
            int m,
            int k) {
        MatMul.mm(w, 0, k, a, 0, aStride, c, 0, cStride, m, n, k);
    }

    /** As above at a weight offset (the MoE expert slice). */
    private static void gemm(
            MemoryView<MemorySegment> w,
            long wOff,
            MemoryView<MemorySegment> a,
            int aStride,
            MemoryView<MemorySegment> c,
            int cStride,
            int n,
            int m,
            int k) {
        MatMul.mm(w, wOff, k, a, 0, aStride, c, 0, cStride, m, n, k);
    }

    /** {@code c = w · a} (single row): the old {@code w.matmul(a, c, m, k)}. */
    private static void gemv(
            MemoryView<MemorySegment> w,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> c,
            int m,
            int k) {
        MatMul.mm(w, 0, k, a, 0, k, c, 0, m, m, 1, k);
    }

    // === Configuration ===

    public record Configuration(
            int embeddingLength,
            int[] feedForwardLength,
            int numberOfLayers,
            int numberOfHeads,
            int[] numberOfKeyValueHeadsPerLayer,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            float ropeTheta,
            int headSizeFull,
            float logitSoftcapping,
            int shortConvLCache,
            int expertCount,
            int expertUsedCount,
            int expertFeedForwardLength,
            int leadingDenseBlockCount,
            int expertGatingFunc,
            boolean causalAttention,
            int poolingType,
            int embeddingLengthOut)
            implements Config {

        /** The widest attention kvDim, for scratch that must fit any layer. */
        public int maxKvDim() {
            int max = 0;
            for (int l = 0; l < numberOfLayers; l++) max = Math.max(max, kvDim(l));
            return max;
        }

        public int headSize() {
            return headSizeFull;
        }

        public int queryDim() {
            return numberOfHeads * headSizeFull;
        }

        public int kvDim(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] * headSizeFull;
        }

        public boolean isRecurrentLayer(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] == 0;
        }

        public boolean isMoE() {
            return expertCount > 0;
        }

        public boolean isMoELayer(int layer) {
            return expertCount > 0 && layer >= leadingDenseBlockCount;
        }

        /** An embedding checkpoint (LFM2.5-Embedding): non-causal attention with CLS pooling. */
        public boolean isEmbedder() {
            return !causalAttention && poolingType == POOLING_CLS;
        }

        /** A ColBERT checkpoint: non-causal with a per-token {@code dense_2} projection width. */
        public boolean isColbert() {
            return !causalAttention && embeddingLengthOut > 0;
        }

        public int maxHiddenDim() {
            int max = expertCount > 0 ? expertFeedForwardLength : 0;
            for (int ff : feedForwardLength) max = Math.max(max, ff);
            return max;
        }
    }

    // === Weights (per-layer union: attention|shortConv, dense|moe) ===

    public record AttentionWeights(
            MemoryView<MemorySegment> wq,
            MemoryView<MemorySegment> wk,
            MemoryView<MemorySegment> wv,
            MemoryView<MemorySegment> wo,
            MemoryView<MemorySegment> qNorm,
            MemoryView<MemorySegment> kNorm) {}

    /** {@code kernel}: per-channel dConv taps (c*dConv + k), as the GGUF lays them out. */
    public record ShortConvWeights(
            MemoryView<MemorySegment> kernel,
            MemoryView<MemorySegment> inProj,
            MemoryView<MemorySegment> outProj) {}

    public record DenseFfnWeights(
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> up,
            MemoryView<MemorySegment> down) {}

    public record MoeFfnWeights(
            MemoryView<MemorySegment> router,
            MemoryView<MemorySegment> gateExps,
            MemoryView<MemorySegment> upExps,
            MemoryView<MemorySegment> downExps,
            MemoryView<MemorySegment> expProbsBias) {}

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> postAttnNorm,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> postFfnNorm,
            AttentionWeights attention,
            ShortConvWeights shortConv,
            DenseFfnWeights dense,
            MoeFfnWeights moe) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbeddings,
            LayerWeights[] layers,
            MemoryView<MemorySegment> finalNorm,
            RoPE.Schedule rope,
            MemoryView<MemorySegment> wcls,
            MemoryView<MemorySegment> dense2) {} // ColBERT's per-token projection; null elsewhere

    // === State ===

    public static final class State extends BaseState {
        final int contextCapacity, batchCapacity;
        final MemoryView<MemorySegment> residual, xb, xbK, xb2, hb, hb2, query, logits;
        final MemoryView<MemorySegment> ropeCos, ropeSin;
        final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch(arena);
        final MemoryView<MemorySegment>[] keyCache,
                valueCache,
                batchK,
                batchV; // per layer; null on recurrent layers
        final MemoryView<MemorySegment>[] shortConvState; // per layer; null on attention layers
        final MemoryView<MemorySegment> shortConvTmp, shortConvOut;
        // MoE scratch (chunk-wide CSR routing); allocated only when the model has experts, else
        // null.
        final MemoryView<MemorySegment> moeRouterB, moeGather, moeDownB, moeOutB;
        final int[] moeExpertCounts, moeRowTopE;
        final float[] moeRowTopP;
        final Moe.Routing moeRouting;

        EmbedScratch embedScratch; // lazy: only the embedding checkpoints ever pay for it

        EmbedScratch embedScratch(Configuration config) {
            if (embedScratch == null) embedScratch = new EmbedScratch(config, batchCapacity, arena);
            return embedScratch;
        }

        /**
         * Recycles this allocation for a fresh sequence: cursor to 0 and the RECURRENT buffers
         * zeroed - stale KV rows beyond the cursor are attention-masked and harmless, but the
         * rolling short-conv state carries values across positions and would leak the previous
         * conversation into the next one.
         */
        @Override
        public void reset() {
            resumeAt(0);
            for (MemoryView<MemorySegment> conv : shortConvState) {
                if (conv != null) {
                    Ops.fillInPlace(conv, 0, Math.toIntExact(conv.logicalSize()), 0f);
                }
            }
        }

        @SuppressWarnings("unchecked")
        State(Configuration config, int contextCapacity, int batchCapacity, Arena arena) {
            super(arena);
            if (contextCapacity > config.contextLength()) {
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds model contextLength "
                                + config.contextLength());
            }
            this.contextCapacity = contextCapacity;
            int c = Math.max(1, batchCapacity);
            this.batchCapacity = c;
            int dim = config.embeddingLength;
            int maxQueryDim = config.queryDim();
            int maxHiddenDim = config.maxHiddenDim();
            this.residual = Views.allocateF32(arena, c * dim);
            this.xb = Views.allocateF32(arena, c * dim);
            this.xb2 = Views.allocateF32(arena, c * dim);
            this.xbK = Views.allocateF32(arena, c * maxQueryDim);
            this.query = Views.allocateF32(arena, c * maxQueryDim);
            this.hb = Views.allocateF32(arena, c * maxHiddenDim);
            this.hb2 = Views.allocateF32(arena, c * maxHiddenDim);
            this.logits = Views.allocateF32(arena, config.vocabularySize);
            // rotary values for the batch about to be ingested: sized by BATCH, never context
            this.ropeCos = Views.allocateF32(arena, c * (config.headSizeFull / 2));
            this.ropeSin = Views.allocateF32(arena, c * (config.headSizeFull / 2));
            this.shortConvTmp = Views.allocateF32(arena, c * 3 * dim);
            this.shortConvOut = Views.allocateF32(arena, c * dim);
            int n = config.numberOfLayers;
            this.keyCache = new MemoryView[n];
            this.valueCache = new MemoryView[n];
            this.batchK = new MemoryView[n];
            this.batchV = new MemoryView[n];
            this.shortConvState = new MemoryView[n];
            int hist = Math.max(config.shortConvLCache - 1, 0);
            for (int l = 0; l < n; l++) {
                if (config.isRecurrentLayer(l)) {
                    shortConvState[l] = Views.allocateF32(arena, hist * dim);
                } else {
                    int kvDim = config.kvDim(l);
                    keyCache[l] = Views.allocateF16(arena, contextCapacity, kvDim);
                    valueCache[l] = Views.allocateF16(arena, contextCapacity, kvDim);
                    batchK[l] = Views.allocateF32(arena, c * kvDim);
                    batchV[l] = Views.allocateF32(arena, c * kvDim);
                }
            }
            if (config.isMoE()) {
                int e = config.expertCount, tk = config.expertUsedCount;
                this.moeRouterB = Views.allocateF32(arena, c * e);
                this.moeGather = Views.allocateF32(arena, c * dim);
                this.moeDownB = Views.allocateF32(arena, c * dim);
                this.moeOutB = Views.allocateF32(arena, c * dim);
                this.moeExpertCounts = new int[e];
                this.moeRowTopE = new int[c * tk];
                this.moeRowTopP = new float[c * tk];
                this.moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts);
            } else {
                this.moeRouterB = this.moeGather = this.moeDownB = this.moeOutB = null;
                this.moeExpertCounts = this.moeRowTopE = null;
                this.moeRowTopP = null;
                this.moeRouting = null;
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

    /**
     * Per-state scratch for the bidirectional embedding path, from the state's own arena (freed
     * with the state): per-sequence Q/K/V/out gathers plus the pooled-output row.
     */
    static final class EmbedScratch {
        final MemoryView<MemorySegment> segQ, segK, segV, segOut, embOut, colbertOut;

        EmbedScratch(Configuration config, int batchCapacity, Arena arena) {
            int queryDim = config.queryDim(), kvDim = config.maxKvDim();
            this.segQ = Views.allocateF32(arena, batchCapacity * queryDim);
            this.segOut = Views.allocateF32(arena, batchCapacity * queryDim);
            this.segK = Views.allocateF32(arena, batchCapacity * kvDim);
            this.segV = Views.allocateF32(arena, batchCapacity * kvDim);
            this.embOut = Views.allocateF32(arena, config.embeddingLength());
            this.colbertOut = Views.allocateF32(arena, Math.max(1, config.embeddingLengthOut()));
        }
    }

    // === Loading ===

    public static Lfm2 loadModel(Path ggufPath, Arena arena) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, arena);
        }
    }

    public static Lfm2 loadModel(FileChannel fileChannel, GGUF gguf, Arena arena)
            throws IOException {
        return loadModel(fileChannel, gguf, arena, null);
    }

    /**
     * As above with a caller-supplied tokenizer; null = the GGUF's own (toknroll's builtin
     * registrations; the jinfer-llm {@code -Djinfer.preTokenizer.*} override layer is NOT part of
     * this slice).
     */
    public static Lfm2 loadModel(
            FileChannel fileChannel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        if (tokenizer == null) {
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        }
        String arch = gguf.getString("general.architecture");

        int contextLength = gguf.getValue(int.class, arch + ".context_length");

        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int headSizeFull = embeddingLength / numberOfHeads;
        float rmsNormEps =
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 1000000f);
        float logitSoftcapping =
                gguf.getValueOrDefault(float.class, arch + ".final_logit_softcapping", 0f);
        int shortConvLCache = gguf.getValueOrDefault(int.class, arch + ".shortconv.l_cache", 3);
        int expertCount = gguf.getValueOrDefault(int.class, arch + ".expert_count", 0);
        int expertUsedCount = gguf.getValueOrDefault(int.class, arch + ".expert_used_count", 0);
        int expertFeedForwardLength =
                gguf.getValueOrDefault(int.class, arch + ".expert_feed_forward_length", 0);
        int leadingDenseBlockCount =
                gguf.getValueOrDefault(
                        int.class, arch + ".leading_dense_block_count", numberOfLayers);
        int expertGatingFunc = gguf.getValueOrDefault(int.class, arch + ".expert_gating_func", 1);
        // the embedding checkpoints (LFM2.5-Embedding) declare non-causal attention + CLS pooling;
        // generative GGUFs carry neither key
        boolean causalAttention =
                gguf.getValueOrDefault(boolean.class, arch + ".attention.causal", true);
        int poolingType = gguf.getValueOrDefault(int.class, arch + ".pooling_type", 0);
        // ColBERT checkpoints project per-token hiddens to this width through dense_2
        int embeddingLengthOut =
                gguf.getValueOrDefault(int.class, arch + ".embedding_length_out", 0);

        int[] feedForwardLength;
        Object ffnRaw = gguf.getValue(Object.class, arch + ".feed_forward_length");
        if (ffnRaw instanceof int[] arr) {
            feedForwardLength = arr;
        } else {
            feedForwardLength = new int[numberOfLayers];
            Arrays.fill(feedForwardLength, (int) ffnRaw);
        }

        // Per-layer kv-head count: 0 marks a recurrent (short-conv) layer (no attn_k tensor);
        // attention layers derive it from the K-projection's row count (GGUF shape[1]).
        int[] kvHeads = new int[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            var kWeight = gguf.getTensor("blk." + i + ".attn_k.weight");
            kvHeads[i] = kWeight != null ? Math.toIntExact(kWeight.shape()[1]) / headSizeFull : 0;
        }

        Configuration config =
                new Configuration(
                        embeddingLength,
                        feedForwardLength,
                        numberOfLayers,
                        numberOfHeads,
                        kvHeads,
                        tokenizer.vocabulary().size(),
                        contextLength,
                        rmsNormEps,
                        ropeTheta,
                        headSizeFull,
                        logitSoftcapping,
                        shortConvLCache,
                        expertCount,
                        expertUsedCount,
                        expertFeedForwardLength,
                        leadingDenseBlockCount,
                        expertGatingFunc,
                        causalAttention,
                        poolingType,
                        embeddingLengthOut);

        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(fileChannel, gguf, arena);
        return new Lfm2(config, tokenizer, loadWeights(tensors, config));
    }

    // ---- loadWeights helpers: the old ModelLoader.toF32Tensor/loadQuantized fail-fast contract --

    private static MemoryView<MemorySegment> quant(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        return Objects.requireNonNull(tensors.get(name), name);
    }

    /** F32 view by name (dtype checked AT LOAD, the old toF32Tensor fail-fast), or throw. */
    private static MemoryView<MemorySegment> f32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> v = quant(tensors, name);
        Views.requireDtype(v, DataType.FP32, name);
        return v;
    }

    private static MemoryView<MemorySegment> quantOrNull(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        return tensors.get(name);
    }

    private static MemoryView<MemorySegment> f32OrNull(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> v = tensors.get(name);
        if (v != null) Views.requireDtype(v, DataType.FP32, name);
        return v;
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors, Configuration config) {
        int n = config.numberOfLayers;
        RoPE.Schedule rope = RoPE.plain(config.headSizeFull, config.ropeTheta);

        MemoryView<MemorySegment> tokenEmbeddings = quant(tensors, "token_embd.weight");
        MemoryView<MemorySegment> wcls =
                tensors.containsKey("output.weight")
                        ? quant(tensors, "output.weight")
                        : tokenEmbeddings; // tied embeddings
        // LFM2.5 names the final norm token_embd_norm (no separate output_norm); embeddings are
        // tied.
        MemoryView<MemorySegment> finalNorm =
                f32(
                        tensors,
                        tensors.containsKey("output_norm.weight")
                                ? "output_norm.weight"
                                : "token_embd_norm.weight");

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            MemoryView<MemorySegment> attnNorm = f32(tensors, p + "attn_norm.weight");
            MemoryView<MemorySegment> postAttnNorm =
                    f32OrNull(tensors, p + "post_attention_norm.weight");
            MemoryView<MemorySegment> ffnNorm = f32(tensors, p + "ffn_norm.weight");
            MemoryView<MemorySegment> postFfnNorm = f32OrNull(tensors, p + "post_ffw_norm.weight");

            AttentionWeights attention = null;
            ShortConvWeights shortConv = null;
            if (config.isRecurrentLayer(i)) {
                shortConv =
                        new ShortConvWeights(
                                f32(tensors, p + "shortconv.conv.weight"),
                                quant(tensors, p + "shortconv.in_proj.weight"),
                                quant(tensors, p + "shortconv.out_proj.weight"));
            } else {
                attention =
                        new AttentionWeights(
                                quant(tensors, p + "attn_q.weight"),
                                quant(tensors, p + "attn_k.weight"),
                                quantOrNull(tensors, p + "attn_v.weight"),
                                quant(tensors, p + "attn_output.weight"),
                                f32(tensors, p + "attn_q_norm.weight"),
                                f32(tensors, p + "attn_k_norm.weight"));
            }

            DenseFfnWeights dense = null;
            MoeFfnWeights moe = null;
            if (config.isMoELayer(i)) {
                moe =
                        new MoeFfnWeights(
                                quant(tensors, p + "ffn_gate_inp.weight"),
                                quant(tensors, p + "ffn_gate_exps.weight"),
                                quant(tensors, p + "ffn_up_exps.weight"),
                                quant(tensors, p + "ffn_down_exps.weight"),
                                f32OrNull(tensors, p + "exp_probs_b.bias"));
            } else {
                dense =
                        new DenseFfnWeights(
                                quant(tensors, p + "ffn_gate.weight"),
                                quant(tensors, p + "ffn_up.weight"),
                                quant(tensors, p + "ffn_down.weight"));
            }
            layers[i] =
                    new LayerWeights(
                            attnNorm,
                            postAttnNorm,
                            ffnNorm,
                            postFfnNorm,
                            attention,
                            shortConv,
                            dense,
                            moe);
        }
        MemoryView<MemorySegment> dense2 =
                tensors.containsKey("dense_2.weight") ? quant(tensors, "dense_2.weight") : null;
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls, dense2);
    }
}
