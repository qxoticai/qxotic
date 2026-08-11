// LFM2.5 (Liquid Foundation Model 2.5) against the MemoryView boundary: a call-site-by-call-site
// port of jinfer-lfm2's Lfm2 (cycle 1 of the FloatTensor migration). Each layer is EITHER GQA
// attention (kv-heads > 0) OR a gated short-convolution mixer (kv-heads == 0); the FFN is EITHER
// dense SwiGLU OR top-k MoE. Text-only. Weights/state/KV are MemoryView<MemorySegment>; kernels
// are the x statics; gemm/gemv entry shims resolve to x.MatMul.mm here (the old virtuals').
package com.qxotic.jinfer.x.models.lfm2;

import static com.qxotic.jinfer.x.Segments.readFloat;
import static com.qxotic.jinfer.x.Segments.writeFloat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.Convert;
import com.qxotic.jinfer.x.Parallel;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.Views.Raw;
import com.qxotic.jinfer.x.boundary.BaseState;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Config;
import com.qxotic.jinfer.x.boundary.EmbeddingModel;
import com.qxotic.jinfer.x.boundary.LanguageModel;
import com.qxotic.jinfer.x.kernels.Activations;
import com.qxotic.jinfer.x.kernels.FlashAttention;
import com.qxotic.jinfer.x.kernels.MatMul;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Moe;
import com.qxotic.jinfer.x.kernels.Norms;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jinfer.x.kernels.RoPE;
import com.qxotic.jinfer.x.kernels.Trace;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryAllocator;
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

    /** The short-conv in_proj's 3-way row split: B | C_gate | x blocks, each dim wide. */
    private static final int SHORTCONV_PARTS = 3;

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
                            s.normed,
                            0,
                            s.residual,
                            (long) row * dim,
                            weights.finalNorm,
                            dim,
                            configuration.rmsNormEps);
                    gemv(weights.wcls, s.normed, s.logits, configuration.vocabularySize, dim);
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
                configuration.headSize / 2,
                weights.rope());
        embedTokens(state, tokens, tokenOffset, seqLen);
        for (int l = 0; l < configuration.numberOfLayers; l++) layer(state, l, startPos, seqLen);
        commitKv(state, startPos, seqLen);
    }

    /** Token-embedding lookup into the residual stream (no scaling, unlike Gemma4). */
    private void embedTokens(State state, int[] tokens, int tokenOffset, int seqLen) {
        Views.checkAlive(weights.tokenEmbeddings, "tokenEmbeddings"); // fail-fast on freed weights
        int dim = configuration.embeddingLength;
        // ponytail: per-row dispatch via Convert.copyToF32 (the cost profile of the old per-row
        // virtual copyTo it replaces); the batched gather-dequant - dispatch hoisted once per
        // table - is a planned, separately-benchmarked commit, not a polish
        for (int s = 0; s < seqLen; s++) {
            Convert.copyToF32(
                    weights.tokenEmbeddings,
                    (long) tokens[tokenOffset + s] * dim,
                    state.residual,
                    (long) s * dim,
                    dim);
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
        ShortConvWeights sc = weights.layers[l].shortConv();
        MemoryView<MemorySegment> preNorm =
                weights.layers[l].attnNorm(); // conv layers use attn_norm as the mixer pre-norm
        Norms.rmsnormRows(
                state.normed, state.residual, preNorm, seqLen, dim, configuration.rmsNormEps);
        gemm(
                sc.inProj(),
                state.normed,
                dim,
                state.shortConvTmp,
                SHORTCONV_PARTS * dim,
                SHORTCONV_PARTS * dim,
                seqLen,
                dim);
        shortConvScan(state, l, seqLen);
        gemm(sc.outProj(), state.branchOut, dim, state.shortConvOut, dim, dim, seqLen, dim);
        Ops.addInPlace(state.residual, 0, state.shortConvOut, 0, seqLen * dim);
    }

    /**
     * Causal short-convolution as a dConv-tap FIR over bx = B∘x rows (scalar; ported from the
     * production {@code Llama.shortConvScan}). For each channel: {@code out[s] = C_gate[s] *
     * (Σ_{k<hist} state[k]·kernel[k] + bx[s]·kernel[hist])}, where {@code state} holds the previous
     * {@code hist=dConv-1} bx values; bx is materialized in place over the B block of shortConvTmp
     * and the newest bx rolls into shortConvState.
     *
     * <p>ponytail: a kernel living in the model - it moves to {@code Convolutions} (with its own
     * differential oracle) when cycle 2 ports {@code Llama.shortConvScan}; the second caller is the
     * API proof, until then it stays where the old tree has it.
     */
    private void shortConvScan(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength;
        int dConv = configuration.shortConvLCache, hist = dConv - 1;
        // seg/base locals, the xcore kernel idiom (MatMul.run): Raw extracted once, then plain
        Raw kernelRaw =
                Views.rawF32(
                        weights.layers[l].shortConv().kernel(),
                        "kernel"); // per channel: dConv taps at c*dConv + k
        Raw convRaw = Views.rawF32(state.shortConvState[l], "shortConvState");
        Raw tmpRaw = Views.rawF32(state.shortConvTmp, "shortConvTmp");
        Raw outRaw = Views.rawF32(state.branchOut, "branchOut");
        MemorySegment ks = kernelRaw.vseg(),
                cs = convRaw.vseg(),
                ts = tmpRaw.vseg(),
                os = outRaw.vseg();
        long kb = kernelRaw.vbase(), cb = convRaw.vbase(), tb = tmpRaw.vbase(), ob = outRaw.vbase();
        for (int s = 0; s < seqLen; s++) {
            int tmpOff = s * SHORTCONV_PARTS * dim, outOff = s * dim;
            for (int c = 0; c < dim; c++) {
                float b = readFloat(ts, tb + 4L * (tmpOff + c));
                float cg = readFloat(ts, tb + 4L * (tmpOff + dim + c));
                float xv = readFloat(ts, tb + 4L * (tmpOff + 2 * dim + c));
                float bx = b * xv;
                writeFloat(ts, tb + 4L * (tmpOff + c), bx);
                int kBase = c * dConv;
                float sum = 0f;
                for (int k = 0; k < hist; k++)
                    sum +=
                            readFloat(cs, cb + 4L * ((long) k * dim + c))
                                    * readFloat(ks, kb + 4L * (kBase + k));
                sum += bx * readFloat(ks, kb + 4L * (kBase + dConv - 1));
                writeFloat(os, ob + 4L * (outOff + c), cg * sum);
                for (int k = 0; k < hist - 1; k++)
                    writeFloat(
                            cs,
                            cb + 4L * ((long) k * dim + c),
                            readFloat(cs, cb + 4L * ((long) (k + 1) * dim + c)));
                if (hist > 0) writeFloat(cs, cb + 4L * ((long) (hist - 1) * dim + c), bx);
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
        int headSize = config.headSize;
        int queryDim = config.queryDim(), kvDim = config.kvDim(l);
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeadsPerLayer[l];
        attentionProject(state, l, seqLen);
        MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];
        float scale = 1.0f / (float) Math.sqrt(headSize);
        if (seqLen > 1) {
            FlashAttention.slidingWindowPrefill(
                    state.query,
                    state.attnOut,
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
                    state.attnOut,
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
        int queryDim = config.queryDim(), kvDim = config.kvDim(l);
        int nKvHeads = config.numberOfKeyValueHeadsPerLayer[l];
        AttentionWeights attn = weights.layers[l].attention();

        MemoryView<MemorySegment> attNormW = weights.layers[l].attnNorm();
        Norms.rmsnormRows(
                state.normed, state.residual, attNormW, seqLen, dim, configuration.rmsNormEps);
        gemm(attn.wq(), state.normed, dim, state.query, queryDim, queryDim, seqLen, dim);
        headNormRope(state, state.query, queryDim, config.numberOfHeads, attn.qNorm(), seqLen);
        MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];
        gemm(attn.wk(), state.normed, dim, bK, kvDim, kvDim, seqLen, dim);
        if (attn.wv() != null) gemm(attn.wv(), state.normed, dim, bV, kvDim, kvDim, seqLen, dim);
        else Convert.copyF32(bK, 0, bV, 0, (long) seqLen * kvDim);
        headNormRope(state, bK, kvDim, nKvHeads, attn.kNorm(), seqLen);
    }

    /** The shared tail: output projection, optional post-norm, added to the residual. */
    private void attentionFinish(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength;
        AttentionWeights attn = weights.layers[l].attention();
        gemm(
                attn.wo(),
                state.attnOut,
                configuration.queryDim(),
                state.branchOut,
                dim,
                dim,
                seqLen,
                configuration.queryDim());
        MemoryView<MemorySegment> postAttW = weights.layers[l].postAttnNorm();
        if (postAttW != null)
            Norms.rmsnormRows(
                    state.branchOut,
                    state.branchOut,
                    postAttW,
                    seqLen,
                    dim,
                    configuration.rmsNormEps);
        Ops.addInPlace(state.residual, 0, state.branchOut, 0, seqLen * dim);
    }

    /** Per-head RMS-norm then NeoX RoPE over each row (shared by Q and K). */
    private void headNormRope(
            State state,
            MemoryView<MemorySegment> t,
            int rowStride,
            int nHeads,
            MemoryView<MemorySegment> normW,
            int seqLen) {
        int headSize = configuration.headSize, halfHeadSize = headSize / 2;
        float eps = configuration.rmsNormEps;
        MemoryView<MemorySegment> cos = state.ropeCos, sin = state.ropeSin;
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
                                halfHeadSize);
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
        DenseFfnWeights ffn = weights.layers[l].dense();
        MemoryView<MemorySegment> ffnNormW = weights.layers[l].ffnNorm(),
                postFfwW = weights.layers[l].postFfnNorm();
        Norms.rmsnormRows(
                state.normed, state.residual, ffnNormW, seqLen, dim, configuration.rmsNormEps);
        gemm(ffn.gate(), state.normed, dim, state.hidden, hiddenDim, hiddenDim, seqLen, dim);
        gemm(ffn.up(), state.normed, dim, state.hidden2, hiddenDim, hiddenDim, seqLen, dim);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.siluMultiply(
                                state.hidden,
                                s * hiddenDim,
                                state.hidden2,
                                s * hiddenDim,
                                hiddenDim));
        gemm(ffn.down(), state.hidden, hiddenDim, state.normed, dim, dim, seqLen, hiddenDim);
        if (postFfwW != null)
            Norms.rmsnormRows(
                    state.normed, state.normed, postFfwW, seqLen, dim, configuration.rmsNormEps);
        Ops.addInPlace(state.residual, 0, state.normed, 0, seqLen * dim);
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

        // pre-norm into normed, then route on it
        Norms.rmsnormRows(
                state.normed, state.residual, ffnNormW, seqLen, dim, configuration.rmsNormEps);
        gemm(moe.router(), state.normed, dim, state.moeRouterB, nExperts, nExperts, seqLen, dim);

        Raw routerB = Views.rawF32(state.moeRouterB, "moeRouterB");
        int[] counts = state.moeExpertCounts;
        Arrays.fill(counts, 0);
        for (int s = 0; s < seqLen; s++) {
            long ro = (long) s * nExperts;
            if (moe.expProbsBias() != null)
                Ops.addInPlace(state.moeRouterB, ro, moe.expProbsBias(), 0, nExperts);
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
                state.normed,
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
                            state.hidden,
                            expertFF,
                            expertFF,
                            n,
                            dim);
                    gemm(
                            moe.upExps(),
                            (long) e * expertFF * dim,
                            gather,
                            dim,
                            state.hidden2,
                            expertFF,
                            expertFF,
                            n,
                            dim);
                    Parallel.forRows(
                            n,
                            j ->
                                    Activations.siluMultiply(
                                            state.hidden,
                                            j * expertFF,
                                            state.hidden2,
                                            j * expertFF,
                                            expertFF));
                    gemm(
                            moe.downExps(),
                            (long) e * dim * expertFF,
                            state.hidden,
                            expertFF,
                            out,
                            dim,
                            dim,
                            n,
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
        EmbedScratch es = state.embedScratch(configuration);
        int[] posOf = es.posOf, segRow0 = es.segRow0;
        int at = 0;
        for (int g = 0; g < seqLen.length; g++) {
            segRow0[g] = at;
            for (int p = 0; p < seqLen[g]; p++) posOf[at++] = p;
        }
        if (at != n)
            throw new IllegalArgumentException("seqLen sums to " + at + ", batch has " + n);
        RoPE.fill(
                state.ropeCos, state.ropeSin, posOf, n, configuration.headSize / 2, weights.rope());
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
        ShortConvWeights sc = weights.layers[l].shortConv();
        MemoryView<MemorySegment> preNorm = weights.layers[l].attnNorm();
        Norms.rmsnormRows(
                state.normed, state.residual, preNorm, seqLen, dim, configuration.rmsNormEps);
        gemm(
                sc.inProj(),
                state.normed,
                dim,
                state.shortConvTmp,
                SHORTCONV_PARTS * dim,
                SHORTCONV_PARTS * dim,
                seqLen,
                dim);

        int dConv = configuration.shortConvLCache, pad = (dConv - 1) / 2;
        Raw kernel = Views.rawF32(sc.kernel(), "kernel"); // per channel: dConv taps at c*dConv + k
        Raw tmp = Views.rawF32(state.shortConvTmp, "shortConvTmp");
        Raw out = Views.rawF32(state.branchOut, "branchOut");
        // materialize bx = B*x in place over the B block first: the centered window reads
        // NEIGHBOUR rows, so every bx must exist before any output row is computed
        for (int s = 0; s < seqLen; s++) {
            int tmpOff = s * SHORTCONV_PARTS * dim;
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
        for (int g = 0; g < segLen.length; g++) {
            int r0 = segRow0[g], rEnd = r0 + segLen[g];
            for (int s = r0; s < rEnd; s++) {
                int tmpOff = s * SHORTCONV_PARTS * dim, outOff = s * dim;
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
                                                            + ((long) row * SHORTCONV_PARTS * dim
                                                                            + c)
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
        gemm(sc.outProj(), state.branchOut, dim, state.shortConvOut, dim, dim, seqLen, dim);
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
        int headSize = config.headSize;
        int queryDim = config.queryDim(), kvDim = config.kvDim(l);
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeadsPerLayer[l];
        attentionProject(state, l, seqLen);
        MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];
        float scale = 1.0f / (float) Math.sqrt(headSize);
        EmbedScratch es = state.embedScratch(config);
        for (int g = 0; g < segLen.length; g++) {
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
            Convert.copyF32(
                    es.segOut, 0, state.attnOut, (long) r0 * queryDim, (long) sl * queryDim);
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
     * The ColBERT per-token read for one retained row (LFM2.5-ColBERT): final-norm, {@code dense_2}
     * projection to {@code embeddingLengthOut}, L2-normalized - what llama.cpp's {@code
     * build_dense_out} does to {@code t_embd}, plus the client-side normalize the reference stack
     * applies before MaxSim. The returned view is a REUSED per-state buffer (the {@link #pool}
     * contract): valid until the next {@code colbertRow}/{@code pool} call - the caller copies it
     * out per row. (The ColBERT face class itself is not part of this slice.)
     */
    MemoryView<?> colbertRow(State s, int row) {
        int dim = configuration.embeddingLength;
        int outDim = configuration.embeddingLengthOut;
        EmbedScratch es = s.embedScratch(configuration);
        Norms.rmsnorm(
                es.embOut,
                0,
                s.residual,
                (long) row * dim,
                weights.finalNorm,
                dim,
                configuration.rmsNormEps);
        gemv(weights.dense2(), es.embOut, es.colbertOut, outDim, dim);
        float inv = l2Inv(es.colbertOut, outDim);
        Ops.mapInPlace(es.colbertOut, 0, outDim, v -> v * inv);
        return es.colbertOut;
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

    // === gemm/gemv shims: one place that owns the model-side matmul contract ===
    // NOT BLAS dgemm: this is llama.cpp's ggml_mul_mat(w, a) worldview, c = w · aᵀ, with
    //   w [m, k]  the weight (out_features x in_features, as the GGUF lays it out),
    //   a [n, k]  the activations (batch rows x in_features),
    //   c [n, m]  the result (batch rows x out_features).
    // - trailing (m, n, k) = (w rows = output width, a rows = batch, contraction) - exactly
    //   MatMul.mm's and JAM's order; no swap anywhere. (BLAS/ONNX-pilled readers: m is the
    //   WEIGHT rows here, ggml's assignment - not dgemm's m = activation rows.)
    // - wStride = k is hardcoded: this model's weight views are dense contiguous rows - a fact,
    //   not a per-call-site choice.
    // (heritage: the old FloatTensor virtuals w.gemm/w.matmul, resolved to MatMul.mm)

    /** {@code c = w · aᵀ} for each of the n activation rows: out is m wide, contraction k. */
    private static void gemm(
            MemoryView<MemorySegment> w,
            MemoryView<MemorySegment> a,
            int aStride,
            MemoryView<MemorySegment> c,
            int cStride,
            int m,
            int n,
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
            int m,
            int n,
            int k) {
        MatMul.mm(w, wOff, k, a, 0, aStride, c, 0, cStride, m, n, k);
    }

    /**
     * {@code c = w · a}, one row: mm's n==1 arm routes this to the decode path - no policy here.
     */
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
            int headSize,
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

        public int queryDim() {
            return numberOfHeads * headSize;
        }

        public int kvDim(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] * headSize;
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

        /**
         * The residual stream every block adds back into (old jinfer {@code residual}; llama.cpp
         * {@code inpL}, with {@code inpSA}/{@code inpFF} its per-sublayer checkpoints).
         */
        final MemoryView<MemorySegment> residual;

        /**
         * Pre-norm output - the input of EVERY projection (wq/wk/wv, conv in_proj, FFN gate/up, MoE
         * router); second life as the FFN branch output (down-proj destination, post-FFN norm)
         * before the residual add (old jinfer {@code xb}; llama.cpp {@code cur} right after the
         * norm).
         */
        final MemoryView<MemorySegment> normed;

        /**
         * The attention/conv branch's output: attention's o_proj destination (post-attn-norm
         * candidate) or the conv FIR scan's output (out-proj input) - normed and added to the
         * residual (old jinfer {@code xb2}; llama.cpp {@code cur} after the o_proj/conv block).
         */
        final MemoryView<MemorySegment> branchOut;

        /**
         * Flash-attention result, all heads concatenated, pre-o_proj (old jinfer {@code xbK};
         * llama.cpp {@code kqv_out}).
         */
        final MemoryView<MemorySegment> attnOut;

        /**
         * FFN gate projection; post silu-multiply the gated hidden (old jinfer {@code hb}; the gate
         * leg of llama.cpp's build_ffn).
         */
        final MemoryView<MemorySegment> hidden;

        /**
         * FFN up projection, silu-multiplied into {@link #hidden} (old jinfer {@code hb2}; the up
         * leg of llama.cpp's build_ffn).
         */
        final MemoryView<MemorySegment> hidden2;

        /**
         * Q projection, per-head normed + RoPE'd in place (old jinfer {@code query}; llama.cpp
         * {@code Qcur}).
         */
        final MemoryView<MemorySegment> query;

        /** The vocab projection of the retained row(s) (llama.cpp's {@code logits} node). */
        final MemoryView<MemorySegment> logits;

        final MemoryView<MemorySegment> ropeCos, ropeSin;
        final FlashAttention.DecodeScratch decodeScratch =
                new FlashAttention.DecodeScratch(memoryArena());
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
            if (embedScratch == null)
                embedScratch = new EmbedScratch(config, batchCapacity, memoryArena());
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
            this.residual = Views.allocateF32(memoryArena(), c * dim);
            this.normed = Views.allocateF32(memoryArena(), c * dim);
            this.branchOut = Views.allocateF32(memoryArena(), c * dim);
            this.attnOut = Views.allocateF32(memoryArena(), c * maxQueryDim);
            this.query = Views.allocateF32(memoryArena(), c * maxQueryDim);
            this.hidden = Views.allocateF32(memoryArena(), c * maxHiddenDim);
            this.hidden2 = Views.allocateF32(memoryArena(), c * maxHiddenDim);
            this.logits = Views.allocateF32(memoryArena(), config.vocabularySize);
            // rotary values for the batch about to be ingested: sized by BATCH, never context
            this.ropeCos = Views.allocateF32(memoryArena(), c * (config.headSize / 2));
            this.ropeSin = Views.allocateF32(memoryArena(), c * (config.headSize / 2));
            this.shortConvTmp = Views.allocateF32(memoryArena(), c * SHORTCONV_PARTS * dim);
            this.shortConvOut = Views.allocateF32(memoryArena(), c * dim);
            int n = config.numberOfLayers;
            this.keyCache = new MemoryView[n];
            this.valueCache = new MemoryView[n];
            this.batchK = new MemoryView[n];
            this.batchV = new MemoryView[n];
            this.shortConvState = new MemoryView[n];
            int hist = Math.max(config.shortConvLCache - 1, 0);
            for (int l = 0; l < n; l++) {
                if (config.isRecurrentLayer(l)) {
                    shortConvState[l] = Views.allocateF32(memoryArena(), hist * dim);
                } else {
                    int kvDim = config.kvDim(l);
                    keyCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
                    valueCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
                    batchK[l] = Views.allocateF32(memoryArena(), c * kvDim);
                    batchV[l] = Views.allocateF32(memoryArena(), c * kvDim);
                }
            }
            if (config.isMoE()) {
                int e = config.expertCount, tk = config.expertUsedCount;
                this.moeRouterB = Views.allocateF32(memoryArena(), c * e);
                this.moeGather = Views.allocateF32(memoryArena(), c * dim);
                this.moeDownB = Views.allocateF32(memoryArena(), c * dim);
                this.moeOutB = Views.allocateF32(memoryArena(), c * dim);
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

        /** Position per row, and first row per segment (refilled per forwardSegmented). */
        final int[] posOf, segRow0;

        EmbedScratch(
                Configuration config, int batchCapacity, MemoryAllocator<MemorySegment> memory) {
            int queryDim = config.queryDim(), kvDim = config.maxKvDim();
            this.segQ = Views.allocateF32(memory, batchCapacity * queryDim);
            this.segOut = Views.allocateF32(memory, batchCapacity * queryDim);
            this.segK = Views.allocateF32(memory, batchCapacity * kvDim);
            this.segV = Views.allocateF32(memory, batchCapacity * kvDim);
            this.embOut = Views.allocateF32(memory, config.embeddingLength());
            this.colbertOut = Views.allocateF32(memory, Math.max(1, config.embeddingLengthOut()));
            this.posOf = new int[batchCapacity];
            this.segRow0 = new int[batchCapacity]; // a segment per row is the densest packing
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
        int headSize = embeddingLength / numberOfHeads;
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
            kvHeads[i] = kWeight != null ? Math.toIntExact(kWeight.shape()[1]) / headSize : 0;
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
                        headSize,
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

    private static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        return Objects.requireNonNull(tensors.get(name), name);
    }

    /** F32 view by name (dtype checked AT LOAD, the old toF32Tensor fail-fast), or throw. */
    private static MemoryView<MemorySegment> requireF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> v = require(tensors, name);
        Views.requireDatatype(v, DataType.FP32, name);
        return v;
    }

    private static MemoryView<MemorySegment> find(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        return tensors.get(name);
    }

    private static MemoryView<MemorySegment> findF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> v = find(tensors, name);
        if (v != null) Views.requireDatatype(v, DataType.FP32, name);
        return v;
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors, Configuration config) {
        int n = config.numberOfLayers;
        RoPE.Schedule rope = RoPE.plain(config.headSize, config.ropeTheta);

        MemoryView<MemorySegment> tokenEmbeddings = require(tensors, "token_embd.weight");
        MemoryView<MemorySegment> wcls =
                tensors.containsKey("output.weight")
                        ? require(tensors, "output.weight")
                        : tokenEmbeddings; // tied embeddings
        // LFM2.5 names the final norm token_embd_norm (no separate output_norm); embeddings are
        // tied.
        MemoryView<MemorySegment> finalNorm =
                requireF32(
                        tensors,
                        tensors.containsKey("output_norm.weight")
                                ? "output_norm.weight"
                                : "token_embd_norm.weight");

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            MemoryView<MemorySegment> attnNorm = requireF32(tensors, p + "attn_norm.weight");
            MemoryView<MemorySegment> postAttnNorm =
                    findF32(tensors, p + "post_attention_norm.weight");
            MemoryView<MemorySegment> ffnNorm = requireF32(tensors, p + "ffn_norm.weight");
            MemoryView<MemorySegment> postFfnNorm = findF32(tensors, p + "post_ffw_norm.weight");

            AttentionWeights attention = null;
            ShortConvWeights shortConv = null;
            if (config.isRecurrentLayer(i)) {
                shortConv =
                        new ShortConvWeights(
                                requireF32(tensors, p + "shortconv.conv.weight"),
                                require(tensors, p + "shortconv.in_proj.weight"),
                                require(tensors, p + "shortconv.out_proj.weight"));
            } else {
                attention =
                        new AttentionWeights(
                                require(tensors, p + "attn_q.weight"),
                                require(tensors, p + "attn_k.weight"),
                                find(tensors, p + "attn_v.weight"),
                                require(tensors, p + "attn_output.weight"),
                                requireF32(tensors, p + "attn_q_norm.weight"),
                                requireF32(tensors, p + "attn_k_norm.weight"));
            }

            DenseFfnWeights dense = null;
            MoeFfnWeights moe = null;
            if (config.isMoELayer(i)) {
                moe =
                        new MoeFfnWeights(
                                require(tensors, p + "ffn_gate_inp.weight"),
                                require(tensors, p + "ffn_gate_exps.weight"),
                                require(tensors, p + "ffn_up_exps.weight"),
                                require(tensors, p + "ffn_down_exps.weight"),
                                findF32(tensors, p + "exp_probs_b.bias"));
            } else {
                dense =
                        new DenseFfnWeights(
                                require(tensors, p + "ffn_gate.weight"),
                                require(tensors, p + "ffn_up.weight"),
                                require(tensors, p + "ffn_down.weight"));
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
                tensors.containsKey("dense_2.weight") ? require(tensors, "dense_2.weight") : null;
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls, dense2);
    }
}
