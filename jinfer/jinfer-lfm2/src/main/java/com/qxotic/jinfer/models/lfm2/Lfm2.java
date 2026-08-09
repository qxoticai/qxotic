// LFM2.5 (Liquid Foundation Model 2.5) against the com.qxotic.jinfer.models model API: a port of
// the
// production
// jinfer LFM2.5 forward (the hybrid in Llama.java). Each layer is EITHER GQA attention (kv-heads >
// 0)
// OR a gated short-convolution mixer (kv-heads == 0); the FFN is EITHER dense SwiGLU OR top-k MoE.
// Text-only (no media encoders, no MTP heads) so this implements only LanguageModel. Mirrors the
// Gemma4
// port's API decomposition (embed / layer / attention / feedForward / commitKv); the deltas vs
// Gemma4:
// no embedding scale, SiLU-GLU (not GeLU), attention scale = 1/sqrt(headSize), no V-norm, no SWA /
// no
// shared-KV / no per-layer-embeddings, and a rolling shortConvState alongside the KV cache (forked
// too).
package com.qxotic.jinfer.models.lfm2;

import static com.qxotic.jinfer.Norms.rmsnorm;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.*;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;

public final class Lfm2
        implements LanguageModel<Lfm2.Configuration, Lfm2.Weights, Lfm2.State>,
                EmbeddingModel<Lfm2.Configuration, Lfm2.Weights, Lfm2.State> {

    /** llama.cpp's pooling_type enum value for CLS - pool the sequence's FIRST row (its BOS). */
    static final int POOLING_CLS = 2;

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final String chatTemplateSource;
    private final byte[] modelSeed;
    private final Weights weights;

    Lfm2(
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
        State state = new State(configuration, contextCapacity, batchCapacity, arena);
        return state;
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
            case Batch.Input.Embeddings e ->
                    throw new UnsupportedOperationException(
                            "LFM2.5 is text-only: embedding input is not supported");
        }
        s.advance(n, batch.outputs());
    }

    @Override
    public FloatTensor head(State s, int output) {
        int dim = configuration.embeddingLength;
        int row = s.lastChunkLen - s.outputCount + output;
        return Parallel.onDecodePool(
                () -> {
                    rmsnorm(
                            s.xb,
                            0,
                            s.residual,
                            (long) row * dim,
                            weights.finalNorm,
                            dim,
                            configuration.rmsNormEps);
                    weights.wcls.matmul(s.xb, s.logits, configuration.vocabularySize, dim);
                    Activations.softcap(
                            s.logits,
                            0,
                            configuration.vocabularySize,
                            configuration.logitSoftcapping);
                    return s.logits;
                });
    }

    /** The turn-delimiter / eos ids that terminate generation (convenience for callers/tests). */
    public Set<Integer> stopTokens() {
        return SpecialTokens.stops(
                tokenizer, -1, "<|im_end|>", "<eos>", "<|endoftext|>", "<end_of_turn>");
    }

    private Lfm2ChatTemplate
            chatTemplate; // memoized: stateless, model-lifetime (pins any construction-time state)

    /**
     * This model bundled with the three text facts its GGUF carries - what an
     * architecture-dispatching loader hands to a caller that does not know the family.
     */
    public LoadedModel<Lfm2.State> loaded() {
        return new LoadedModel<>(
                this,
                tokenizer(),
                chatTemplateSource,
                stopTokens(),
                modelSeed,
                Optional.of(template()),
                LoadedModel.SamplingDefaults.NONE);
    }

    /** The per-turn view of the same template (turn-aligned cache scenarios refine through it). */
    public Optional<TurnTemplate> turnTemplate() {
        return Optional.of(template());
    }

    private Lfm2ChatTemplate template() {
        if (chatTemplate == null) {
            // the 2.6B-era template's generation prompt is "<|im_start|>assistant\n<think>" -
            // detect the pre-opened think span from the checkpoint's own template source
            boolean opensThink =
                    chatTemplateSource != null && chatTemplateSource.contains("assistant\n<think>");
            chatTemplate = new Lfm2ChatTemplate(tokenizer(), opensThink);
        }
        return chatTemplate;
    }

    @Override
    public Optional<StateCodec<Lfm2.State>> stateCodec() {
        return Optional.of(new Lfm2StateCodec(config()));
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
        weights.tokenEmbeddings.safetyCanary(); // fail-fast on freed weights, before raw reads
        int dim = configuration.embeddingLength;
        for (int s = 0; s < seqLen; s++) {
            weights.tokenEmbeddings.copyTo(
                    (long) tokens[tokenOffset + s] * dim, state.residual, (long) s * dim, dim);
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
        F32FloatTensor preNorm =
                weights.layers[l].attnNorm(); // conv layers use attn_norm as the mixer pre-norm
        Parallel.forRows(
                seqLen,
                s ->
                        rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                preNorm,
                                dim,
                                eps));
        sc.inProj().gemm(state.xb, dim, state.shortConvTmp, 3 * dim, seqLen, 3 * dim, dim);
        shortConvScan(state, l, seqLen);
        sc.outProj().gemm(state.xb2, dim, state.shortConvOut, dim, seqLen, dim, dim);
        state.residual.addInPlace(0, state.shortConvOut, 0, seqLen * dim);
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
        F32FloatTensor kernel =
                weights.layers[l].shortConv().kernel(); // per channel: dConv taps at c*dConv + k
        FloatTensor convState = state.shortConvState[l];
        FloatTensor tmp = state.shortConvTmp, out = state.xb2;
        for (int s = 0; s < seqLen; s++) {
            int tmpOff = s * 3 * dim, outOff = s * dim;
            for (int c = 0; c < dim; c++) {
                float b = tmp.getFloat(tmpOff + c);
                float cg = tmp.getFloat(tmpOff + dim + c);
                float xv = tmp.getFloat(tmpOff + 2 * dim + c);
                float bx = b * xv;
                tmp.setFloat(tmpOff + c, bx);
                int kBase = c * dConv;
                float sum = 0f;
                for (int k = 0; k < hist; k++)
                    sum += convState.getFloat((long) k * dim + c) * kernel.getFloat(kBase + k);
                sum += bx * kernel.getFloat(kBase + dConv - 1);
                out.setFloat(outOff + c, cg * sum);
                for (int k = 0; k < hist - 1; k++)
                    convState.setFloat(
                            (long) k * dim + c, convState.getFloat((long) (k + 1) * dim + c));
                if (hist > 0) convState.setFloat((long) (hist - 1) * dim + c, bx);
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
        FloatTensor bK = state.batchK[l], bV = state.batchV[l];
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
                    (F32FloatTensor) state.query,
                    (F32FloatTensor) state.xbK,
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

        F32FloatTensor attNormW = weights.layers[l].attnNorm();
        Parallel.forRows(
                seqLen,
                s ->
                        rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                attNormW,
                                dim,
                                eps));
        attn.wq().gemm(state.xb, dim, state.query, queryDim, seqLen, queryDim, dim);
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
        FloatTensor bK = state.batchK[l], bV = state.batchV[l];
        attn.wk().gemm(state.xb, dim, bK, kvDim, seqLen, kvDim, dim);
        if (attn.wv() != null) attn.wv().gemm(state.xb, dim, bV, kvDim, seqLen, kvDim, dim);
        else bK.copyTo(0, bV, 0, seqLen * kvDim);
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
        attn.wo()
                .gemm(
                        state.xbK,
                        configuration.queryDim(),
                        state.xb2,
                        dim,
                        seqLen,
                        dim,
                        configuration.queryDim());
        F32FloatTensor postAttW = weights.layers[l].postAttnNorm();
        if (postAttW != null)
            Parallel.forRows(
                    seqLen,
                    s ->
                            rmsnorm(
                                    state.xb2,
                                    (long) s * dim,
                                    state.xb2,
                                    (long) s * dim,
                                    postAttW,
                                    dim,
                                    eps));
        state.residual.addInPlace(0, state.xb2, 0, seqLen * dim);
    }

    /** Per-head RMS-norm then NeoX RoPE over each row (shared by Q and K). */
    private void headNormRope(
            FloatTensor t,
            int rowStride,
            int nHeads,
            int headSize,
            int halfHead,
            F32FloatTensor normW,
            int seqLen,
            FloatTensor cos,
            FloatTensor sin) {
        float eps = configuration.rmsNormEps;
        Parallel.forRows(
                seqLen,
                s -> {
                    for (int h = 0; h < nHeads; h++) {
                        long off = (long) s * rowStride + (long) h * headSize;
                        rmsnorm(t, off, t, off, normW, headSize, eps);
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
        F32FloatTensor ffnNormW = weights.layers[l].ffnNorm(),
                postFfwW = weights.layers[l].postFfnNorm();
        Parallel.forRows(
                seqLen,
                s ->
                        rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                ffnNormW,
                                dim,
                                eps));
        ffn.gate().gemm(state.xb, dim, state.hb, hiddenDim, seqLen, hiddenDim, dim);
        ffn.up().gemm(state.xb, dim, state.hb2, hiddenDim, seqLen, hiddenDim, dim);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.siluMultiply(
                                state.hb, s * hiddenDim, state.hb2, s * hiddenDim, hiddenDim));
        ffn.down().gemm(state.hb, hiddenDim, state.xb, dim, seqLen, dim, hiddenDim);
        if (postFfwW != null)
            Parallel.forRows(
                    seqLen,
                    s ->
                            rmsnorm(
                                    state.xb,
                                    (long) s * dim,
                                    state.xb,
                                    (long) s * dim,
                                    postFfwW,
                                    dim,
                                    eps));
        state.residual.addInPlace(0, state.xb, 0, seqLen * dim);
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
        F32FloatTensor ffnNormW = weights.layers[l].ffnNorm(),
                postFfnNorm = weights.layers[l].postFfnNorm();

        // pre-norm into xb, then route on it
        Parallel.forRows(
                seqLen,
                s ->
                        rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                ffnNormW,
                                dim,
                                eps));
        moe.router().gemm(state.xb, dim, state.moeRouterB, nExperts, seqLen, nExperts, dim);

        int[] counts = state.moeExpertCounts;
        Arrays.fill(counts, 0);
        for (int s = 0; s < seqLen; s++) {
            long ro = (long) s * nExperts;
            if (moe.expProbsBias() != null) {
                for (int i = 0; i < nExperts; i++)
                    state.moeRouterB.setFloat(
                            ro + i,
                            state.moeRouterB.getFloat(ro + i) + moe.expProbsBias().getFloat(i));
            }
            if (config.expertGatingFunc == 2)
                state.moeRouterB.mapInPlace(
                        ro, nExperts, v -> (float) (1.0 / (1.0 + Math.exp(-v))));
            else state.moeRouterB.softmaxInPlace(ro, nExperts);
            for (int ki = 0; ki < topK; ki++) {
                int best = 0;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int ei = 0; ei < nExperts; ei++) {
                    float v = state.moeRouterB.getFloat(ro + ei);
                    if (v > bestVal) {
                        bestVal = v;
                        best = ei;
                    }
                }
                state.moeRowTopE[s * topK + ki] = best;
                state.moeRowTopP[s * topK + ki] = bestVal;
                state.moeRouterB.setFloat(ro + best, Float.NEGATIVE_INFINITY);
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
                    moe.gateExps()
                            .gemm(
                                    gather,
                                    dim,
                                    state.hb,
                                    expertFF,
                                    n,
                                    expertFF,
                                    dim,
                                    (long) e * expertFF * dim);
                    moe.upExps()
                            .gemm(
                                    gather,
                                    dim,
                                    state.hb2,
                                    expertFF,
                                    n,
                                    expertFF,
                                    dim,
                                    (long) e * expertFF * dim);
                    Parallel.forRows(
                            n,
                            j ->
                                    Activations.siluMultiply(
                                            state.hb,
                                            j * expertFF,
                                            state.hb2,
                                            j * expertFF,
                                            expertFF));
                    moe.downExps()
                            .gemm(
                                    state.hb,
                                    expertFF,
                                    out,
                                    dim,
                                    n,
                                    dim,
                                    expertFF,
                                    (long) e * dim * expertFF);
                });

        Parallel.forRows(
                seqLen,
                s -> {
                    if (postFfnNorm != null)
                        rmsnorm(
                                state.moeOutB,
                                (long) s * dim,
                                state.moeOutB,
                                (long) s * dim,
                                postFfnNorm,
                                dim,
                                eps);
                    state.residual.addInPlace((long) s * dim, state.moeOutB, (long) s * dim, dim);
                });
    }

    /** Write the chunk's K/V into the (linear) cache for attention layers. */
    private void commitKv(State state, int startPos, int seqLen) {
        for (int l = 0; l < configuration.numberOfLayers; l++) {
            if (state.keyCache[l] == null) continue; // recurrent layer
            int kvDim = configuration.kvDim(l);
            for (int s = 0; s < seqLen; s++) {
                long kvPos = startPos + s;
                state.batchK[l].copyTo((long) s * kvDim, state.keyCache[l], kvPos * kvDim, kvDim);
                state.batchV[l].copyTo((long) s * kvDim, state.valueCache[l], kvPos * kvDim, kvDim);
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
        F32FloatTensor preNorm = weights.layers[l].attnNorm();
        Parallel.forRows(
                seqLen,
                s ->
                        rmsnorm(
                                state.xb,
                                (long) s * dim,
                                state.residual,
                                (long) s * dim,
                                preNorm,
                                dim,
                                eps));
        sc.inProj().gemm(state.xb, dim, state.shortConvTmp, 3 * dim, seqLen, 3 * dim, dim);

        int dConv = configuration.shortConvLCache, pad = (dConv - 1) / 2;
        F32FloatTensor kernel = sc.kernel(); // per channel: dConv taps at c*dConv + k
        FloatTensor tmp = state.shortConvTmp, out = state.xb2;
        // materialize bx = B*x in place over the B block first: the centered window reads
        // NEIGHBOUR rows, so every bx must exist before any output row is computed
        for (int s = 0; s < seqLen; s++) {
            int tmpOff = s * 3 * dim;
            for (int c = 0; c < dim; c++) {
                tmp.setFloat(
                        tmpOff + c, tmp.getFloat(tmpOff + c) * tmp.getFloat(tmpOff + 2 * dim + c));
            }
        }
        for (int g = 0; g < segRow0.length; g++) {
            int r0 = segRow0[g], rEnd = r0 + segLen[g];
            for (int s = r0; s < rEnd; s++) {
                int tmpOff = s * 3 * dim, outOff = s * dim;
                for (int c = 0; c < dim; c++) {
                    float cg = tmp.getFloat(tmpOff + dim + c);
                    int kBase = c * dConv;
                    float sum = 0f;
                    for (int k = 0; k < dConv; k++) {
                        int row = s - pad + k; // zero beyond this sequence's own edges
                        if (row >= r0 && row < rEnd) {
                            sum += tmp.getFloat(row * 3 * dim + c) * kernel.getFloat(kBase + k);
                        }
                    }
                    out.setFloat(outOff + c, cg * sum);
                }
            }
        }
        sc.outProj().gemm(state.xb2, dim, state.shortConvOut, dim, seqLen, dim, dim);
        state.residual.addInPlace(0, state.shortConvOut, 0, seqLen * dim);
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
        FloatTensor bK = state.batchK[l], bV = state.batchV[l];
        float scale = 1.0f / (float) Math.sqrt(headSize);
        EmbedScratch es = state.embedScratch(config);
        for (int g = 0; g < segRow0.length; g++) {
            int r0 = segRow0[g], sl = segLen[g];
            state.query.copyTo((long) r0 * queryDim, es.segQ, 0, sl * queryDim);
            bK.copyTo((long) r0 * kvDim, es.segK, 0, sl * kvDim);
            bV.copyTo((long) r0 * kvDim, es.segV, 0, sl * kvDim);
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
            es.segOut.copyTo(0, state.xbK, (long) r0 * queryDim, sl * queryDim);
        }
        attentionFinish(state, l, seqLen);
    }

    /**
     * The sentence embedding: final-norm the pooled row, L2-normalize - CLS pooling reads the
     * sequence's FIRST retained row (its BOS). {@code index} addresses retained rows exactly as
     * {@code logits}' output does. The returned tensor is a REUSED per-state buffer.
     */
    @Override
    public FloatTensor pool(State s, int index) {
        int dim = configuration.embeddingLength;
        int row = s.lastChunkLen - s.outputCount + index;
        FloatTensor out = s.embedScratch(configuration).embOut;
        rmsnorm(
                out,
                0,
                s.residual,
                (long) row * dim,
                weights.finalNorm,
                dim,
                configuration.rmsNormEps);
        float inv = l2Inv(out, dim);
        out.mapInPlace(0, dim, v -> v * inv);
        return out;
    }

    /** {@code 1/||t[0..n)||}, or 0 for a zero vector - the shared L2-normalization factor. */
    private static float l2Inv(FloatTensor t, int n) {
        float ss = Norms.sumOfSquares(t, 0, n);
        return ss > 0 ? (float) (1.0 / Math.sqrt(ss)) : 0f;
    }

    /**
     * The ColBERT per-token read for one retained row: final-norm, {@code dense_2} projection to
     * {@code embeddingLengthOut}, L2-normalized - what llama.cpp's {@code build_dense_out} does to
     * {@code t_embd}, plus the client-side normalize the reference stack applies before MaxSim.
     * Package-private for {@link Lfm2Colbert}; {@code out} is the caller's buffer.
     */
    void colbertRow(State s, int row, float[] out) {
        int dim = configuration.embeddingLength;
        int outDim = configuration.embeddingLengthOut;
        FloatTensor normed = s.embedScratch(configuration).embOut;
        rmsnorm(
                normed,
                0,
                s.residual,
                (long) row * dim,
                weights.finalNorm,
                dim,
                configuration.rmsNormEps);
        FloatTensor projected = s.embedScratch(configuration).colbertOut;
        weights.dense2().matmul(normed, projected, outDim, dim);
        float inv = l2Inv(projected, outDim);
        for (int i = 0; i < outDim; i++) out[i] = projected.getFloat(i) * inv;
    }

    /**
     * Bidirectional embedding overrides the generic chunk-streaming default: a sequence attends to
     * ALL of its tokens, so it must be forwarded WHOLE - {@link #forEachSequence} re-cuts groups on
     * sequence boundaries. Emits each sequence's CLS (first-row) embedding, in input order.
     */
    @Override
    public void embed(State state, Batch.Input.Sequences seqs, Consumer<FloatTensor> sink) {
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
            FloatTensor wq,
            FloatTensor wk,
            FloatTensor wv,
            FloatTensor wo,
            F32FloatTensor qNorm,
            F32FloatTensor kNorm) {}

    /** {@code kernel}: per-channel dConv taps (c*dConv + k), as the GGUF lays them out. */
    public record ShortConvWeights(
            F32FloatTensor kernel, FloatTensor inProj, FloatTensor outProj) {}

    public record DenseFfnWeights(FloatTensor gate, FloatTensor up, FloatTensor down) {}

    public record MoeFfnWeights(
            FloatTensor router,
            FloatTensor gateExps,
            FloatTensor upExps,
            FloatTensor downExps,
            F32FloatTensor expProbsBias) {}

    public record LayerWeights(
            F32FloatTensor attnNorm,
            F32FloatTensor postAttnNorm,
            F32FloatTensor ffnNorm,
            F32FloatTensor postFfnNorm,
            AttentionWeights attention,
            ShortConvWeights shortConv,
            DenseFfnWeights dense,
            MoeFfnWeights moe) {}

    public record Weights(
            FloatTensor tokenEmbeddings,
            LayerWeights[] layers,
            F32FloatTensor finalNorm,
            RoPE.Schedule rope,
            FloatTensor wcls,
            FloatTensor dense2) {} // ColBERT's per-token projection; null elsewhere

    // === State ===

    public static final class State extends BaseState {
        final int contextCapacity, batchCapacity;
        final FloatTensor residual, xb, xbK, xb2, hb, hb2, query, logits;
        final FloatTensor ropeCos, ropeSin;
        final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch(arena);
        final FloatTensor[] keyCache,
                valueCache,
                batchK,
                batchV; // per layer; null on recurrent layers
        final FloatTensor[] shortConvState; // per layer; null on attention layers
        final FloatTensor shortConvTmp, shortConvOut;
        // MoE scratch (chunk-wide CSR routing); allocated only when the model has experts, else
        // null.
        final FloatTensor moeRouterB, moeGather, moeDownB, moeOutB;
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
            for (FloatTensor conv : shortConvState) {
                if (conv != null) {
                    conv.fillInPlace(0, Math.toIntExact(conv.size()), 0f);
                }
            }
        }

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
            this.residual = FloatTensor.allocateF32(arena, c * dim);
            this.xb = FloatTensor.allocateF32(arena, c * dim);
            this.xb2 = FloatTensor.allocateF32(arena, c * dim);
            this.xbK = FloatTensor.allocateF32(arena, c * maxQueryDim);
            this.query = FloatTensor.allocateF32(arena, c * maxQueryDim);
            this.hb = FloatTensor.allocateF32(arena, c * maxHiddenDim);
            this.hb2 = FloatTensor.allocateF32(arena, c * maxHiddenDim);
            this.logits = FloatTensor.allocateF32(arena, config.vocabularySize);
            // rotary values for the batch about to be ingested: sized by BATCH, never context
            this.ropeCos = FloatTensor.allocateF32(arena, c * (config.headSizeFull / 2));
            this.ropeSin = FloatTensor.allocateF32(arena, c * (config.headSizeFull / 2));
            this.shortConvTmp = FloatTensor.allocateF32(arena, c * 3 * dim);
            this.shortConvOut = FloatTensor.allocateF32(arena, c * dim);
            int n = config.numberOfLayers;
            this.keyCache = new FloatTensor[n];
            this.valueCache = new FloatTensor[n];
            this.batchK = new FloatTensor[n];
            this.batchV = new FloatTensor[n];
            this.shortConvState = new FloatTensor[n];
            int hist = Math.max(config.shortConvLCache - 1, 0);
            for (int l = 0; l < n; l++) {
                if (config.isRecurrentLayer(l)) {
                    shortConvState[l] = FloatTensor.allocateF32(arena, hist * dim);
                } else {
                    int kvDim = config.kvDim(l);
                    keyCache[l] = FloatTensor.allocateF16(arena, contextCapacity, kvDim);
                    valueCache[l] = FloatTensor.allocateF16(arena, contextCapacity, kvDim);
                    batchK[l] = FloatTensor.allocateF32(arena, c * kvDim);
                    batchV[l] = FloatTensor.allocateF32(arena, c * kvDim);
                }
            }
            if (config.isMoE()) {
                int e = config.expertCount, tk = config.expertUsedCount;
                this.moeRouterB = FloatTensor.allocateF32(arena, c * e);
                this.moeGather = FloatTensor.allocateF32(arena, c * dim);
                this.moeDownB = FloatTensor.allocateF32(arena, c * dim);
                this.moeOutB = FloatTensor.allocateF32(arena, c * dim);
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
        final FloatTensor segQ, segK, segV, segOut, embOut, colbertOut;

        EmbedScratch(Configuration config, int batchCapacity, Arena arena) {
            int queryDim = config.queryDim(), kvDim = config.maxKvDim();
            this.segQ = FloatTensor.allocateF32(arena, batchCapacity * queryDim);
            this.segOut = FloatTensor.allocateF32(arena, batchCapacity * queryDim);
            this.segK = FloatTensor.allocateF32(arena, batchCapacity * kvDim);
            this.segV = FloatTensor.allocateF32(arena, batchCapacity * kvDim);
            this.embOut = FloatTensor.allocateF32(arena, config.embeddingLength());
            this.colbertOut =
                    FloatTensor.allocateF32(arena, Math.max(1, config.embeddingLengthOut()));
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
     * As above with a caller-supplied tokenizer; null = the GGUF's own (see Models for the
     * contract).
     */
    public static Lfm2 loadModel(
            FileChannel fileChannel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        byte[] seed = PromptCache.modelSeed(fileChannel);
        if (tokenizer == null) {
            tokenizer = Tokenizers.fromGGUF(gguf);
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
        // attention
        // layers derive it from the K-projection's row count.
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

        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf, arena);
        return new Lfm2(
                config,
                tokenizer,
                Tokenizers.chatTemplateSource(gguf),
                seed,
                loadWeights(tensors, config, arena));
    }

    static Weights loadWeights(
            Map<String, GGMLTensorEntry> tensors, Configuration config, Arena arena) {
        int n = config.numberOfLayers;
        RoPE.Schedule rope = RoPE.plain(config.headSizeFull, config.ropeTheta);

        FloatTensor tokenEmbeddings = ModelLoader.loadQuantized(tensors.get("token_embd.weight"));
        FloatTensor wcls =
                tensors.containsKey("output.weight")
                        ? ModelLoader.loadQuantized(tensors.get("output.weight"))
                        : tokenEmbeddings; // tied embeddings
        // LFM2.5 names the final norm token_embd_norm (no separate output_norm); embeddings are
        // tied.
        F32FloatTensor finalNorm =
                ModelLoader.toF32Tensor(
                        tensors.containsKey("output_norm.weight")
                                ? tensors.get("output_norm.weight")
                                : tensors.get("token_embd_norm.weight"));

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            F32FloatTensor attnNorm = ModelLoader.toF32Tensor(tensors.get(p + "attn_norm.weight"));
            F32FloatTensor postAttnNorm =
                    ModelLoader.f32OrNull(tensors, p + "post_attention_norm.weight");
            F32FloatTensor ffnNorm = ModelLoader.toF32Tensor(tensors.get(p + "ffn_norm.weight"));
            F32FloatTensor postFfnNorm = ModelLoader.f32OrNull(tensors, p + "post_ffw_norm.weight");

            AttentionWeights attention = null;
            ShortConvWeights shortConv = null;
            if (config.isRecurrentLayer(i)) {
                shortConv =
                        new ShortConvWeights(
                                ModelLoader.toF32Tensor(tensors.get(p + "shortconv.conv.weight")),
                                ModelLoader.loadQuantized(
                                        tensors.get(p + "shortconv.in_proj.weight")),
                                ModelLoader.loadQuantized(
                                        tensors.get(p + "shortconv.out_proj.weight")));
            } else {
                attention =
                        new AttentionWeights(
                                ModelLoader.loadQuantized(tensors.get(p + "attn_q.weight")),
                                ModelLoader.loadQuantized(tensors.get(p + "attn_k.weight")),
                                ModelLoader.quantOrNull(tensors, p + "attn_v.weight"),
                                ModelLoader.loadQuantized(tensors.get(p + "attn_output.weight")),
                                ModelLoader.toF32Tensor(tensors.get(p + "attn_q_norm.weight")),
                                ModelLoader.toF32Tensor(tensors.get(p + "attn_k_norm.weight")));
            }

            DenseFfnWeights dense = null;
            MoeFfnWeights moe = null;
            if (config.isMoELayer(i)) {
                moe =
                        new MoeFfnWeights(
                                ModelLoader.loadQuantized(tensors.get(p + "ffn_gate_inp.weight")),
                                ModelLoader.loadQuantized(tensors.get(p + "ffn_gate_exps.weight")),
                                ModelLoader.loadQuantized(tensors.get(p + "ffn_up_exps.weight")),
                                ModelLoader.loadQuantized(tensors.get(p + "ffn_down_exps.weight")),
                                ModelLoader.f32OrNull(tensors, p + "exp_probs_b.bias"));
            } else {
                dense =
                        new DenseFfnWeights(
                                ModelLoader.loadQuantized(tensors.get(p + "ffn_gate.weight")),
                                ModelLoader.loadQuantized(tensors.get(p + "ffn_up.weight")),
                                ModelLoader.loadQuantized(tensors.get(p + "ffn_down.weight")));
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
        FloatTensor dense2 =
                tensors.containsKey("dense_2.weight")
                        ? ModelLoader.loadQuantized(tensors.get("dense_2.weight"))
                        : null;
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls, dense2);
    }
}
