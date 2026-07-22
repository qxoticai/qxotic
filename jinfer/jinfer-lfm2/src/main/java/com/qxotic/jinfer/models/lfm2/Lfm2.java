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
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class Lfm2 implements LanguageModel<Lfm2.Configuration, Lfm2.Weights, Lfm2.State> {

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
    public State newState(int contextCapacity, int batchCapacity) {
        State state = new State(configuration, contextCapacity, batchCapacity);
        return state;
    }

    @Override
    public void ingest(State s, com.qxotic.jinfer.Batch batch) {
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
            case com.qxotic.jinfer.Batch.Input.Tokens t -> {
                int[] ids = t.ids();
                if (n == 1)
                    Parallel.onDecodePool(
                            () -> {
                                forward(s, ids, 0, from, n);
                                return null;
                            });
                else forward(s, ids, 0, from, n);
            }
            case com.qxotic.jinfer.Batch.Input.Sequences seq ->
                    throw new UnsupportedOperationException(
                            "LFM2.5 is generative: packed sequences (batched embedding) not"
                                    + " supported");
            case com.qxotic.jinfer.Batch.Input.Embeddings e ->
                    throw new UnsupportedOperationException(
                            "LFM2.5 is text-only: embedding input is not supported");
        }
        s.advance(n, batch.outputs());
    }

    @Override
    public FloatTensor logits(State s, int output) {
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
        Set<Integer> stops = new HashSet<>();
        for (String name : new String[] {"<|im_end|>", "<eos>", "<|endoftext|>", "<end_of_turn>"}) {
            SpecialTokens.find(tokenizer, name).ifPresent(stops::add);
        }
        return stops;
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
                java.util.Optional.of(template()));
    }

    /** The per-turn view of the same template (turn-aligned cache scenarios refine through it). */
    public java.util.Optional<com.qxotic.jinfer.chat.TurnTemplate> turnTemplate() {
        return java.util.Optional.of(template());
    }

    private Lfm2ChatTemplate template() {
        if (chatTemplate == null) chatTemplate = new Lfm2ChatTemplate(tokenizer());
        return chatTemplate;
    }

    @Override
    public java.util.Optional<com.qxotic.jinfer.cache.StateCodec<Lfm2.State>> stateCodec() {
        return java.util.Optional.of(new Lfm2StateCodec(config()));
    }

    // === Forward ===

    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        embed(state, tokens, tokenOffset, seqLen);
        for (int l = 0; l < configuration.numberOfLayers; l++) layer(state, l, startPos, seqLen);
        commitKv(state, startPos, seqLen);
    }

    /** Token-embedding lookup into the residual stream (no scaling, unlike Gemma4). */
    private void embed(State state, int[] tokens, int tokenOffset, int seqLen) {
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
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;
        int headSize = config.headSizeFull, halfHead = headSize / 2;
        int queryDim = config.queryDim(), kvDim = config.kvDim(l);
        int nKvHeads = config.numberOfKeyValueHeadsPerLayer[l],
                kvMul = config.numberOfHeads / nKvHeads;
        AttentionWeights attn = weights.layers[l].attention();
        F32FloatTensor real = weights.ropeReal, imag = weights.ropeImag;

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
                startPos,
                seqLen,
                real,
                imag);

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
                startPos,
                seqLen,
                real,
                imag);

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

        attn.wo().gemm(state.xbK, queryDim, state.xb2, dim, seqLen, dim, queryDim);
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
            int startPos,
            int seqLen,
            F32FloatTensor real,
            F32FloatTensor imag) {
        float eps = configuration.rmsNormEps;
        int ctx = configuration.contextLength;
        Parallel.forRows(
                seqLen,
                s -> {
                    for (int h = 0; h < nHeads; h++) {
                        long off = (long) s * rowStride + (long) h * headSize;
                        rmsnorm(t, off, t, off, normW, headSize, eps);
                    }
                    int ropePos = Math.max(0, Math.min(ctx - 1, startPos + s));
                    RoPE.applyNeox(
                            t,
                            (long) s * rowStride,
                            nHeads,
                            headSize,
                            halfHead,
                            ropePos,
                            real,
                            imag);
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
            int expertGatingFunc)
            implements Config {
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

        public int maxKvDim() {
            int max = 0;
            for (int l = 0; l < numberOfLayers; l++) max = Math.max(max, kvDim(l));
            return max;
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
            F32FloatTensor ropeReal,
            F32FloatTensor ropeImag,
            FloatTensor wcls) {}

    // === State ===

    public static final class State extends com.qxotic.jinfer.BaseState {
        final int contextCapacity, batchCapacity;
        final FloatTensor residual, xb, xbK, xb2, hb, hb2, query, logits;
        final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch();
        final FloatTensor[] keyCache,
                valueCache,
                batchK,
                batchV; // per layer; null on recurrent layers
        final FloatTensor[] shortConvState; // per layer; null on attention layers
        final FloatTensor shortConvTmp, shortConvOut;
        // MoE scratch (chunk-wide CSR routing); allocated only when the model has experts, else
        // null.
        final FloatTensor moeRouterB, moeGather, moeDownB, moeOutB;
        final int[] moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeRowTopE;
        final float[] moeProbByExpert, moeRowTopP;
        final Moe.Routing moeRouting;

        State(Configuration config, int contextCapacity, int batchCapacity) {
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
            this.residual = FloatTensor.allocateF32(c * dim);
            this.xb = FloatTensor.allocateF32(c * dim);
            this.xb2 = FloatTensor.allocateF32(c * dim);
            this.xbK = FloatTensor.allocateF32(c * maxQueryDim);
            this.query = FloatTensor.allocateF32(c * maxQueryDim);
            this.hb = FloatTensor.allocateF32(c * maxHiddenDim);
            this.hb2 = FloatTensor.allocateF32(c * maxHiddenDim);
            this.logits = FloatTensor.allocateF32(config.vocabularySize);
            this.shortConvTmp = FloatTensor.allocateF32(c * 3 * dim);
            this.shortConvOut = FloatTensor.allocateF32(c * dim);
            int n = config.numberOfLayers;
            this.keyCache = new FloatTensor[n];
            this.valueCache = new FloatTensor[n];
            this.batchK = new FloatTensor[n];
            this.batchV = new FloatTensor[n];
            this.shortConvState = new FloatTensor[n];
            int hist = Math.max(config.shortConvLCache - 1, 0);
            for (int l = 0; l < n; l++) {
                if (config.isRecurrentLayer(l)) {
                    shortConvState[l] = FloatTensor.allocateF32(hist * dim);
                } else {
                    int kvDim = config.kvDim(l);
                    keyCache[l] = FloatTensor.allocateF16(contextCapacity, kvDim);
                    valueCache[l] = FloatTensor.allocateF16(contextCapacity, kvDim);
                    batchK[l] = FloatTensor.allocateF32(c * kvDim);
                    batchV[l] = FloatTensor.allocateF32(c * kvDim);
                }
            }
            if (config.isMoE()) {
                int e = config.expertCount, tk = config.expertUsedCount;
                this.moeRouterB = FloatTensor.allocateF32(c * e);
                this.moeGather = FloatTensor.allocateF32(c * dim);
                this.moeDownB = FloatTensor.allocateF32(c * dim);
                this.moeOutB = FloatTensor.allocateF32(c * dim);
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
                this.moeRouterB = this.moeGather = this.moeDownB = this.moeOutB = null;
                this.moeExpertCounts =
                        this.moeExpertOffsets =
                                this.moeCursor = this.moeRowByExpert = this.moeRowTopE = null;
                this.moeProbByExpert = this.moeRowTopP = null;
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

    // === Loading ===

    public static Lfm2 loadModel(Path ggufPath, int contextLength) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, contextLength, true);
        }
    }

    public static Lfm2 loadModel(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag)
            throws IOException {
        byte[] seed = com.qxotic.jinfer.cache.PromptCache.modelSeed(fileChannel);
        Tokenizer tokenizer = Tokenizers.fromGGUF(gguf);
        String arch = gguf.getString("general.architecture");

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength)
            contextLength = modelContextLength;

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
                        expertGatingFunc);

        if (!loadWeightsFlag)
            return new Lfm2(config, tokenizer, Tokenizers.chatTemplateSource(gguf), seed, null);
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        return new Lfm2(
                config,
                tokenizer,
                Tokenizers.chatTemplateSource(gguf),
                seed,
                loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
        int n = config.numberOfLayers;
        Pair<float[], float[]> rope =
                RoPE.precomputeFreqsCis(
                        config.contextLength(), config.headSizeFull, config.ropeTheta);
        F32FloatTensor ropeReal = F32FloatTensor.of(rope.first());
        F32FloatTensor ropeImag = F32FloatTensor.of(rope.second());

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
        return new Weights(tokenEmbeddings, layers, finalNorm, ropeReal, ropeImag, wcls);
    }
}
