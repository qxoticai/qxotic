// LFM2.5 (Liquid Foundation Model 2.5) against the MemoryView boundary. Each layer is EITHER GQA
// attention (kv-heads > 0) OR a gated short-convolution mixer (kv-heads == 0); the FFN is EITHER
// dense SwiGLU OR top-k MoE. Weights/state/KV are MemoryView<MemorySegment>; GEMM/GEMV use the
// shared MatMul entry points. An optional LFM2-VL sidecar projects images directly into the same
// residual stream.
package com.qxotic.jinfer.models.lfm2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.EmbeddingModel;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Moe;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jinfer.kernels.Trace;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

public final class Lfm2
        implements LanguageModel<Lfm2.Configuration, Lfm2.Weights, Lfm2.State>,
                EmbeddingModel<Lfm2.Configuration, Lfm2.Weights, Lfm2.State>,
                Multimodal {

    /** llama.cpp's pooling_type enum value for CLS - pool the sequence's FIRST row (its BOS). */
    static final int POOLING_CLS = 2;

    /** The short-conv in_proj's 3-way row split: B | C_gate | x blocks, each dim wide. */
    private static final int SHORTCONV_PARTS = 3;

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;
    private final Lfm2Vision vision;

    Lfm2(Configuration configuration, Tokenizer tokenizer, Weights weights) {
        this(configuration, tokenizer, weights, null);
    }

    private Lfm2(
            Configuration configuration, Tokenizer tokenizer, Weights weights, Lfm2Vision vision) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
        this.vision = vision;
    }

    @Override
    public Configuration configuration() {
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
    @SuppressWarnings("unchecked")
    public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
        if (modality == Media.Image.class && vision != null)
            return Optional.of((MediaProjector<R>) vision);
        return Optional.empty();
    }

    Lfm2Vision vision() {
        return vision;
    }

    @Override
    public Optional<CheckpointCodec<State>> checkpointCodec() {
        return configuration.causalAttention
                ? Optional.of(new Lfm2CheckpointCodec(configuration))
                : Optional.empty();
    }

    @Override
    public State newState(
            int contextCapacity, int batchCapacity, MemoryArena<MemorySegment> arena) {
        return new State(configuration, contextCapacity, batchCapacity, arena, false);
    }

    @Override
    public State newState(int contextCapacity, int batchCapacity) {
        MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
        try {
            return new State(configuration, contextCapacity, batchCapacity, arena, true);
        } catch (RuntimeException | Error failure) {
            Arenas.close(arena);
            throw failure;
        }
    }

    @Override
    public void ingest(State state, Batch batch) {
        state.exclusively(() -> forward(state, batch));
        Reference.reachabilityFence(this);
    }

    private void forward(State s, Batch batch) {
        int n = batch.count();
        if (n <= 0) throw new IllegalArgumentException("batch must not be empty");
        if (n > s.batchCapacity())
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + s.batchCapacity());
        int from = s.position();
        if (from + n > s.contextCapacity()) {
            throw new IllegalArgumentException(
                    "ingest of "
                            + n
                            + " at position "
                            + from
                            + " exceeds contextCapacity "
                            + s.contextCapacity());
        }
        switch (batch.input()) {
            case Batch.Input.Tokens t -> {
                if (!configuration.causalAttention)
                    throw new UnsupportedOperationException(
                            "retrieval checkpoints require packed sequences for bidirectional"
                                    + " attention");
                int[] ids = t.ids();
                requireTokens(ids);
                forward(s, ids, from, n);
            }
            case Batch.Input.Sequences seq -> {
                if (configuration.causalAttention)
                    throw new UnsupportedOperationException(
                            "this LFM2.5 checkpoint is generative: batched embedding needs the"
                                    + " embedding checkpoint (LFM2.5-Embedding, attention.causal ="
                                    + " false)");
                requireComplete(seq);
                requireTokens(seq.tokens().ids());
                forwardSegmented(s, seq.tokens().ids(), seq.seqLen(), n);
            }
            case Batch.Input.Embeddings e -> {
                if (vision == null)
                    throw new UnsupportedOperationException("no media encoder loaded");
                if (e.rows().shape().flatAt(1) != configuration.embeddingLength)
                    throw new IllegalArgumentException(
                            "embedding width "
                                    + e.rows().shape().flatAt(1)
                                    + " != model width "
                                    + configuration.embeddingLength);
                forwardEmbeddings(
                        s,
                        Views.castToSegmentBacked(e.rows(), "embedding rows"),
                        from,
                        n,
                        e.bidirectional());
            }
        }
        s.advance(batch);
    }

    @Override
    public MemoryView<?> logits(State s, int output) {
        MemoryView<?> result = s.exclusively(() -> projectLogits(s, output));
        Reference.reachabilityFence(this);
        return result;
    }

    private MemoryView<?> projectLogits(State s, int output) {
        requireOutput(s, output);
        int dim = configuration.embeddingLength;
        int row = s.lastBatchSize() - s.outputCount() + output;
        Norms.rmsnorm(
                s.normed,
                0,
                s.residual,
                (long) row * dim,
                weights.finalNorm,
                dim,
                configuration.rmsNormEps);
        MatMul.gemv(weights.wcls, s.normed, s.logits);
        Activations.softcap(
                s.logits, 0, configuration.vocabularySize, configuration.logitSoftcapping);
        return s.logits;
    }

    // === Forward ===

    private void forward(State state, int[] tokens, int startPos, int seqLen) {
        // ONCE for the batch: an angle never depends on the layer
        RoPE.fill(
                state.ropeCos,
                state.ropeSin,
                startPos,
                seqLen,
                configuration.headSize / 2,
                weights.rope());
        embedTokens(state, tokens, seqLen);
        for (int l = 0; l < configuration.numberOfLayers; l++)
            layer(state, l, startPos, seqLen, false);
        commitKv(state, startPos, seqLen);
    }

    private void forwardEmbeddings(
            State state,
            MemoryView<MemorySegment> rows,
            int startPos,
            int seqLen,
            boolean bidirectional) {
        RoPE.fill(
                state.ropeCos,
                state.ropeSin,
                startPos,
                seqLen,
                configuration.headSize / 2,
                weights.rope());
        Convert.copyF32(rows, 0, state.residual, 0, (long) seqLen * configuration.embeddingLength);
        for (int l = 0; l < configuration.numberOfLayers; l++)
            layer(state, l, startPos, seqLen, bidirectional);
        commitKv(state, startPos, seqLen);
    }

    /** Token-embedding lookup into the residual stream (no scaling, unlike Gemma4). */
    private void embedTokens(State state, int[] tokens, int seqLen) {
        Views.checkAlive(weights.tokenEmbeddings, "tokenEmbeddings"); // fail-fast on freed weights
        Convert.gatherToF32(
                weights.tokenEmbeddings,
                tokens,
                0,
                seqLen,
                state.residual,
                0,
                configuration.embeddingLength);
    }

    private void requireTokens(int[] tokens) {
        for (int token : tokens) {
            if (token < 0 || token >= configuration.vocabularySize)
                throw new IllegalArgumentException(
                        "token id " + token + " outside [0," + configuration.vocabularySize + ")");
        }
    }

    /** One block: short-conv mixer OR attention, then the FFN, in place on the residual. */
    private void layer(State state, int l, int startPos, int seqLen, boolean bidirectional) {
        if (configuration.isRecurrentLayer(l)) shortConvMixer(state, l, seqLen);
        else attention(state, l, startPos, seqLen, bidirectional);
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
        MatMul.gemm(sc.inProj(), state.normed, state.shortConvTmp, seqLen);
        Convolutions.shortConvScan(
                weights.layers[l].shortConv().kernel(),
                state.shortConvState[l],
                state.shortConvTmp,
                state.branchOut,
                seqLen,
                configuration.embeddingLength,
                configuration.shortConvLCache,
                SHORTCONV_PARTS);
        MatMul.gemm(sc.outProj(), state.branchOut, state.shortConvOut, seqLen);
        Ops.addInPlace(state.residual, 0, state.shortConvOut, 0, seqLen * dim);
    }

    // --- attention (GQA) ---

    /**
     * Pre-norm GQA: per-head Q/K RMS-norm + NeoX RoPE (no V-norm), full causal attention with
     * {@code scale = 1/sqrt(headSize)}, output projection, optional post-norm, added to the
     * residual.
     */
    private void attention(State state, int l, int startPos, int seqLen, boolean bidirectional) {
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
                    null,
                    bidirectional);
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
        MatMul.gemm(attn.wq(), state.normed, state.query, seqLen);
        headNormRope(state, state.query, queryDim, config.numberOfHeads, attn.qNorm(), seqLen);
        MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];
        MatMul.gemm(attn.wk(), state.normed, bK, seqLen);
        if (attn.wv() != null) MatMul.gemm(attn.wv(), state.normed, bV, seqLen);
        else Convert.copyF32(bK, 0, bV, 0, (long) seqLen * kvDim);
        headNormRope(state, bK, kvDim, nKvHeads, attn.kNorm(), seqLen);
    }

    /** The shared tail: output projection, optional post-norm, added to the residual. */
    private void attentionFinish(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength;
        AttentionWeights attn = weights.layers[l].attention();
        MatMul.gemm(attn.wo(), state.attnOut, state.branchOut, seqLen);
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
        Parallel.forLoop(
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
                postFfnNormW = weights.layers[l].postFfnNorm();
        Norms.rmsnormRows(
                state.normed, state.residual, ffnNormW, seqLen, dim, configuration.rmsNormEps);
        MatMul.gemm(ffn.gate(), state.normed, state.hidden, seqLen);
        MatMul.gemm(ffn.up(), state.normed, state.hidden2, seqLen);
        // hidden/hidden2 are [batch, maxHiddenDim]: the gemms write rows at the buffer's stride,
        // which is wider than this layer's hiddenDim when widths differ across layers
        int hiddenStride = Math.toIntExact(state.hidden.stride().flatAt(0));
        Parallel.forLoop(
                seqLen,
                s ->
                        Activations.siluMultiply(
                                state.hidden,
                                s * hiddenStride,
                                state.hidden2,
                                s * hiddenStride,
                                hiddenDim));
        MatMul.gemm(ffn.down(), state.hidden, state.normed, seqLen);
        if (postFfnNormW != null)
            Norms.rmsnormRows(
                    state.normed,
                    state.normed,
                    postFfnNormW,
                    seqLen,
                    dim,
                    configuration.rmsNormEps);
        Ops.addInPlace(state.residual, 0, state.normed, 0, seqLen * dim);
    }

    /**
     * Top-k MoE FFN (LFM-style): no shared MLP, no expert pre/post norms. Router → softmax|sigmoid
     * → top-k (optional {@code exp_probs_b} bias steers selection only, DeepSeek-style; the expert
     * weights stay unbiased) → normalize the k weights → per-expert (separate) gate/up/SiLU/down,
     * prob-weighted into the residual via the shared CSR {@link Moe#dispatch}.
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
        MatMul.gemm(moe.router(), state.normed, state.moeRouterB, seqLen);

        for (int s = 0; s < seqLen; s++) {
            long ro = (long) s * nExperts;
            if (config.expertGatingFunc == 2)
                Ops.mapInPlace(
                        state.moeRouterB, ro, nExperts, v -> (float) (1.0 / (1.0 + Math.exp(-v))));
            else Ops.softmaxInPlace(state.moeRouterB, ro, nExperts);
        }
        // exp_probs_b is a selection-time bias only (llama.cpp build_moe_ffn): it is added to a
        // scratch copy of the gating probabilities to pick the top-k, while the routed weights are
        // read from the UNBIASED probabilities (the two sources of Moe.selectTopK).
        MemoryView<MemorySegment> selection = state.moeRouterB;
        if (moe.expProbsBias() != null) {
            selection = state.moeSelectionB;
            for (int s = 0; s < seqLen; s++) {
                long ro = (long) s * nExperts;
                Ops.copyStrided(selection, ro, 1, state.moeRouterB, ro, nExperts);
                Ops.addInPlace(selection, ro, moe.expProbsBias(), 0, nExperts);
            }
        }
        Moe.selectTopK(
                selection,
                state.moeRouterB,
                seqLen,
                nExperts,
                topK,
                state.moeRowTopE,
                state.moeRowTopP,
                state.moeExpertCounts);
        Moe.normalizeTopP(state.moeRowTopP, seqLen, topK);

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
                    MatMul.gemm(moe.gateExps()[e], gather, state.moeHidden, n);
                    MatMul.gemm(moe.upExps()[e], gather, state.moeHidden2, n);
                    Parallel.forLoop(
                            n,
                            j ->
                                    Activations.siluMultiply(
                                            state.moeHidden,
                                            j * expertFF,
                                            state.moeHidden2,
                                            j * expertFF,
                                            expertFF));
                    MatMul.gemm(moe.downExps()[e], state.moeHidden, out, n);
                });

        Parallel.forLoop(
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
            int elements = Math.multiplyExact(seqLen, kvDim);
            long cacheOffset = (long) startPos * kvDim;
            Convert.f32ToF16(state.batchK[l], 0, state.keyCache[l], cacheOffset, elements);
            Convert.f32ToF16(state.batchV[l], 0, state.valueCache[l], cacheOffset, elements);
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
        RoPE.fill(
                state.ropeCos, state.ropeSin, posOf, n, configuration.headSize / 2, weights.rope());
        embedTokens(state, tokens, n);
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
        MatMul.gemm(sc.inProj(), state.normed, state.shortConvTmp, seqLen);

        Convolutions.segmentedShortConv(
                sc.kernel(),
                state.shortConvTmp,
                state.branchOut,
                segRow0,
                segLen,
                seqLen,
                configuration.embeddingLength,
                configuration.shortConvLCache,
                SHORTCONV_PARTS);
        MatMul.gemm(sc.outProj(), state.branchOut, state.shortConvOut, seqLen);
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
     * The sentence embedding: final-norm the pooled row, then L2-normalize. CLS pooling reads the
     * sequence's first retained row (its BOS); {@code outputIndex} addresses retained rows exactly
     * as {@code logits} does.
     */
    @Override
    public void projectEmbedding(State s, int outputIndex, Consumer<MemoryView<?>> consumer) {
        Objects.requireNonNull(consumer, "consumer");
        try {
            s.exclusively(() -> consumer.accept(projectEmbedding0(s, outputIndex)));
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    private MemoryView<?> projectEmbedding0(State s, int outputIndex) {
        requireOutput(s, outputIndex);
        int dim = configuration.embeddingLength;
        int row = s.lastBatchSize() - s.outputCount() + outputIndex;
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

    private static void requireOutput(State state, int output) {
        if (output < 0 || output >= state.outputCount())
            throw new IllegalArgumentException(
                    "output " + output + " outside [0," + state.outputCount() + ")");
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
     * applies before MaxSim. The returned view is a reused per-state buffer, so the caller copies
     * it before projecting another row.
     */
    MemoryView<MemorySegment> colbertRow(State s, int row) {
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
        MatMul.gemv(weights.dense2(), es.embOut, es.colbertOut);
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
    public void embedAll(State state, Batch.Input.Sequences seqs, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(seqs, "sequences");
        Objects.requireNonNull(sink, "sink");
        requireComplete(seqs);
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
                state, sequences, (index, rowStart) -> projectEmbedding(state, rowStart, sink));
    }

    private static void requireComplete(Batch.Input.Sequences sequences) {
        long total = 0;
        int[] lengths = sequences.seqLen();
        for (int i = 0; i < lengths.length; i++) {
            if (lengths[i] <= 0)
                throw new IllegalArgumentException(
                        "sequence " + i + " has invalid length " + lengths[i]);
            total += lengths[i];
        }
        int tokens = sequences.tokens().ids().length;
        if (total != tokens)
            throw new IllegalArgumentException(
                    "packed token count " + tokens + " != sequence lengths " + total);
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
     * Holds the state once across all groups; per-ingest access nests inside.
     */
    int forEachSequence(State state, int[][] seqs, SequenceVisitor visitor) {
        return state.exclusively(() -> forEachSequence0(state, seqs, visitor));
    }

    private int forEachSequence0(State state, int[][] seqs, SequenceVisitor visitor) {
        int cap = Math.min(state.batchCapacity(), state.contextCapacity());
        int total = 0, seq = 0;
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
            implements ContextConfiguration {

        public Configuration {
            feedForwardLength = feedForwardLength.clone();
            numberOfKeyValueHeadsPerLayer = numberOfKeyValueHeadsPerLayer.clone();
        }

        @Override
        public int[] feedForwardLength() {
            return feedForwardLength.clone();
        }

        @Override
        public int[] numberOfKeyValueHeadsPerLayer() {
            return numberOfKeyValueHeadsPerLayer.clone();
        }

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
            MemoryView<MemorySegment>[] gateExps,
            MemoryView<MemorySegment>[] upExps,
            MemoryView<MemorySegment>[] downExps,
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

    public static final class State extends ContextState {

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
        final MemoryView<MemorySegment> moeRouterB, moeSelectionB, moeGather, moeDownB, moeOutB;
        // Per-expert gate/up at EXACTLY expertFeedForwardLength wide: hidden/hidden2 are sized to
        // the model's max FFN width (dense layers can be wider), and the silu-multiply between
        // gate/up and down addresses rows packed at the expert width.
        final MemoryView<MemorySegment> moeHidden, moeHidden2;
        final int[] moeExpertCounts, moeRowTopE;
        final float[] moeRowTopP;
        final Moe.Routing moeRouting;

        EmbedScratch embedScratch; // lazy: only the embedding checkpoints ever pay for it

        EmbedScratch embedScratch(Configuration config) {
            if (embedScratch == null)
                embedScratch = new EmbedScratch(config, batchCapacity(), memoryArena());
            return embedScratch;
        }

        /**
         * Recycles this allocation for a fresh sequence: cursor to 0 and the RECURRENT buffers
         * zeroed - stale KV rows beyond the cursor are attention-masked and harmless, but the
         * rolling short-conv state carries values across positions and would leak the previous
         * conversation into the next one.
         */
        @Override
        protected void clearHistory() {
            for (MemoryView<MemorySegment> conv : shortConvState) {
                if (conv != null) {
                    Ops.fillInPlace(conv, 0, Math.toIntExact(conv.logicalSize()), 0f);
                }
            }
        }

        @SuppressWarnings("unchecked")
        State(
                Configuration config,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity > config.contextLength())
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds model contextLength "
                                + config.contextLength());
            int c = batchCapacity;
            int dim = config.embeddingLength;
            int maxQueryDim = config.queryDim();
            int maxHiddenDim = config.maxHiddenDim();
            this.residual = Views.allocateF32(memoryArena(), c, dim);
            this.normed = Views.allocateF32(memoryArena(), c, dim);
            this.branchOut = Views.allocateF32(memoryArena(), c, dim);
            this.attnOut = Views.allocateF32(memoryArena(), c, maxQueryDim);
            this.query = Views.allocateF32(memoryArena(), c, maxQueryDim);
            this.hidden = Views.allocateF32(memoryArena(), c, maxHiddenDim);
            this.hidden2 = Views.allocateF32(memoryArena(), c, maxHiddenDim);
            this.logits = Views.allocateF32(memoryArena(), 1, config.vocabularySize);
            // rotary values for the batch about to be ingested: sized by BATCH, never context
            this.ropeCos = Views.allocateF32(memoryArena(), c, config.headSize / 2);
            this.ropeSin = Views.allocateF32(memoryArena(), c, config.headSize / 2);
            this.shortConvTmp = Views.allocateF32(memoryArena(), c, SHORTCONV_PARTS * dim);
            this.shortConvOut = Views.allocateF32(memoryArena(), c, dim);
            int n = config.numberOfLayers;
            this.keyCache = new MemoryView[n];
            this.valueCache = new MemoryView[n];
            this.batchK = new MemoryView[n];
            this.batchV = new MemoryView[n];
            this.shortConvState = new MemoryView[n];
            int hist = Math.max(config.shortConvLCache - 1, 0);
            for (int l = 0; l < n; l++) {
                if (config.isRecurrentLayer(l)) {
                    if (config.causalAttention)
                        shortConvState[l] = Views.allocateF32(memoryArena(), hist * dim);
                } else {
                    int kvDim = config.kvDim(l);
                    if (config.causalAttention) {
                        keyCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
                        valueCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
                    }
                    batchK[l] = Views.allocateF32(memoryArena(), c, kvDim);
                    batchV[l] = Views.allocateF32(memoryArena(), c, kvDim);
                }
            }
            if (config.isMoE()) {
                int e = config.expertCount, tk = config.expertUsedCount;
                this.moeRouterB = Views.allocateF32(memoryArena(), c, e);
                this.moeSelectionB = Views.allocateF32(memoryArena(), c, e);
                this.moeGather = Views.allocateF32(memoryArena(), c, dim);
                this.moeDownB = Views.allocateF32(memoryArena(), c, dim);
                this.moeOutB = Views.allocateF32(memoryArena(), c, dim);
                this.moeHidden =
                        Views.allocateF32(memoryArena(), c, config.expertFeedForwardLength);
                this.moeHidden2 =
                        Views.allocateF32(memoryArena(), c, config.expertFeedForwardLength);
                this.moeExpertCounts = new int[e];
                this.moeRowTopE = new int[c * tk];
                this.moeRowTopP = new float[c * tk];
                this.moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts);
            } else {
                this.moeRouterB =
                        this.moeSelectionB = this.moeGather = this.moeDownB = this.moeOutB = null;
                this.moeHidden = this.moeHidden2 = null;
                this.moeExpertCounts = this.moeRowTopE = null;
                this.moeRowTopP = null;
                this.moeRouting = null;
            }
        }

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }
    }

    /**
     * Per-state scratch for the bidirectional embedding path, from the state's own arena (freed
     * with the state): per-sequence Q/K/V/out gathers plus the pooled-output row.
     */
    static final class EmbedScratch {
        final MemoryView<MemorySegment> segQ, segK, segV, segOut, embOut, colbertOut, colbertRows;

        /** Position per row, and first row per segment (refilled per forwardSegmented). */
        final int[] posOf, segRow0;

        EmbedScratch(
                Configuration config, int batchCapacity, MemoryAllocator<MemorySegment> memory) {
            int queryDim = config.queryDim(), kvDim = config.maxKvDim();
            this.segQ = Views.allocateF32(memory, batchCapacity, queryDim);
            this.segOut = Views.allocateF32(memory, batchCapacity, queryDim);
            this.segK = Views.allocateF32(memory, batchCapacity, kvDim);
            this.segV = Views.allocateF32(memory, batchCapacity, kvDim);
            this.embOut = Views.allocateF32(memory, 1, config.embeddingLength());
            if (config.embeddingLengthOut() > 0) {
                this.colbertOut = Views.allocateF32(memory, 1, config.embeddingLengthOut());
                this.colbertRows =
                        Views.allocateF32(memory, batchCapacity, config.embeddingLengthOut());
            } else {
                this.colbertOut = this.colbertRows = null;
            }
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

    /** Loads the text backbone and its LFM2 vision sidecar into {@code arena}. */
    public static Lfm2 loadModel(Path textPath, Path mmprojPath, Arena arena) throws IOException {
        return loadModel(textPath, arena).withMedia(mmprojPath, arena);
    }

    /** Returns a model sharing this backbone's weights with a validated vision sidecar attached. */
    public Lfm2 withMedia(Path mmprojPath, Arena arena) throws IOException {
        Objects.requireNonNull(mmprojPath, "mmprojPath");
        Objects.requireNonNull(arena, "arena");
        try (FileChannel channel = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, mmprojPath.toString());
            int projectionDim = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", 0);
            if (projectionDim != configuration.embeddingLength)
                throw new IllegalArgumentException(
                        "'"
                                + mmprojPath.getFileName()
                                + "' projector width "
                                + projectionDim
                                + " does not match model width "
                                + configuration.embeddingLength);
            Lfm2Vision encoder =
                    Lfm2Vision.loadModel(
                            mmprojPath, gguf, ModelLoader.loadTensors(channel, gguf, arena));
            return new Lfm2(configuration, tokenizer, weights, encoder);
        }
    }

    public static Lfm2 loadModel(FileChannel fileChannel, GGUF gguf, Arena arena)
            throws IOException {
        return loadModel(fileChannel, gguf, arena, null);
    }

    /**
     * As above with a caller-supplied tokenizer; null = the GGUF's own (toknroll's builtin
     * registrations, with the {@code -Dtoknroll.gguf.pre.*} escape hatch applied at build).
     */
    public static Lfm2 loadModel(
            FileChannel fileChannel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        if (tokenizer == null) {
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        }
        Configuration config = readConfiguration(gguf, tokenizer.vocabulary().size());

        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(fileChannel, gguf, arena);
        return new Lfm2(config, tokenizer, loadWeights(tensors, config));
    }

    static Configuration readConfiguration(GGUF gguf, int vocabularySize) {
        String arch = gguf.getString("general.architecture");
        require(
                arch.equals("lfm2") || arch.equals("lfm2moe"),
                "unsupported architecture '" + arch + "'");

        int contextLength = gguf.getValue(int.class, arch + ".context_length");
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        require(
                contextLength > 0
                        && embeddingLength > 0
                        && numberOfHeads > 0
                        && numberOfLayers > 0
                        && vocabularySize > 0
                        && embeddingLength % numberOfHeads == 0,
                "invalid core dimensions");
        int headSize = embeddingLength / numberOfHeads;
        require((headSize & 1) == 0, "attention head size must be even");
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
        } else if (ffnRaw instanceof Number value) {
            feedForwardLength = new int[numberOfLayers];
            Arrays.fill(feedForwardLength, value.intValue());
        } else throw new IllegalArgumentException("LFM2: invalid feed_forward_length metadata");
        require(feedForwardLength.length == numberOfLayers, "invalid feed-forward layout");
        for (int width : feedForwardLength)
            require(width > 0, "feed-forward widths must be positive");

        // Per-layer kv-head count: 0 marks a recurrent (short-conv) layer (no attn_k tensor);
        // attention layers derive it from the K-projection's row count (GGUF shape[1]).
        int[] kvHeads = new int[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            var kWeight = gguf.getTensor("blk." + i + ".attn_k.weight");
            if (kWeight == null) continue;
            long[] shape = kWeight.shape();
            require(
                    shape.length >= 2 && shape[1] > 0 && shape[1] % headSize == 0,
                    "invalid K projection at layer " + i);
            kvHeads[i] = Math.toIntExact(shape[1] / headSize);
            require(
                    numberOfHeads % kvHeads[i] == 0,
                    "KV heads do not divide query heads at layer " + i);
        }

        Configuration config =
                new Configuration(
                        embeddingLength,
                        feedForwardLength,
                        numberOfLayers,
                        numberOfHeads,
                        kvHeads,
                        vocabularySize,
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
        require(
                rmsNormEps > 0f
                        && Float.isFinite(rmsNormEps)
                        && ropeTheta > 0f
                        && Float.isFinite(ropeTheta)
                        && logitSoftcapping >= 0f
                        && Float.isFinite(logitSoftcapping),
                "invalid normalization, RoPE, or softcapping metadata");
        require(shortConvLCache > 0, "short-convolution cache must be positive");
        require(embeddingLengthOut >= 0 && poolingType >= 0, "invalid retrieval metadata");
        if (expertCount == 0) {
            require(
                    arch.equals("lfm2") && expertUsedCount == 0 && expertFeedForwardLength == 0,
                    "inconsistent dense/MoE metadata");
        } else {
            require(
                    arch.equals("lfm2moe")
                            && expertUsedCount > 0
                            && expertUsedCount <= expertCount
                            && expertFeedForwardLength > 0
                            && leadingDenseBlockCount >= 0
                            && leadingDenseBlockCount <= numberOfLayers
                            && (expertGatingFunc == 1 || expertGatingFunc == 2),
                    "invalid MoE metadata");
        }
        require(
                gguf.getValueOrDefault(int.class, arch + ".vocab_size", vocabularySize)
                        == vocabularySize,
                "tokenizer vocabulary does not match the model");
        return config;
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors, Configuration config) {
        int n = config.numberOfLayers;
        RoPE.Schedule rope = RoPE.plain(config.headSize, config.ropeTheta);
        int dim = config.embeddingLength;

        MemoryView<MemorySegment> tokenEmbeddings =
                weight(tensors, "token_embd.weight", config.vocabularySize, dim);
        MemoryView<MemorySegment> wcls =
                ModelLoader.find(tensors, "output.weight").orElse(tokenEmbeddings);
        requireWeight(wcls, "output.weight", config.vocabularySize, dim);
        // LFM2.5 names the final norm token_embd_norm (no separate output_norm); embeddings are
        // tied.
        MemoryView<MemorySegment> finalNorm =
                f32(
                        tensors,
                        tensors.containsKey("output_norm.weight")
                                ? "output_norm.weight"
                                : "token_embd_norm.weight",
                        dim);

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            MemoryView<MemorySegment> attnNorm = f32(tensors, p + "attn_norm.weight", dim);
            MemoryView<MemorySegment> postAttnNorm =
                    optionalF32(tensors, p + "post_attention_norm.weight", dim);
            MemoryView<MemorySegment> ffnNorm = f32(tensors, p + "ffn_norm.weight", dim);
            MemoryView<MemorySegment> postFfnNorm =
                    optionalF32(tensors, p + "post_ffw_norm.weight", dim);

            AttentionWeights attention = null;
            ShortConvWeights shortConv = null;
            if (config.isRecurrentLayer(i)) {
                shortConv =
                        new ShortConvWeights(
                                f32(
                                        tensors,
                                        p + "shortconv.conv.weight",
                                        dim,
                                        config.shortConvLCache),
                                weight(
                                        tensors,
                                        p + "shortconv.in_proj.weight",
                                        SHORTCONV_PARTS * dim,
                                        dim),
                                weight(tensors, p + "shortconv.out_proj.weight", dim, dim));
            } else {
                int queryDim = config.queryDim(), kvDim = config.kvDim(i);
                MemoryView<MemorySegment> value =
                        ModelLoader.find(tensors, p + "attn_v.weight").orElse(null);
                if (value != null) requireWeight(value, p + "attn_v.weight", kvDim, dim);
                attention =
                        new AttentionWeights(
                                weight(tensors, p + "attn_q.weight", queryDim, dim),
                                weight(tensors, p + "attn_k.weight", kvDim, dim),
                                value,
                                weight(tensors, p + "attn_output.weight", dim, queryDim),
                                f32(tensors, p + "attn_q_norm.weight", config.headSize),
                                f32(tensors, p + "attn_k_norm.weight", config.headSize));
            }

            DenseFfnWeights dense = null;
            MoeFfnWeights moe = null;
            if (config.isMoELayer(i)) {
                int experts = config.expertCount, expertFf = config.expertFeedForwardLength;
                moe =
                        new MoeFfnWeights(
                                weight(tensors, p + "ffn_gate_inp.weight", experts, dim),
                                experts(
                                        tensors,
                                        p + "ffn_gate_exps.weight",
                                        experts,
                                        expertFf,
                                        dim),
                                experts(tensors, p + "ffn_up_exps.weight", experts, expertFf, dim),
                                experts(
                                        tensors,
                                        p + "ffn_down_exps.weight",
                                        experts,
                                        dim,
                                        expertFf),
                                optionalF32(tensors, p + "exp_probs_b.bias", experts));
            } else {
                int hidden = config.feedForwardLength[i];
                dense =
                        new DenseFfnWeights(
                                weight(tensors, p + "ffn_gate.weight", hidden, dim),
                                weight(tensors, p + "ffn_up.weight", hidden, dim),
                                weight(tensors, p + "ffn_down.weight", dim, hidden));
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
        MemoryView<MemorySegment> dense2 = null;
        if (config.embeddingLengthOut > 0)
            dense2 = weight(tensors, "dense_2.weight", config.embeddingLengthOut, dim);
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls, dense2);
    }

    private static MemoryView<MemorySegment> weight(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int rows, int columns) {
        MemoryView<MemorySegment> value = ModelLoader.require(tensors, name);
        requireWeight(value, name, rows, columns);
        return value;
    }

    private static void requireWeight(
            MemoryView<MemorySegment> value, String name, int rows, int columns) {
        Shape actual = value.dataType().logicalShape(value.shape());
        Shape expected = Shape.flat(rows, columns);
        require(actual.equals(expected), name + " expected " + expected + " but was " + actual);
    }

    private static MemoryView<MemorySegment> f32(
            Map<String, MemoryView<MemorySegment>> tensors, String name, long... shape) {
        MemoryView<MemorySegment> value = ModelLoader.requireF32(tensors, name);
        Shape expected = Shape.flat(shape);
        require(
                value.shape().equals(expected),
                name + " expected " + expected + " but was " + value.shape());
        return value;
    }

    private static MemoryView<MemorySegment> optionalF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name, long... shape) {
        if (!tensors.containsKey(name)) return null;
        return f32(tensors, name, shape);
    }

    private static MemoryView<MemorySegment>[] experts(
            Map<String, MemoryView<MemorySegment>> tensors,
            String name,
            int count,
            int rows,
            int columns) {
        MemoryView<MemorySegment>[] values =
                Views.sliceLeadingAxis(ModelLoader.require(tensors, name));
        require(values.length == count, name + " has the wrong expert count");
        for (MemoryView<MemorySegment> value : values) requireWeight(value, name, rows, columns);
        return values;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("LFM2: " + message);
    }
}
