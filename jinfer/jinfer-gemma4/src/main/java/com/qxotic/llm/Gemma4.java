// Gemma 4 against the com.qxotic.llm model API: a faithful port of the production jinfer Gemma4
// forward (per-layer sliding-window/full attention with distinct RoPE thetas, shared KV across the
// tail layers, per-head Q/K RMS norms + a bare V norm, embedding scaling, GELU MLP, pre/post norms,
// per-layer output scaling, final-logit softcap, optional per-layer embeddings) — reusing the
// jinfer kernels (Norms, RoPE, FlashAttention, Moe, ModelLoader) now exposed for it. Dense and MoE
// (A4B) FFN paths are both ported. gemma-4 is a multimodal, MTP-capable architecture, so this
// implements
// MultiModal and MultiToken (instanceof = the arch supports it). The current text-only GGUFs load
// neither
// the media adapters nor MTP heads, so both report "supported but not loaded" via empty gates —
// modalities()/embedder() empty, depth() empty — never a sentinel.
package com.qxotic.llm;

import static com.qxotic.jinfer.Norms.rmsnorm;
import static com.qxotic.jinfer.Norms.rmsnormNoWeight;
import static com.qxotic.jinfer.Norms.scaleByWeight;
import static com.qxotic.jinfer.Norms.sumOfSquares;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.*;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;

public final class Gemma4
        implements LanguageModel<Gemma4.Configuration, Gemma4.Weights, Gemma4.State>,
                MultiModal,
                MultiToken<Gemma4.State> {

    private final Configuration configuration;
    private final GgufTokenizer tokenizer;
    private final Weights weights;
    private Embedder<Media.Image>
            vision; // image encoder; null on text-only loads (set by loadModel(text, mmproj, ctx))
    private Embedder<Media.Audio>
            audio; // audio encoder; null unless the mmproj carries a gemma4ua adapter
    private Gemma4Mtp mtp; // MTP draft sidecar; null unless loadModel(..., mtpSidecar) loaded one
    private Gemma4MtpDecoder mtpDecoder; // paired draft forward (single-threaded scratch)

    Gemma4(Configuration configuration, GgufTokenizer tokenizer, Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
    }

    // === com.qxotic.llm.Model seam ===

    @Override
    public Configuration config() {
        return configuration;
    }

    @Override
    public Weights weights() {
        return weights;
    }

    public GgufTokenizer tokenizer() {
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
        if (n > s.batchCapacity) {
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + s.batchCapacity);
        }
        int from = s.position(); // append at the cursor; position-agnostic batch
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
                s.lastTokens = ids; // row->token map for the MTP draft seam (logits(s,out,head))
                // decode step (one token): bandwidth-bound, so run at physical-core width on the
                // decode
                // pool (no SMT contention); prefill (n>1) stays compute-bound on the common pool.
                if (n == 1)
                    Parallel.onDecodePool(
                            () -> {
                                forward(s, ids, 0, from, n);
                                return null;
                            });
                else forward(s, ids, 0, from, n);
            }
            case com.qxotic.jinfer.Batch.Input.Embeddings e -> {
                if (vision == null && audio == null)
                    throw new UnsupportedOperationException(
                            "no media encoder loaded — use loadModel(text, mmproj, ctx)");
                int pad = tokenizer.getSpecialTokens().getOrDefault("<pad>", 0);
                forwardEmbeddings(s, e.rows(), pad, from, n, e.bidirectional());
            }
            case com.qxotic.jinfer.Batch.Input.Sequences seq ->
                    throw new UnsupportedOperationException(
                            "Gemma4 is generative: packed sequences (batched embedding) not"
                                    + " supported");
        }
        s.advance(n, batch.outputs());
    }

    /**
     * Logits for the {@code output}-th retained row: the layer loop leaves every row's final
     * residual in {@code state.residual}; finalize the output head for that row. (LAST → output 0
     * is the last row's residual; ALL → output indexes the rows in order.)
     */
    @Override
    public FloatTensor logits(State s, int output) {
        int dim = configuration.embeddingLength();
        int row = s.lastChunkLen - s.outputCount + output; // map retained index → chunk row
        // The vocab projection (vocab × dim) is the heaviest single op in decode and
        // memory-bandwidth
        // bound — run it (and the final norm) at physical-core width on the decode pool, not the
        // SMT-wide common pool, matching the production engine's computeLogits.
        return Parallel.onDecodePool(
                () -> {
                    rmsnorm(
                            s.xb,
                            0,
                            s.residual,
                            row * dim,
                            weights.rmsFinalWeights,
                            dim,
                            configuration.rmsNormEps());
                    weights.classifierWeights.matmul(
                            s.xb, s.logits, configuration.vocabularySize(), dim);
                    Activations.softcap(
                            s.logits,
                            0,
                            configuration.vocabularySize(),
                            configuration.logitSoftcapping());
                    return s.logits;
                });
    }

    /**
     * Logits for ALL retained rows in one head pass: rmsnorm each row into {@code s.xb}, then ONE
     * vocab GEMM - the classifier weight (the heaviest stream in decode) is read once instead of
     * once per row. {@code dst} holds {@code outputCount x vocab}, row-major; softcapped in place.
     * The speculative verify walk is the consumer (its per-row {@link #logits(State, int)} calls
     * re-streamed ~the whole head weight per draft row).
     */
    public void logitsAll(State s, FloatTensor dst) {
        int dim = configuration.embeddingLength();
        int vocab = configuration.vocabularySize();
        int n = s.outputCount;
        int first = s.lastChunkLen - n;
        Parallel.onDecodePool(
                () -> {
                    for (int r = 0; r < n; r++) {
                        rmsnorm(
                                s.xb,
                                r * dim,
                                s.residual,
                                (long) (first + r) * dim,
                                weights.rmsFinalWeights,
                                dim,
                                configuration.rmsNormEps());
                    }
                    weights.classifierWeights.gemm(s.xb, dim, dst, vocab, n, vocab, dim);
                    Activations.softcap(dst, 0, n * vocab, configuration.logitSoftcapping());
                    return null;
                });
    }

    // === Capabilities: gemma-4's architecture supports MTP and media input, so this implements
    // both.
    // Whether either is usable is a runtime fact gated by un-ignorable empties — the current
    // text-only
    // GGUFs load no MTP heads and no media adapters, so both report supported-but-not-loaded. ===

    @Override
    public OptionalInt depth() {
        return mtp == null
                ? OptionalInt.empty()
                : OptionalInt.of(2); // draft depth the loop defaults to
    }

    /**
     * NOTE - seam gap, deliberate: {@link com.qxotic.jinfer.MultiToken}'s per-head logits view is
     * too weak for an efficient speculative loop (it cannot expose the chained draft hidden), so
     * the loop ({@link Gemma4Speculative}) drives {@link Gemma4MtpDecoder} directly and this seam
     * only reports {@link #depth()}. Reading per-head logits through the seam would recompute whole
     * draft chains; it stays unimplemented until the seam is reshaped drafter-style
     * (draftSeed/draftNext) in core.
     */
    @Override
    public FloatTensor logits(State s, int output, int head) {
        throw new UnsupportedOperationException(
                "drive Gemma4MtpDecoder directly (see Gemma4Speculative)");
    }

    @Override
    public Set<Class<? extends Media>> modalities() {
        Set<Class<? extends Media>> m =
                new java.util.HashSet<>(); // adapters loaded iff an mmproj carried them
        if (vision != null) m.add(Media.Image.class);
        if (audio != null) m.add(Media.Audio.class);
        return m;
    }

    @Override
    @SuppressWarnings("unchecked")
    public <R extends Media> Optional<Embedder<R>> embedder(Class<R> modality) {
        if (vision != null && modality == Media.Image.class)
            return Optional.of((Embedder<R>) vision);
        if (audio != null && modality == Media.Audio.class) return Optional.of((Embedder<R>) audio);
        return Optional.empty();
    }

    /** Convenience for callers/tests: the turn-delimiter / eos ids that terminate generation. */
    public java.util.Set<Integer> stopTokens() {
        java.util.Set<Integer> stops = new java.util.HashSet<>();
        for (String name : new String[] {"<turn|>", "<end_of_turn>", "<eos>", "<|endoftext|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        return stops;
    }

    private com.qxotic.jinfer.chat.TurnTemplate
            turnTemplate; // memoized: stateless, model-lifetime (pins any construction-time state)

    @Override
    public java.util.Optional<com.qxotic.jinfer.chat.TurnTemplate> turnTemplate() {
        if (turnTemplate == null)
            turnTemplate = new Gemma4TurnTemplate(tokenizer(), this, config().embeddingLength());
        return java.util.Optional.of(turnTemplate);
    }

    @Override
    public java.util.Optional<com.qxotic.jinfer.cache.StateCodec<Gemma4.State>> stateCodec() {
        return java.util.Optional.of(new Gemma4StateCodec(config()));
    }

    // -Dgemma.trace: per-layer residual checkpoints matching llama.cpp's eval-callback node names.

    // === Configuration ===

    public record Configuration(
            int embeddingLength,
            int[] feedForwardLength, // per-layer (shared MLP)
            int numberOfLayers,
            int numberOfHeads,
            int[] numberOfKeyValueHeadsPerLayer, // per-layer KV head count
            int vocabularySize,
            int contextLength, // model ceiling (gemma4.context_length)
            float rmsNormEps,
            float ropeThetaFull, // full-attention RoPE theta
            float ropeThetaSWA, // sliding-window RoPE theta
            int headSizeFull, // head size for full-attention layers
            int headSizeSWA, // head size for sliding-window layers
            int slidingWindow, // power of 2
            float logitSoftcapping,
            boolean[] isSWA, // per-layer: true = sliding window, false = full
            int ownKvLayers, // first N layers own their KV; the rest reuse it
            int embeddingLengthPerLayer, // per-layer embedding dim (0 = disabled)
            int expertCount, // 0 = dense
            int expertUsedCount, // top-k experts per token
            int expertFeedForwardLength) // expert FFN hidden dim
            implements Config {

        public Configuration {
            if (slidingWindow <= 0 || Integer.bitCount(slidingWindow) != 1) {
                throw new IllegalArgumentException(
                        "slidingWindow must be a power of 2, got " + slidingWindow);
            }
        }

        boolean isMoE() {
            return expertCount > 0;
        }

        boolean hasKv(int layer) {
            return layer < ownKvLayers;
        }

        /**
         * For a layer without its own KV, the earlier layer whose cache it reuses (last own-KV
         * layer of the same attention type: SWA -> ownKvLayers-2, full -> -1).
         */
        int kvSourceLayer(int layer) {
            if (layer < ownKvLayers) return layer;
            return ownKvLayers - (isSWA[layer] ? 2 : 1);
        }

        int headSize(int layer) {
            return isSWA[layer] ? headSizeSWA : headSizeFull;
        }

        /**
         * KV ring slots a state with this {@code contextCapacity} needs for {@code layer}: the
         * capacity for full layers, capped at the window for SWA layers (the ring never holds more
         * than the window, and a sub-window capacity never wraps the {@code & (slidingWindow-1)}
         * index).
         */
        int kvCachePositions(int layer, int contextCapacity) {
            return isSWA[layer] ? Math.min(contextCapacity, slidingWindow) : contextCapacity;
        }

        int kvCacheIndex(int layer, int position) {
            return isSWA[layer] ? (position & (slidingWindow - 1)) : position;
        }

        int kvDim(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] * headSize(layer);
        }

        int queryDim(int layer) {
            return numberOfHeads * headSize(layer);
        }

        int maxHiddenDim() {
            return Arrays.stream(feedForwardLength).max().orElseThrow();
        }
    }

    // === Weights ===

    public static final class Weights {
        final FloatTensor tokenEmbeddings;
        final F32FloatTensor[] rmsAttentionWeights;
        final FloatTensor[] queryWeights,
                keyWeights,
                valueWeights,
                outputWeights; // valueWeights[l] null => V = K
        final F32FloatTensor[] queryNormWeights, keyNormWeights, rmsPostAttentionWeights;
        final F32FloatTensor[] rmsFFNWeights;
        final FloatTensor[] ffnGate, ffnDown, ffnUp;
        final F32FloatTensor[] rmsPostFFNWeights;
        final F32FloatTensor rmsFinalWeights;
        final float[] layerOutputScales;
        final F32FloatTensor ropeRealFull, ropeImagFull, ropeRealSWA, ropeImagSWA;
        final FloatTensor classifierWeights;
        // Per-layer embeddings (null when absent)
        final FloatTensor perLayerTokenEmbeddings, perLayerModelProjection;
        final F32FloatTensor rmsPerLayerProjectionWeights;
        final FloatTensor[] perLayerInputGate, perLayerProjection;
        final F32FloatTensor[] rmsPerLayerPostWeights;
        // MoE (null for dense models)
        final FloatTensor[] expertRouter, expertGateUp, expertDown;
        final F32FloatTensor[] expertRouterScale,
                expertDownScale,
                rmsExpertPostNorm1,
                rmsExpertPreNorm2,
                rmsExpertPostNorm2;

        Weights(
                FloatTensor tokenEmbeddings,
                F32FloatTensor[] rmsAttentionWeights,
                FloatTensor[] queryWeights,
                FloatTensor[] keyWeights,
                FloatTensor[] valueWeights,
                FloatTensor[] outputWeights,
                F32FloatTensor[] queryNormWeights,
                F32FloatTensor[] keyNormWeights,
                F32FloatTensor[] rmsPostAttentionWeights,
                F32FloatTensor[] rmsFFNWeights,
                FloatTensor[] ffnGate,
                FloatTensor[] ffnDown,
                FloatTensor[] ffnUp,
                F32FloatTensor[] rmsPostFFNWeights,
                F32FloatTensor rmsFinalWeights,
                float[] layerOutputScales,
                F32FloatTensor ropeRealFull,
                F32FloatTensor ropeImagFull,
                F32FloatTensor ropeRealSWA,
                F32FloatTensor ropeImagSWA,
                FloatTensor classifierWeights,
                FloatTensor perLayerTokenEmbeddings,
                FloatTensor perLayerModelProjection,
                F32FloatTensor rmsPerLayerProjectionWeights,
                FloatTensor[] perLayerInputGate,
                FloatTensor[] perLayerProjection,
                F32FloatTensor[] rmsPerLayerPostWeights,
                FloatTensor[] expertRouter,
                F32FloatTensor[] expertRouterScale,
                FloatTensor[] expertGateUp,
                FloatTensor[] expertDown,
                F32FloatTensor[] expertDownScale,
                F32FloatTensor[] rmsExpertPostNorm1,
                F32FloatTensor[] rmsExpertPreNorm2,
                F32FloatTensor[] rmsExpertPostNorm2) {
            this.tokenEmbeddings = tokenEmbeddings;
            this.rmsAttentionWeights = rmsAttentionWeights;
            this.queryWeights = queryWeights;
            this.keyWeights = keyWeights;
            this.valueWeights = valueWeights;
            this.outputWeights = outputWeights;
            this.queryNormWeights = queryNormWeights;
            this.keyNormWeights = keyNormWeights;
            this.rmsPostAttentionWeights = rmsPostAttentionWeights;
            this.rmsFFNWeights = rmsFFNWeights;
            this.ffnGate = ffnGate;
            this.ffnDown = ffnDown;
            this.ffnUp = ffnUp;
            this.rmsPostFFNWeights = rmsPostFFNWeights;
            this.rmsFinalWeights = rmsFinalWeights;
            this.layerOutputScales = layerOutputScales;
            this.ropeRealFull = ropeRealFull;
            this.ropeImagFull = ropeImagFull;
            this.ropeRealSWA = ropeRealSWA;
            this.ropeImagSWA = ropeImagSWA;
            this.classifierWeights = classifierWeights;
            this.perLayerTokenEmbeddings = perLayerTokenEmbeddings;
            this.perLayerModelProjection = perLayerModelProjection;
            this.rmsPerLayerProjectionWeights = rmsPerLayerProjectionWeights;
            this.perLayerInputGate = perLayerInputGate;
            this.perLayerProjection = perLayerProjection;
            this.rmsPerLayerPostWeights = rmsPerLayerPostWeights;
            this.expertRouter = expertRouter;
            this.expertRouterScale = expertRouterScale;
            this.expertGateUp = expertGateUp;
            this.expertDown = expertDown;
            this.expertDownScale = expertDownScale;
            this.rmsExpertPostNorm1 = rmsExpertPostNorm1;
            this.rmsExpertPreNorm2 = rmsExpertPreNorm2;
            this.rmsExpertPostNorm2 = rmsExpertPostNorm2;
        }
    }

    // === State ===

    public static final class State extends com.qxotic.jinfer.BaseState {
        // Batched scratch (batchCapacity rows): the residual stream and projections hold one row
        // per
        // token in the current chunk; KV cache is the cross-row source of truth. (MoE buffers and
        // the single-token reference scratch from the production State are dropped here.)
        final int contextCapacity, batchCapacity;
        final FloatTensor residual, xb, xbK, xb2, hb, hb2, query, logits;
        int[] lastTokens; // ids of the last ingested token batch (row->token, for the MTP draft
        // seam)
        final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch();
        final FloatTensor[] keyCache, valueCache; // own-KV layers only (ring/linear F16)
        final FloatTensor[] batchK, batchV; // current chunk's linear K/V, committed at chunk end
        final FloatTensor perLayerInputs, plGate, plProj;
        // MoE scratch (chunk-wide CSR routing + expert/shared buffers); allocated only for MoE
        // (A4B), else null.
        final FloatTensor moeShared,
                moeInputB,
                moeRouterScaled,
                moeRouterB,
                moeOutB,
                moeGather,
                moeDownB;
        final int[] moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeRowTopE;
        final float[] moeProbByExpert, moeRowTopP;
        final Moe.Routing moeRouting;

        State(Configuration config, int contextCapacity, int batchCapacity) {
            if (contextCapacity > config.contextLength()) {
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds model maxContextLength "
                                + config.contextLength());
            }
            this.contextCapacity = contextCapacity;
            int c = Math.max(1, batchCapacity);
            this.batchCapacity = c;
            int dim = config.embeddingLength();
            int maxQueryDim = config.numberOfHeads() * config.headSizeFull();
            int maxHiddenDim = config.maxHiddenDim();
            this.residual = FloatTensor.allocateF32(c * dim);
            this.xb = FloatTensor.allocateF32(c * dim);
            this.xbK = FloatTensor.allocateF32(c * maxQueryDim);
            this.xb2 = FloatTensor.allocateF32(c * dim);
            this.hb = FloatTensor.allocateF32(c * maxHiddenDim);
            this.hb2 = FloatTensor.allocateF32(c * maxHiddenDim);
            this.query = FloatTensor.allocateF32(c * maxQueryDim);
            this.logits = FloatTensor.allocateF32(config.vocabularySize());
            int plDim = config.embeddingLengthPerLayer();
            this.perLayerInputs =
                    plDim > 0 ? FloatTensor.allocateF32(c * plDim * config.numberOfLayers()) : null;
            this.plGate = plDim > 0 ? FloatTensor.allocateF32(c * plDim) : null;
            this.plProj = plDim > 0 ? FloatTensor.allocateF32(c * dim) : null;
            this.keyCache = new FloatTensor[config.ownKvLayers()];
            this.valueCache = new FloatTensor[config.ownKvLayers()];
            this.batchK = new FloatTensor[config.ownKvLayers()];
            this.batchV = new FloatTensor[config.ownKvLayers()];
            for (int l = 0; l < config.ownKvLayers(); l++) {
                int kvDim = config.kvDim(l);
                int kvPositions = config.kvCachePositions(l, contextCapacity);
                keyCache[l] = FloatTensor.allocateF16(kvPositions, kvDim);
                valueCache[l] = FloatTensor.allocateF16(kvPositions, kvDim);
                batchK[l] = FloatTensor.allocateF32(c * kvDim);
                batchV[l] = FloatTensor.allocateF32(c * kvDim);
            }
            if (config.isMoE()) {
                int e = config.expertCount(), tk = config.expertUsedCount();
                this.moeShared = FloatTensor.allocateF32(c * dim);
                this.moeInputB = FloatTensor.allocateF32(c * dim);
                this.moeRouterScaled = FloatTensor.allocateF32(c * dim);
                this.moeRouterB = FloatTensor.allocateF32(c * e);
                this.moeOutB = FloatTensor.allocateF32(c * dim);
                this.moeGather = FloatTensor.allocateF32(c * dim);
                this.moeDownB = FloatTensor.allocateF32(c * dim);
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
                this.moeShared = this.moeInputB = this.moeRouterScaled = this.moeRouterB = null;
                this.moeOutB = this.moeGather = this.moeDownB = null;
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

    // === Forward (gated activation + softcap run through the shared vectorized Activations
    // kernels) ===

    /**
     * Single forward pass over {@code seqLen} tokens at {@code [startPos, startPos+seqLen)}:
     * per-token Q/K/V/O, shared MLP and PLE projections become GEMMs (GEMM short-circuits to GEMV
     * at seqLen=1, so one decode token flows through the same path), K/V written to the cache
     * before attention, which dispatches scalar single-token vs SWA flash per chunk. Leaves the
     * post-final-layer residual for every row in {@code state.residual}; {@link #logits} finalizes
     * the retained row(s).
     */
    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        embed(state, tokens, tokenOffset, seqLen);
        buildPerLayerInputs(state, tokens, tokenOffset, seqLen);
        for (int l = 0; l < configuration.numberOfLayers(); l++) {
            layer(state, l, startPos, seqLen, false);
        }
        commitKv(state, startPos, seqLen);
    }

    /**
     * Forward an externally-produced embedding chunk (projected vision rows) at {@code startPos}.
     * The rows go into the residual UNSCALED — the vision projector already sets the magnitude,
     * unlike token embeddings which are ×sqrt(dim) — and the per-layer-embedding (PLE) path is
     * keyed on the image {@code <pad>} token (gemma fills the per-layer-token-embedding for image
     * positions from the pad id). Same batched stack.
     */
    void forwardEmbeddings(
            State state,
            FloatTensor rows,
            int pleToken,
            int startPos,
            int seqLen,
            boolean bidirectional) {
        int dim = configuration.embeddingLength();
        rows.copyTo(0, state.residual, 0, seqLen * dim);
        int[] ple = new int[seqLen];
        java.util.Arrays.fill(ple, pleToken);
        buildPerLayerInputs(state, ple, 0, seqLen);
        // Image soft tokens attend bidirectionally (mtmd_decode_use_non_causal is true for
        // GEMMA4V/GEMMA4UV):
        // every image token sees every other, so the LLM sees the whole image at once. Audio
        // (GEMMA4UA) is
        // causal, so the caller passes bidirectional=false for it.
        for (int l = 0; l < configuration.numberOfLayers(); l++)
            layer(state, l, startPos, seqLen, bidirectional);
        commitKv(state, startPos, seqLen);
    }

    /** Token-embedding lookup into the residual stream (all rows), scaled by {@code sqrt(dim)}. */
    private void embed(State state, int[] tokens, int tokenOffset, int seqLen) {
        int dim = configuration.embeddingLength();
        float sqrtDim = (float) Math.sqrt(dim);
        for (int s = 0; s < seqLen; s++) {
            int token = tokens[tokenOffset + s];
            weights.tokenEmbeddings.copyTo((long) token * dim, state.residual, s * dim, dim);
        }
        state.residual.mapInPlace(0, seqLen * dim, v -> v * sqrtDim);
        LLM.traceSum("inp_scaled", state.residual, seqLen * dim);
    }

    /**
     * Per-layer embeddings (Gemma-3n PLE): project the scaled residual to the per-layer width,
     * RMS-norm each layer's slice, and fold in the per-layer token embedding. No-op when the model
     * has no PLE.
     */
    private void buildPerLayerInputs(State state, int[] tokens, int tokenOffset, int seqLen) {
        Configuration config = configuration;
        int dim = config.embeddingLength(), numLayers = config.numberOfLayers();
        int plDim = config.embeddingLengthPerLayer(), plTotal = plDim * numLayers;
        if (plDim == 0 || weights.perLayerTokenEmbeddings == null) return;
        float sqrtPlDim = (float) Math.sqrt(plDim);
        float projScale = (float) (1.0 / Math.sqrt(dim));
        float inputScale = (float) (1.0 / Math.sqrt(2.0));
        weights.perLayerModelProjection.gemm(
                state.residual, dim, state.perLayerInputs, plTotal, seqLen, plTotal, dim);
        Parallel.forRows(
                seqLen,
                s -> { // per-layer RMS-norm + token-embedding add, parallel over rows
                    int base = s * plTotal;
                    for (int i = 0; i < plTotal; i++)
                        state.perLayerInputs.setFloat(
                                base + i, state.perLayerInputs.getFloat(base + i) * projScale);
                    for (int l = 0; l < numLayers; l++) {
                        rmsnorm(
                                state.perLayerInputs,
                                base + l * plDim,
                                state.perLayerInputs,
                                base + l * plDim,
                                weights.rmsPerLayerProjectionWeights,
                                plDim,
                                config.rmsNormEps());
                    }
                    long tokEmbOffset = (long) tokens[tokenOffset + s] * plTotal;
                    for (int i = 0; i < plTotal; i++) {
                        float tokEmb =
                                weights.perLayerTokenEmbeddings.getFloat(tokEmbOffset + i)
                                        * sqrtPlDim;
                        state.perLayerInputs.setFloat(
                                base + i,
                                (state.perLayerInputs.getFloat(base + i) + tokEmb) * inputScale);
                    }
                });
    }

    /**
     * One decoder block, in place on {@code state.residual}: attention, FFN, the optional PLE
     * projection, then the per-layer output scale.
     */
    private void layer(State state, int l, int startPos, int seqLen, boolean nonCausal) {
        attention(state, l, startPos, seqLen, nonCausal);
        feedForward(state, l, seqLen);
        mergePerLayerInput(state, l, seqLen);
        float scale = weights.layerOutputScales[l];
        if (scale != 1.0f) {
            state.residual.mapInPlace(0, seqLen * configuration.embeddingLength(), v -> v * scale);
        }
        if (LLM.TRACE)
            LLM.traceSum("l_out-" + l, state.residual, seqLen * configuration.embeddingLength());
    }

    /**
     * Pre-norm attention: per-head Q/K RMS-norm + RoPE, ring-SWA or full causal attention (the QK
     * scale is folded into the Q-norm), output projection, post-norm, added back to the residual.
     * K/V land in the chunk buffer (committed to the ring by {@link #commitKv}); shared-KV tail
     * layers reuse an earlier layer's cache, so they project Q only.
     */
    private void attention(State state, int l, int startPos, int seqLen, boolean nonCausal) {
        Configuration config = configuration;
        int dim = config.embeddingLength();
        float eps = config.rmsNormEps();
        boolean swa = config.isSWA()[l];
        int headSize = config.headSize(l), halfHead = headSize / 2;
        int queryDim = config.queryDim(l), kvDim = config.kvDim(l);
        int nKvHeads = config.numberOfKeyValueHeadsPerLayer()[l],
                kvMul = config.numberOfHeads() / nKvHeads;
        int kvLayer = config.kvSourceLayer(l);
        F32FloatTensor real = swa ? weights.ropeRealSWA : weights.ropeRealFull;
        F32FloatTensor imag = swa ? weights.ropeImagSWA : weights.ropeImagFull;

        // pre-attention norm (rows independent -> parallel)
        F32FloatTensor attNormW = weights.rmsAttentionWeights[l];
        Parallel.forRows(
                seqLen,
                s -> rmsnorm(state.xb, s * dim, state.residual, s * dim, attNormW, dim, eps));

        // Q projection, per-head Q-norm + RoPE
        weights.queryWeights[l].gemm(state.xb, dim, state.query, queryDim, seqLen, queryDim, dim);
        headNormRope(
                state.query,
                queryDim,
                config.numberOfHeads(),
                headSize,
                halfHead,
                weights.queryNormWeights[l],
                startPos,
                seqLen,
                real,
                imag);

        // K/V projection into the per-layer LINEAR batch buffer (own-KV layers): K-norm + RoPE, V
        // no-weight norm. NOT written to the ring yet — committed at chunk end so prior reads stay
        // intact.
        if (config.hasKv(l)) {
            FloatTensor bKl = state.batchK[l], bVl = state.batchV[l];
            weights.keyWeights[l].gemm(state.xb, dim, bKl, kvDim, seqLen, kvDim, dim);
            if (weights.valueWeights[l] != null) {
                weights.valueWeights[l].gemm(state.xb, dim, bVl, kvDim, seqLen, kvDim, dim);
            } else {
                bKl.copyTo(0, bVl, 0, seqLen * kvDim);
            }
            headNormRope(
                    bKl,
                    kvDim,
                    nKvHeads,
                    headSize,
                    halfHead,
                    weights.keyNormWeights[l],
                    startPos,
                    seqLen,
                    real,
                    imag);
            Parallel.forRows(
                    seqLen,
                    s -> {
                        for (int h = 0; h < nKvHeads; h++) {
                            rmsnormNoWeight(
                                    bVl,
                                    s * kvDim + h * headSize,
                                    bVl,
                                    s * kvDim + h * headSize,
                                    headSize,
                                    eps);
                        }
                    });
        }

        // Batched causal attention (scale = 1.0): flash/tiled for prefill, simple 2-pass for decode
        // (a single query doesn't amortize the flash tiling/rescales).
        if (seqLen > 1) {
            flashAttention(
                    state, l, startPos, seqLen, headSize, kvDim, queryDim, kvLayer, kvMul, swa,
                    nonCausal);
        } else {
            decodeAttention(state, l, startPos, headSize, kvDim, queryDim, kvLayer, kvMul, swa);
        }

        // O = outputWeights @ xbK (GEMM), post-attention norm + residual
        weights.outputWeights[l].gemm(state.xbK, queryDim, state.xb2, dim, seqLen, dim, queryDim);
        F32FloatTensor postAttW = weights.rmsPostAttentionWeights[l];
        Parallel.forRows(
                seqLen, s -> rmsnorm(state.xb2, s * dim, state.xb2, s * dim, postAttW, dim, eps));
        state.residual.addInPlace(0, state.xb2, 0, seqLen * dim);
    }

    /**
     * Per-head RMS-norm then NeoX RoPE over each row of {@code t}: {@code nHeads} heads of {@code
     * headSize}, rows {@code rowStride} apart, rotated at absolute position {@code startPos + row}.
     * Shared by the Q and K projections.
     */
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
        float eps = configuration.rmsNormEps();
        Parallel.forRows(
                seqLen,
                s -> {
                    for (int h = 0; h < nHeads; h++) {
                        int off = s * rowStride + h * headSize;
                        rmsnorm(t, off, t, off, normW, headSize, eps);
                    }
                    RoPE.applyNeox(
                            t, s * rowStride, nHeads, headSize, halfHead, startPos + s, real, imag);
                });
    }

    /**
     * Position-wise FFN over the chunk, added to the residual: dense GeGLU, or the shared-MLP +
     * top-k expert MoE (A4B) when the layer routes.
     */
    private void feedForward(State state, int l, int seqLen) {
        Configuration config = configuration;
        int dim = config.embeddingLength(), hiddenDim = config.feedForwardLength()[l];
        if (config.isMoE() && weights.expertRouter[l] != null) {
            moeFeedForward(state, l, dim, hiddenDim, seqLen);
            return;
        }
        float eps = config.rmsNormEps();
        F32FloatTensor ffnNormW = weights.rmsFFNWeights[l], postFfwW = weights.rmsPostFFNWeights[l];
        Parallel.forRows(
                seqLen,
                s -> rmsnorm(state.xb, s * dim, state.residual, s * dim, ffnNormW, dim, eps));
        weights.ffnGate[l].gemm(state.xb, dim, state.hb, hiddenDim, seqLen, hiddenDim, dim);
        weights.ffnUp[l].gemm(state.xb, dim, state.hb2, hiddenDim, seqLen, hiddenDim, dim);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.geluMultiply(
                                state.hb, s * hiddenDim, state.hb2, s * hiddenDim, hiddenDim));
        weights.ffnDown[l].gemm(state.hb, hiddenDim, state.xb, dim, seqLen, dim, hiddenDim);
        Parallel.forRows(
                seqLen, s -> rmsnorm(state.xb, s * dim, state.xb, s * dim, postFfwW, dim, eps));
        state.residual.addInPlace(0, state.xb, 0, seqLen * dim);
    }

    /**
     * Per-layer embedding (Gemma-3n PLE): GELU-gated projection of this layer's PLE slice, added to
     * the residual. No-op when the model has no PLE.
     */
    private void mergePerLayerInput(State state, int l, int seqLen) {
        Configuration config = configuration;
        int dim = config.embeddingLength();
        int plDim = config.embeddingLengthPerLayer();
        if (plDim == 0 || weights.perLayerInputGate == null) return;
        float eps = config.rmsNormEps();
        int plTotal = plDim * config.numberOfLayers();
        F32FloatTensor plPostW = weights.rmsPerLayerPostWeights[l];
        weights.perLayerInputGate[l].gemm(
                state.residual, dim, state.plGate, plDim, seqLen, plDim, dim);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.geluMultiply(
                                state.plGate,
                                s * plDim,
                                state.perLayerInputs,
                                s * plTotal + l * plDim,
                                plDim));
        weights.perLayerProjection[l].gemm(
                state.plGate, plDim, state.plProj, dim, seqLen, dim, plDim);
        Parallel.forRows(
                seqLen,
                s -> rmsnorm(state.plProj, s * dim, state.plProj, s * dim, plPostW, dim, eps));
        state.residual.addInPlace(0, state.plProj, 0, seqLen * dim);
    }

    /**
     * Write the chunk's K/V into the ring caches (own-KV layers) in position order; later positions
     * overwrite the oldest ring slots, leaving the last {@code window} positions live for SWA.
     */
    private void commitKv(State state, int startPos, int seqLen) {
        Configuration config = configuration;
        for (int l = 0; l < config.ownKvLayers(); l++) {
            int kvDim = config.kvDim(l);
            for (int s = 0; s < seqLen; s++) {
                int kvPos = config.kvCacheIndex(l, startPos + s);
                state.batchK[l].copyTo(s * kvDim, state.keyCache[l], kvPos * kvDim, kvDim);
                state.batchV[l].copyTo(s * kvDim, state.valueCache[l], kvPos * kvDim, kvDim);
            }
        }
    }

    /**
     * Gemma4 adapter: ring-SWA (or full) flash attention via the shared {@link
     * FlashAttention#slidingWindowPrefill} block. {@code scale = 1.0} (Gemma folds the QK scale
     * into the per-head Q-norm), no attention sinks. SWA layers ring their KV cache (slot = {@code
     * pos & (slidingWindow-1)}, power-of-two enforced in {@link Configuration}); full layers store
     * linearly. The batch K/V buffer is stride {@code kvDim}.
     */
    private void flashAttention(
            State state,
            int layer,
            int startPos,
            int seqLen,
            int headSize,
            int kvDim,
            int queryDim,
            int kvLayer,
            int kvMul,
            boolean isSWA,
            boolean nonCausal) {
        int window = isSWA ? configuration.slidingWindow() : 0;
        int ringMask = isSWA ? configuration.slidingWindow() - 1 : 0;
        FlashAttention.slidingWindowPrefill(
                state.query,
                state.xbK,
                state.keyCache[kvLayer],
                state.valueCache[kvLayer],
                state.batchK[kvLayer],
                state.batchV[kvLayer],
                configuration.numberOfHeads(),
                startPos,
                seqLen,
                headSize,
                kvDim,
                queryDim,
                kvDim,
                kvMul,
                1.0f,
                window,
                ringMask,
                null,
                nonCausal);
    }

    /**
     * Rolling (online-softmax) causal/windowed attention for one query (decode), scale = 1.0:
     * parallel over heads, streaming keys with a running max/sum so the score row is never
     * materialized (no {@code att} scratch). Reads the single in-chunk key from the batch buffer
     * and prior keys from the ring cache. Mathematically identical to the two-pass softmax — and to
     * the flash prefill path ({@link #flashAttention}), which already uses online softmax.
     */
    private void decodeAttention(
            State state,
            int layer,
            int position,
            int headSize,
            int kvDim,
            int queryDim,
            int kvLayer,
            int kvMul,
            boolean isSWA) {
        int window = configuration.slidingWindow();
        int attStart = isSWA ? Math.max(0, position - window + 1) : 0;
        FlashAttention.flashDecode(
                (F32FloatTensor) state.query,
                (F32FloatTensor) state.xbK,
                state.keyCache[kvLayer],
                state.valueCache[kvLayer],
                state.batchK[kvLayer],
                state.batchV[kvLayer],
                configuration.numberOfHeads(),
                position,
                attStart,
                headSize,
                kvDim,
                kvMul,
                1.0f,
                isSWA ? window - 1 : 0,
                null,
                state.decodeScratch);
    }

    /**
     * Shared dense MLP + top-k expert MoE FFN (the A4B variant), batched across the chunk. Shared
     * MLP runs as batched GEMMs; the expert FFN groups rows by routed expert (CSR, via {@link
     * Moe#dispatch}) so each expert's weights are read once per chunk. Output is {@code
     * post_ffw_norm(post_norm_1(shared) + post_norm_2(experts))}, added to the residual.
     */
    private void moeFeedForward(State state, int l, int dim, int hiddenDim, int seqLen) {
        Configuration config = configuration;
        Weights weights = this.weights;
        float eps = config.rmsNormEps();
        int nExperts = config.expertCount(),
                topK = config.expertUsedCount(),
                expertFF = config.expertFeedForwardLength();
        int gateUpDim = 2 * expertFF;
        F32FloatTensor gateInpScale = weights.expertRouterScale[l];
        float invSqrtDim = 1.0f / (float) Math.sqrt(dim);

        // Shared MLP (batched): ffn_norm -> gate/up/down -> post_norm_1 -> moeShared
        F32FloatTensor ffnNormW = weights.rmsFFNWeights[l],
                postNorm1 = weights.rmsExpertPostNorm1[l];
        Parallel.forRows(
                seqLen,
                s -> rmsnorm(state.xb, s * dim, state.residual, s * dim, ffnNormW, dim, eps));
        weights.ffnGate[l].gemm(state.xb, dim, state.hb, hiddenDim, seqLen, hiddenDim, dim);
        weights.ffnUp[l].gemm(state.xb, dim, state.hb2, hiddenDim, seqLen, hiddenDim, dim);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.geluMultiply(
                                state.hb, s * hiddenDim, state.hb2, s * hiddenDim, hiddenDim));
        weights.ffnDown[l].gemm(state.hb, hiddenDim, state.moeShared, dim, seqLen, dim, hiddenDim);
        Parallel.forRows(
                seqLen,
                s ->
                        rmsnorm(
                                state.moeShared,
                                s * dim,
                                state.moeShared,
                                s * dim,
                                postNorm1,
                                dim,
                                eps));

        // Expert routing: both the expert input (pre_ffw_norm2) and the router input normalize the
        // same
        // residual row by its RMS — compute the sum of squares once and apply the shared 1/rms
        // scale to
        // both (expert input keeps the norm weight; router input is rms·invSqrtDim·gateInpScale).
        F32FloatTensor preNorm2 = weights.rmsExpertPreNorm2[l];
        Parallel.forRows(
                seqLen,
                s -> {
                    float rms =
                            (float)
                                    (1.0
                                            / Math.sqrt(
                                                    sumOfSquares(state.residual, s * dim, dim) / dim
                                                            + eps));
                    scaleByWeight(
                            state.moeInputB, s * dim, state.residual, s * dim, preNorm2, dim, rms);
                    scaleByWeight(
                            state.moeRouterScaled,
                            s * dim,
                            state.residual,
                            s * dim,
                            gateInpScale,
                            dim,
                            rms * invSqrtDim);
                });
        weights.expertRouter[l].gemm(
                state.moeRouterScaled, dim, state.moeRouterB, nExperts, seqLen, nExperts, dim);

        // Per-row softmax + top-k selection; bucket (row, prob) by expert into CSR counts.
        int[] counts = state.moeExpertCounts;
        Arrays.fill(counts, 0);
        for (int s = 0; s < seqLen; s++) {
            state.moeRouterB.softmaxInPlace(s * nExperts, nExperts);
            for (int ki = 0; ki < topK; ki++) {
                int bestIdx = 0;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int ei = 0; ei < nExperts; ei++) {
                    float val = state.moeRouterB.getFloat(s * nExperts + ei);
                    if (val > bestVal) {
                        bestVal = val;
                        bestIdx = ei;
                    }
                }
                state.moeRowTopE[s * topK + ki] = bestIdx;
                state.moeRowTopP[s * topK + ki] = bestVal;
                state.moeRouterB.setFloat(s * nExperts + bestIdx, Float.NEGATIVE_INFINITY);
                counts[bestIdx]++;
            }
        }
        Moe.Routing r = state.moeRouting;
        r.seqLen = seqLen;
        r.topK = topK;
        r.numExperts = nExperts;
        Moe.dispatch(
                r,
                dim,
                state.moeInputB,
                state.moeGather,
                state.moeDownB,
                state.moeOutB,
                weights.expertDownScale[l],
                (e, n, gather, out) -> {
                    weights.expertGateUp[l].gemm(
                            gather,
                            dim,
                            state.hb,
                            gateUpDim,
                            n,
                            gateUpDim,
                            dim,
                            (long) e * gateUpDim * dim);
                    Parallel.forRows(
                            n,
                            j ->
                                    Activations.geluMultiply(
                                            state.hb,
                                            j * gateUpDim,
                                            state.hb,
                                            j * gateUpDim + expertFF,
                                            expertFF));
                    weights.expertDown[l].gemm(
                            state.hb,
                            gateUpDim,
                            out,
                            dim,
                            n,
                            dim,
                            expertFF,
                            (long) e * dim * expertFF);
                });

        // post_norm_2(experts) + shared, then post_ffw_norm, added to the residual (per row).
        F32FloatTensor postNorm2 = weights.rmsExpertPostNorm2[l],
                postFfw = weights.rmsPostFFNWeights[l];
        Parallel.forRows(
                seqLen,
                s -> {
                    rmsnorm(state.moeOutB, s * dim, state.moeOutB, s * dim, postNorm2, dim, eps);
                    state.moeShared.addInPlace(s * dim, state.moeOutB, s * dim, dim);
                    rmsnorm(state.moeShared, s * dim, state.moeShared, s * dim, postFfw, dim, eps);
                    state.residual.addInPlace(s * dim, state.moeShared, s * dim, dim);
                });
    }

    // === Loading ===

    public static Gemma4 loadModel(Path ggufPath, int maxContextLength) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, maxContextLength, true);
        }
    }

    /**
     * Multimodal load: the text model plus the paired vision encoder from an mmproj GGUF, which
     * enables the image {@link Embedder} ({@link #modalities()} then contains {@link Media.Image}).
     */
    public static Gemma4 loadModel(Path textGguf, Path mmprojGguf, int maxContextLength)
            throws IOException {
        Gemma4 model = loadModel(textGguf, maxContextLength);
        model.vision = loadVision(mmprojGguf);
        model.audio = loadAudio(mmprojGguf);
        return model;
    }

    /**
     * MTP load: the text model plus the {@code gemma4-assistant} draft sidecar, which enables
     * self-speculative decoding ({@link #depth()} becomes present).
     */
    public static Gemma4 loadModel(Path textGguf, int maxContextLength, Path mtpSidecar)
            throws IOException {
        Gemma4 model = loadModel(textGguf, maxContextLength);
        model.mtp = Gemma4Mtp.loadSidecar(mtpSidecar, model.config().vocabularySize());
        model.mtpDecoder = new Gemma4MtpDecoder(model.mtp, model);
        return model;
    }

    /**
     * The paired MTP draft forward, or null when no sidecar is loaded. Single-threaded (owns
     * scratch); {@link Gemma4Speculative} is the decode loop over it.
     */
    Gemma4MtpDecoder mtpDecoder() {
        return mtpDecoder;
    }

    /**
     * Pick the image encoder that matches the mmproj's projector_type: {@code gemma4uv} -> the
     * minimal {@link Gemma4VisionUnified} (12b), otherwise the full-ViT {@link Gemma4Vision}
     * (E2B/gemma4v).
     */
    private static Embedder<Media.Image> loadVision(Path mmprojGguf) throws IOException {
        String type;
        try (FileChannel fc = FileChannel.open(mmprojGguf, StandardOpenOption.READ)) {
            type =
                    ModelLoader.readGguf(fc, mmprojGguf.toString())
                            .getStringOrDefault("clip.vision.projector_type", "");
        }
        if (type.isEmpty()) return null; // audio-only mmproj: no vision adapter
        return "gemma4uv".equals(type)
                ? Gemma4VisionUnified.loadModel(mmprojGguf)
                : Gemma4Vision.loadModel(mmprojGguf);
    }

    /**
     * Load the audio encoder if the mmproj carries a {@code gemma4ua} adapter ({@link
     * Gemma4Audio}); else null. (The E2B {@code gemma4a} conformer is a different, unimplemented
     * projector, so only gemma4ua matches.)
     */
    private static Embedder<Media.Audio> loadAudio(Path mmprojGguf) throws IOException {
        String type;
        try (FileChannel fc = FileChannel.open(mmprojGguf, StandardOpenOption.READ)) {
            type =
                    ModelLoader.readGguf(fc, mmprojGguf.toString())
                            .getStringOrDefault("clip.audio.projector_type", "");
        }
        return "gemma4ua".equals(type) ? Gemma4Audio.loadModel(mmprojGguf) : null;
    }

    public static Gemma4 loadModel(
            FileChannel fileChannel, GGUF gguf, int maxContextLength, boolean loadWeightsFlag)
            throws IOException {
        GgufTokenizer tokenizer = new GgufTokenizer(gguf, JinjaRenderer::template);

        int modelContextLength = gguf.getValue(int.class, "gemma4.context_length");
        if (maxContextLength < 0 || modelContextLength < maxContextLength) {
            maxContextLength = modelContextLength;
        }
        int embeddingLength = gguf.getValue(int.class, "gemma4.embedding_length");
        int numberOfHeads = gguf.getValue(int.class, "gemma4.attention.head_count");
        // head_count_kv is a scalar on some checkpoints (E2B) and a per-layer int[] on others
        // (A4B).
        Object numberOfKeyValueHeadsRaw =
                gguf.getValueOrDefault(
                        Object.class, "gemma4.attention.head_count_kv", numberOfHeads);
        int numberOfLayers = gguf.getValue(int.class, "gemma4.block_count");
        int headSizeFull = gguf.getValue(int.class, "gemma4.attention.key_length");
        int headSizeSWA = gguf.getValue(int.class, "gemma4.attention.key_length_swa");
        int slidingWindow = gguf.getValue(int.class, "gemma4.attention.sliding_window");
        float logitSoftcapping =
                gguf.getValueOrDefault(float.class, "gemma4.final_logit_softcapping", 0f);
        float rmsNormEps =
                gguf.getValueOrDefault(
                        float.class, "gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeThetaFull =
                gguf.getValueOrDefault(float.class, "gemma4.rope.freq_base", 1000000f);
        float ropeThetaSWA =
                gguf.getValueOrDefault(float.class, "gemma4.rope.freq_base_swa", 10000f);
        int expertCount = gguf.getValueOrDefault(int.class, "gemma4.expert_count", 0);
        int expertUsedCount = gguf.getValueOrDefault(int.class, "gemma4.expert_used_count", 0);
        int expertFeedForwardLength =
                gguf.getValueOrDefault(int.class, "gemma4.expert_feed_forward_length", 0);
        int embeddingLengthPerLayer =
                gguf.getValueOrDefault(int.class, "gemma4.embedding_length_per_layer_input", 0);
        int sharedKvLayers =
                gguf.getValueOrDefault(int.class, "gemma4.attention.shared_kv_layers", 0);
        int ownKvLayers = numberOfLayers - sharedKvLayers;

        int[] feedForwardLength;
        Object ffnRaw = gguf.getValue(Object.class, "gemma4.feed_forward_length");
        if (ffnRaw instanceof int[] arr) {
            feedForwardLength = arr;
        } else {
            feedForwardLength = new int[numberOfLayers];
            Arrays.fill(feedForwardLength, (int) ffnRaw);
        }

        // Per-layer sliding-window vs full attention: prefer the explicit pattern, else derive
        // from the Q-norm size (== headSizeSWA for sliding-window layers).
        boolean[] isSWA = new boolean[numberOfLayers];
        Object swaRaw =
                gguf.getValueOrDefault(
                        Object.class, "gemma4.attention.sliding_window_pattern", null);
        if (swaRaw instanceof boolean[] arr && arr.length == numberOfLayers) {
            isSWA = arr;
        } else {
            for (int i = 0; i < numberOfLayers; i++) {
                TensorEntry qNorm = gguf.getTensor("blk." + i + ".attn_q_norm.weight");
                isSWA[i] =
                        qNorm != null
                                ? FloatTensor.numberOfElementsLong(
                                                Arrays.stream(qNorm.shape())
                                                        .mapToInt(Math::toIntExact)
                                                        .toArray())
                                        == headSizeSWA
                                : (i % 5 != 4);
            }
        }

        int[] numberOfKeyValueHeadsPerLayer = new int[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            TensorEntry kWeight = gguf.getTensor("blk." + i + ".attn_k.weight");
            int headSize = isSWA[i] ? headSizeSWA : headSizeFull;
            // Prefer the tensor shape; fall back to head_count_kv only when attn_k.weight is
            // absent, so a
            // (malformed) short per-layer array is never indexed for a layer that doesn't need it.
            numberOfKeyValueHeadsPerLayer[i] =
                    kWeight != null
                            ? Math.toIntExact(kWeight.shape()[1]) / headSize
                            : (numberOfKeyValueHeadsRaw instanceof int[] arr
                                    ? arr[i]
                                    : ((Number) numberOfKeyValueHeadsRaw).intValue());
        }

        Configuration config =
                new Configuration(
                        embeddingLength,
                        feedForwardLength,
                        numberOfLayers,
                        numberOfHeads,
                        numberOfKeyValueHeadsPerLayer,
                        tokenizer.vocabularySize(),
                        maxContextLength,
                        rmsNormEps,
                        ropeThetaFull,
                        ropeThetaSWA,
                        headSizeFull,
                        headSizeSWA,
                        slidingWindow,
                        logitSoftcapping,
                        isSWA,
                        ownKvLayers,
                        embeddingLengthPerLayer,
                        expertCount,
                        expertUsedCount,
                        expertFeedForwardLength);

        // Shared-KV tail layers index their source layer's cache, so the KV shapes must match
        // (fail loudly at load rather than mis-index a future model with a different layout).
        for (int l = ownKvLayers; l < numberOfLayers; l++) {
            int src = config.kvSourceLayer(l);
            if (src < 0 || src >= ownKvLayers || config.kvDim(l) != config.kvDim(src)) {
                throw new IllegalStateException(
                        "layer "
                                + l
                                + " reuses KV from layer "
                                + src
                                + " but their KV shapes differ (kvDim "
                                + config.kvDim(l)
                                + " vs "
                                + config.kvDim(src)
                                + ")");
            }
        }

        if (!loadWeightsFlag) {
            return new Gemma4(config, tokenizer, null);
        }
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        return new Gemma4(config, tokenizer, loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
        int n = config.numberOfLayers();
        Pair<float[], float[]> ropeSwa =
                RoPE.precomputeFreqsCis(
                        config.contextLength(), config.headSizeSWA(), config.ropeThetaSWA());
        float[] freqs = ModelLoader.ropeFreqFactors(tensors);
        Pair<float[], float[]> ropeFull =
                freqs != null
                        ? RoPE.precomputeFreqsCisFromFreqs(
                                config.contextLength(),
                                config.headSizeFull(),
                                config.ropeThetaFull(),
                                freqs)
                        : RoPE.precomputeFreqsCis(
                                config.contextLength(),
                                config.headSizeFull(),
                                config.ropeThetaFull());

        FloatTensor tokenEmbeddings = ModelLoader.loadQuantized(tensors.get("token_embd.weight"));

        float[] layerOutputScales = new float[n];
        for (int i = 0; i < n; i++) {
            GGMLTensorEntry scale = tensors.get("blk." + i + ".layer_output_scale.weight");
            layerOutputScales[i] =
                    scale != null ? ModelLoader.toF32Tensor(scale).getFloat(0) : 1.0f;
        }

        FloatTensor[] valueWeights = new FloatTensor[n];
        for (int i = 0; i < n; i++) {
            valueWeights[i] = ModelLoader.quantOrNull(tensors, "blk." + i + ".attn_v.weight");
        }

        // Per-layer embeddings (PLE)
        FloatTensor perLayerTokenEmbeddings = null, perLayerModelProjection = null;
        F32FloatTensor rmsPerLayerProjectionWeights = null;
        FloatTensor[] perLayerInputGate = null, perLayerProjection = null;
        F32FloatTensor[] rmsPerLayerPostWeights = null;
        if (config.embeddingLengthPerLayer() > 0
                && tensors.containsKey("per_layer_token_embd.weight")) {
            perLayerTokenEmbeddings =
                    ModelLoader.loadQuantized(tensors.get("per_layer_token_embd.weight"));
            perLayerModelProjection =
                    ModelLoader.loadQuantized(tensors.get("per_layer_model_proj.weight"));
            rmsPerLayerProjectionWeights =
                    ModelLoader.toF32Tensor(tensors.get("per_layer_proj_norm.weight"));
            perLayerInputGate =
                    ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".inp_gate.weight"));
            perLayerProjection =
                    ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".proj.weight"));
            rmsPerLayerPostWeights =
                    ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".post_norm.weight"));
        }

        // MoE (A4B)
        FloatTensor[] expertRouter = null, expertGateUp = null, expertDown = null;
        F32FloatTensor[] expertRouterScale = null,
                expertDownScale = null,
                rmsExpertPostNorm1 = null,
                rmsExpertPreNorm2 = null,
                rmsExpertPostNorm2 = null;
        if (config.isMoE()) {
            expertRouter =
                    ModelLoader.quantArray(
                            n, i -> tensors.get("blk." + i + ".ffn_gate_inp.weight"));
            expertRouterScale =
                    ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_gate_inp.scale"));
            expertGateUp =
                    ModelLoader.quantArray(
                            n, i -> tensors.get("blk." + i + ".ffn_gate_up_exps.weight"));
            expertDown =
                    ModelLoader.quantArray(
                            n, i -> tensors.get("blk." + i + ".ffn_down_exps.weight"));
            expertDownScale =
                    ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_down_exps.scale"));
            rmsExpertPostNorm1 =
                    ModelLoader.f32Array(
                            n, i -> tensors.get("blk." + i + ".post_ffw_norm_1.weight"));
            rmsExpertPreNorm2 =
                    ModelLoader.f32Array(
                            n, i -> tensors.get("blk." + i + ".pre_ffw_norm_2.weight"));
            rmsExpertPostNorm2 =
                    ModelLoader.f32Array(
                            n, i -> tensors.get("blk." + i + ".post_ffw_norm_2.weight"));
        }

        return new Weights(
                tokenEmbeddings,
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_q.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_k.weight")),
                valueWeights,
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_output.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_q_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_k_norm.weight")),
                ModelLoader.f32Array(
                        n, i -> tensors.get("blk." + i + ".post_attention_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".post_ffw_norm.weight")),
                ModelLoader.toF32Tensor(tensors.get("output_norm.weight")),
                layerOutputScales,
                F32FloatTensor.of(ropeFull.first()),
                F32FloatTensor.of(ropeFull.second()),
                F32FloatTensor.of(ropeSwa.first()),
                F32FloatTensor.of(ropeSwa.second()),
                tensors.containsKey("output.weight")
                        ? ModelLoader.loadQuantized(tensors.get("output.weight"))
                        : tokenEmbeddings,
                perLayerTokenEmbeddings,
                perLayerModelProjection,
                rmsPerLayerProjectionWeights,
                perLayerInputGate,
                perLayerProjection,
                rmsPerLayerPostWeights,
                expertRouter,
                expertRouterScale,
                expertGateUp,
                expertDown,
                expertDownScale,
                rmsExpertPostNorm1,
                rmsExpertPreNorm2,
                rmsExpertPostNorm2);
    }
}
