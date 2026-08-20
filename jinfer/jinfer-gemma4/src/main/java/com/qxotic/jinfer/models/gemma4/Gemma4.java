package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Moe;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jinfer.kernels.Trace;
import com.qxotic.jinfer.llm.Generator.Constraints;
import com.qxotic.jinfer.llm.Generator.GenerationListener;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpeculativeDecoding;
import com.qxotic.jinfer.llm.SpeculativeDecoding.SpeculationAudit;
import com.qxotic.jinfer.llm.SpeculativeDecoding.SpeculationResult;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jota.DataType;
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

/** Gemma4 inference against the MemoryView boundary. */
public final class Gemma4
        implements LanguageModel<Gemma4.Configuration, Gemma4.Weights, Gemma4.State>,
                Multimodal,
                SpeculativeDecoding<Gemma4.State> {
    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;
    private final MediaProjector<Media.Image> vision;
    private final MediaProjector<Media.Audio> audio;
    private Gemma4Mtp mtp; // MTP draft sidecar; null unless attachMtp loaded one

    Gemma4(Configuration configuration, Tokenizer tokenizer, Weights weights) {
        this(configuration, tokenizer, weights, null, null);
    }

    private Gemma4(
            Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            MediaProjector<Media.Image> vision,
            MediaProjector<Media.Audio> audio) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
        this.vision = vision;
        this.audio = audio;
    }

    @Override
    public Configuration configuration() {
        return configuration;
    }

    @Override
    public Optional<CheckpointCodec<State>> checkpointCodec() {
        return Optional.of(new Gemma4CheckpointCodec(configuration));
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
        if (modality == Media.Audio.class && audio != null)
            return Optional.of((MediaProjector<R>) audio);
        return Optional.empty();
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

    private void forward(State state, Batch batch) {
        int n = batch.count();
        if (n <= 0) throw new IllegalArgumentException("batch must not be empty");
        if (n > state.batchCapacity())
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + state.batchCapacity());
        int from = state.position();
        if (from + n > state.contextCapacity())
            throw new IllegalArgumentException(
                    "ingest of "
                            + n
                            + " at position "
                            + from
                            + " exceeds contextCapacity "
                            + state.contextCapacity());
        switch (batch.input()) {
            case Batch.Input.Tokens t -> {
                int[] ids = t.ids();
                state.lastTokens = ids; // row->token map for the MTP draft seam
                for (int id : ids)
                    if (id < 0 || id >= configuration.vocabularySize)
                        throw new IllegalArgumentException("token id out of range: " + id);
                if (n == 1)
                    Parallel.onDecodePool(
                            () -> {
                                forward(state, ids, 0, from, n);
                                return null;
                            });
                else forward(state, ids, 0, from, n);
            }
            case Batch.Input.Embeddings e -> {
                if (vision == null && audio == null)
                    throw new UnsupportedOperationException("no media encoder loaded");
                if (e.rows().shape().flatAt(1) != configuration.embeddingLength)
                    throw new IllegalArgumentException(
                            "embedding width "
                                    + e.rows().shape().flatAt(1)
                                    + " != model width "
                                    + configuration.embeddingLength);
                int pad = tokenizer.vocabulary().findId("<pad>").orElse(0);
                forwardEmbeddings(
                        state,
                        Views.castToSegmentBacked(e.rows(), "embedding rows"),
                        pad,
                        from,
                        n,
                        e.bidirectional());
            }
            case Batch.Input.Sequences ignored ->
                    throw new UnsupportedOperationException(
                            "Gemma4 is generative: packed sequences are not supported");
        }
        state.advance(batch);
    }

    @Override
    public MemoryView<?> logits(State state, int output) {
        return state.exclusively(() -> projectLogits(state, output));
    }

    private MemoryView<?> projectLogits(State state, int output) {
        if (output < 0 || output >= state.outputCount())
            throw new IllegalArgumentException(
                    "output " + output + " outside [0," + state.outputCount() + ")");
        int dim = configuration.embeddingLength;
        int row = state.lastBatchSize() - state.outputCount() + output;
        return Parallel.onDecodePool(
                () -> {
                    Norms.rmsnorm(
                            state.normed,
                            0,
                            state.residual,
                            (long) row * dim,
                            weights.finalNorm,
                            dim,
                            configuration.rmsNormEps);
                    MatMul.gemv(weights.classifier, state.normed, state.logits);
                    Activations.softcap(
                            state.logits,
                            0,
                            configuration.vocabularySize,
                            configuration.logitSoftcapping);
                    return state.logits;
                });
    }

    /**
     * Every retained row's logits in one head GEMM: {@code dst} holds {@code outputCount x vocab},
     * row-major, softcapped in place. The speculative verify walk is the consumer (per-row {@code
     * head} calls would re-stream the whole head weight per draft row).
     */
    void logitsAll(State state, MemoryView<MemorySegment> dst) {
        int dim = configuration.embeddingLength;
        int vocab = configuration.vocabularySize;
        int n = state.outputCount();
        int first = state.lastBatchSize() - n;
        Parallel.onDecodePool(
                () -> {
                    for (int r = 0; r < n; r++) {
                        Norms.rmsnorm(
                                state.normed,
                                (long) r * dim,
                                state.residual,
                                (long) (first + r) * dim,
                                weights.finalNorm,
                                dim,
                                configuration.rmsNormEps);
                    }
                    MatMul.gemm(weights.classifier, state.normed, dst, n);
                    Activations.softcap(dst, 0, n * vocab, configuration.logitSoftcapping);
                    return null;
                });
        Reference.reachabilityFence(state);
    }

    private void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        fillRotaryTables(state, startPos, seqLen);
        embed(state, tokens, tokenOffset, seqLen);
        buildPerLayerInputs(state, tokens, tokenOffset, seqLen);
        forwardLayers(state, startPos, seqLen, false);
    }

    private void forwardEmbeddings(
            State state,
            MemoryView<MemorySegment> rows,
            int pleToken,
            int startPos,
            int seqLen,
            boolean bidirectional) {
        fillRotaryTables(state, startPos, seqLen);
        Convert.copyF32(rows, 0, state.residual, 0, (long) seqLen * configuration.embeddingLength);
        int[] ple = new int[seqLen];
        Arrays.fill(ple, pleToken);
        buildPerLayerInputs(state, ple, 0, seqLen);
        forwardLayers(state, startPos, seqLen, bidirectional);
    }

    private void fillRotaryTables(State state, int startPos, int seqLen) {
        Configuration c = configuration;
        RoPE.fill(
                state.ropeCosFull,
                state.ropeSinFull,
                startPos,
                seqLen,
                c.headSizeFull / 2,
                weights.ropeFull);
        RoPE.fill(
                state.ropeCosSwa,
                state.ropeSinSwa,
                startPos,
                seqLen,
                c.headSizeSwa / 2,
                weights.ropeSwa);
    }

    private void forwardLayers(State state, int startPos, int seqLen, boolean bidirectional) {
        for (int layer = 0; layer < configuration.numberOfLayers; layer++)
            layer(state, layer, startPos, seqLen, bidirectional);
        commitKv(state, startPos, seqLen);
    }

    private void embed(State state, int[] tokens, int tokenOffset, int seqLen) {
        int dim = configuration.embeddingLength;
        Views.checkAlive(weights.tokenEmbeddings, "tokenEmbeddings");
        Convert.gatherToF32(
                weights.tokenEmbeddings, tokens, tokenOffset, seqLen, state.residual, 0, dim);
        float scale = (float) Math.sqrt(dim);
        Ops.mapInPlace(state.residual, 0, seqLen * dim, v -> v * scale);
    }

    private void buildPerLayerInputs(State state, int[] tokens, int tokenOffset, int seqLen) {
        Configuration c = configuration;
        int plDim = c.embeddingLengthPerLayer;
        if (plDim == 0 || weights.perLayerTokenEmbeddings == null) return;
        int dim = c.embeddingLength, total = plDim * c.numberOfLayers;
        MatMul.gemm(weights.perLayerModelProjection, state.residual, state.perLayerInputs, seqLen);
        float projectionScale = (float) (1.0 / Math.sqrt(dim));
        float tokenScale = (float) Math.sqrt(plDim);
        float inputScale = (float) (1.0 / Math.sqrt(2.0));
        Ops.mapInPlace(state.perLayerInputs, 0, seqLen * total, value -> value * projectionScale);
        for (int s = 0; s < seqLen; s++) {
            long base = (long) s * total;
            for (int l = 0; l < c.numberOfLayers; l++)
                Norms.rmsnorm(
                        state.perLayerInputs,
                        base + (long) l * plDim,
                        state.perLayerInputs,
                        base + (long) l * plDim,
                        weights.perLayerProjectionNorm,
                        plDim,
                        c.rmsNormEps);
            Convert.copyToF32(
                    weights.perLayerTokenEmbeddings,
                    (long) tokens[tokenOffset + s] * total,
                    state.perLayerTokenRow,
                    0,
                    total);
            Ops.mapInPlace(state.perLayerTokenRow, 0, total, value -> value * tokenScale);
            Ops.addInPlace(state.perLayerInputs, base, state.perLayerTokenRow, 0, total);
            Ops.mapInPlace(state.perLayerInputs, base, total, value -> value * inputScale);
        }
    }

    private void layer(State state, int l, int startPos, int seqLen, boolean bidirectional) {
        attention(state, l, startPos, seqLen, bidirectional);
        feedForward(state, l, seqLen);
        mergePerLayerInput(state, l, seqLen);
        float scale = weights.layers[l].outputScale;
        if (scale != 1f)
            Ops.mapInPlace(
                    state.residual,
                    0,
                    seqLen * configuration.embeddingLength,
                    value -> value * scale);
        if (Trace.ENABLED)
            Trace.sum("l_out-" + l, state.residual, seqLen * configuration.embeddingLength);
    }

    private void attention(State state, int l, int startPos, int seqLen, boolean bidirectional) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[l];
        int dim = c.embeddingLength;
        boolean swa = c.isSwa[l];
        int headSize = c.headSize(l), queryDim = c.queryDim(l), kvDim = c.kvDim(l);
        int nKvHeads = c.numberOfKeyValueHeadsPerLayer[l];
        int kvMul = c.numberOfHeads / nKvHeads;
        int kvLayer = c.kvSourceLayer(l);
        MemoryView<MemorySegment> cos = swa ? state.ropeCosSwa : state.ropeCosFull;
        MemoryView<MemorySegment> sin = swa ? state.ropeSinSwa : state.ropeSinFull;

        Norms.rmsnormRows(state.normed, state.residual, w.attnNorm, seqLen, dim, c.rmsNormEps);
        MatMul.gemm(w.wq, state.normed, state.query, seqLen);
        headNormRope(state.query, queryDim, c.numberOfHeads, headSize, w.qNorm, seqLen, cos, sin);
        if (c.hasKv(l)) {
            MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];
            MatMul.gemm(w.wk, state.normed, bK, seqLen);
            if (w.wv != null) MatMul.gemm(w.wv, state.normed, bV, seqLen);
            else Convert.copyF32(bK, 0, bV, 0, (long) seqLen * kvDim);
            headNormRope(bK, kvDim, nKvHeads, headSize, w.kNorm, seqLen, cos, sin);
            Parallel.forRows(
                    seqLen,
                    s -> {
                        for (int h = 0; h < nKvHeads; h++)
                            Norms.rmsnormNoWeight(
                                    bV,
                                    (long) s * kvDim + (long) h * headSize,
                                    bV,
                                    (long) s * kvDim + (long) h * headSize,
                                    headSize,
                                    c.rmsNormEps);
                    });
        }
        MemoryView<MemorySegment> bK = state.batchK[kvLayer], bV = state.batchV[kvLayer];
        if (seqLen > 1)
            FlashAttention.slidingWindowPrefill(
                    state.query,
                    state.attnOut,
                    state.keyCache[kvLayer],
                    state.valueCache[kvLayer],
                    bK,
                    bV,
                    c.numberOfHeads,
                    startPos,
                    seqLen,
                    headSize,
                    kvDim,
                    queryDim,
                    kvDim,
                    kvMul,
                    1f,
                    swa ? c.slidingWindow : 0,
                    swa ? c.slidingWindow - 1 : 0,
                    null,
                    bidirectional);
        else
            FlashAttention.flashDecode(
                    state.query,
                    state.attnOut,
                    state.keyCache[kvLayer],
                    state.valueCache[kvLayer],
                    bK,
                    bV,
                    c.numberOfHeads,
                    startPos,
                    swa ? Math.max(0, startPos - c.slidingWindow + 1) : 0,
                    headSize,
                    kvDim,
                    kvMul,
                    1f,
                    swa ? c.slidingWindow - 1 : 0,
                    null,
                    state.decodeScratch);
        MatMul.gemm(w.wo, state.attnOut, state.branchOut, seqLen);
        Norms.rmsnormRows(
                state.branchOut, state.branchOut, w.postAttnNorm, seqLen, dim, c.rmsNormEps);
        Ops.addInPlace(state.residual, 0, state.branchOut, 0, seqLen * dim);
    }

    private void headNormRope(
            MemoryView<MemorySegment> t,
            int stride,
            int heads,
            int headSize,
            MemoryView<MemorySegment> norm,
            int seqLen,
            MemoryView<MemorySegment> cos,
            MemoryView<MemorySegment> sin) {
        Parallel.forRows(
                seqLen,
                s -> {
                    for (int h = 0; h < heads; h++) {
                        long off = (long) s * stride + (long) h * headSize;
                        Norms.rmsnorm(t, off, t, off, norm, headSize, configuration.rmsNormEps);
                        RoPE.applyNeox(t, off, s, cos, sin, headSize / 2);
                    }
                });
    }

    private void feedForward(State state, int l, int seqLen) {
        LayerWeights w = weights.layers[l];
        if (w.moe != null) {
            moeFeedForward(state, l, seqLen);
            return;
        }
        denseMlp(state, l, w, seqLen, state.normed, w.postFfnNorm);
        Ops.addInPlace(state.residual, 0, state.normed, 0, seqLen * configuration.embeddingLength);
    }

    private void denseMlp(
            State state,
            int layer,
            LayerWeights w,
            int seqLen,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> postNorm) {
        int dim = configuration.embeddingLength, hidden = configuration.feedForwardLength[layer];
        Norms.rmsnormRows(
                state.normed, state.residual, w.ffnNorm, seqLen, dim, configuration.rmsNormEps);
        MatMul.gemm(w.gate, state.normed, state.hidden, seqLen);
        MatMul.gemm(w.up, state.normed, state.hidden2, seqLen);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.geluMultiply(
                                state.hidden, s * hidden, state.hidden2, s * hidden, hidden));
        MatMul.gemm(w.down, state.hidden, output, seqLen);
        Norms.rmsnormRows(output, output, postNorm, seqLen, dim, configuration.rmsNormEps);
    }

    private void mergePerLayerInput(State state, int l, int seqLen) {
        Configuration c = configuration;
        int plDim = c.embeddingLengthPerLayer;
        if (plDim == 0 || weights.layers[l].inputGate == null) return;
        int dim = c.embeddingLength, total = plDim * c.numberOfLayers;
        LayerWeights w = weights.layers[l];
        MatMul.gemm(w.inputGate, state.residual, state.plGate, seqLen);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.geluMultiply(
                                state.plGate,
                                s * plDim,
                                state.perLayerInputs,
                                s * total + l * plDim,
                                plDim));
        MatMul.gemm(w.projection, state.plGate, state.plProjection, seqLen);
        Norms.rmsnormRows(
                state.plProjection,
                state.plProjection,
                w.postProjectionNorm,
                seqLen,
                dim,
                c.rmsNormEps);
        Ops.addInPlace(state.residual, 0, state.plProjection, 0, seqLen * dim);
    }

    private void moeFeedForward(State state, int l, int seqLen) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[l];
        MoeWeights moe = w.moe;
        int dim = c.embeddingLength;
        int experts = c.expertCount, topK = c.expertUsedCount;
        int expertFf = c.expertFeedForwardLength, gateUp = 2 * expertFf;

        denseMlp(state, l, w, seqLen, state.moeShared, moe.postNorm1);

        float invSqrtDim = 1f / (float) Math.sqrt(dim);
        Parallel.forRows(
                seqLen,
                s -> {
                    long off = (long) s * dim;
                    float rms =
                            (float)
                                    (1.0
                                            / Math.sqrt(
                                                    Norms.sumOfSquares(state.residual, off, dim)
                                                                    / dim
                                                            + c.rmsNormEps));
                    Norms.scaleByWeight(
                            state.moeInput, off, state.residual, off, moe.preNorm2, dim, rms);
                    Norms.scaleByWeight(
                            state.moeRouterInput,
                            off,
                            state.residual,
                            off,
                            moe.routerScale,
                            dim,
                            rms * invSqrtDim);
                });
        MatMul.gemm(moe.router, state.moeRouterInput, state.moeRouter, seqLen);
        selectTopExperts(state, seqLen, experts, topK);
        Moe.Routing routing = state.moeRouting;
        routing.seqLen = seqLen;
        Moe.dispatch(
                routing,
                dim,
                state.moeInput,
                state.moeGather,
                state.moeExpertOut,
                state.moeOut,
                moe.downScale,
                (e, n, gather, out) -> {
                    MatMul.gemm(moe.gateUp[e], gather, state.moeHidden, n);
                    Parallel.forRows(
                            n,
                            row ->
                                    Activations.geluMultiply(
                                            state.moeHidden,
                                            row * gateUp,
                                            state.moeHidden,
                                            row * gateUp + expertFf,
                                            expertFf));
                    MatMul.gemm(moe.down[e], state.moeHidden, out, n);
                });
        Norms.rmsnormRows(state.moeOut, state.moeOut, moe.postNorm2, seqLen, dim, c.rmsNormEps);
        Ops.addInPlace(state.moeShared, 0, state.moeOut, 0, seqLen * dim);
        Norms.rmsnormRows(
                state.moeShared, state.moeShared, w.postFfnNorm, seqLen, dim, c.rmsNormEps);
        Ops.addInPlace(state.residual, 0, state.moeShared, 0, seqLen * dim);
    }

    private static void selectTopExperts(State state, int seqLen, int experts, int topK) {
        for (int row = 0; row < seqLen; row++) {
            Ops.softmaxInPlace(state.moeRouter, (long) row * experts, experts);
        }
        Moe.selectTopK(
                state.moeRouter,
                seqLen,
                experts,
                topK,
                state.moeRowTopE,
                state.moeRowTopP,
                state.moeExpertCounts);
    }

    private void commitKv(State state, int startPos, int seqLen) {
        for (int l = 0; l < configuration.ownKvLayers; l++) {
            int kvDim = configuration.kvDim(l);
            for (int s = 0; s < seqLen; s++) {
                long pos = configuration.kvCacheIndex(l, startPos + s);
                Convert.f32ToF16(
                        state.batchK[l], (long) s * kvDim, state.keyCache[l], pos * kvDim, kvDim);
                Convert.f32ToF16(
                        state.batchV[l], (long) s * kvDim, state.valueCache[l], pos * kvDim, kvDim);
            }
        }
    }

    public record Configuration(
            int embeddingLength,
            int[] feedForwardLength,
            int numberOfLayers,
            int numberOfHeads,
            int[] numberOfKeyValueHeadsPerLayer,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            float ropeThetaFull,
            float ropeThetaSwa,
            int headSizeFull,
            int headSizeSwa,
            int slidingWindow,
            float logitSoftcapping,
            boolean[] isSwa,
            int ownKvLayers,
            int embeddingLengthPerLayer,
            int expertCount,
            int expertUsedCount,
            int expertFeedForwardLength)
            implements ContextConfiguration {
        public Configuration {
            if (slidingWindow <= 0 || Integer.bitCount(slidingWindow) != 1)
                throw new IllegalArgumentException(
                        "slidingWindow must be a power of 2, got " + slidingWindow);
        }

        boolean isMoe() {
            return expertCount > 0;
        }

        boolean hasKv(int layer) {
            return layer < ownKvLayers;
        }

        int kvSourceLayer(int layer) {
            return hasKv(layer) ? layer : ownKvLayers - (isSwa[layer] ? 2 : 1);
        }

        int headSize(int layer) {
            return isSwa[layer] ? headSizeSwa : headSizeFull;
        }

        int queryDim(int layer) {
            return numberOfHeads * headSize(layer);
        }

        int kvDim(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] * headSize(layer);
        }

        int kvCachePositions(int layer, int contextCapacity) {
            return isSwa[layer] ? Math.min(contextCapacity, slidingWindow) : contextCapacity;
        }

        int kvCacheIndex(int layer, int position) {
            return isSwa[layer] ? position & (slidingWindow - 1) : position;
        }

        int maxQueryDim() {
            return numberOfHeads * Math.max(headSizeFull, headSizeSwa);
        }

        int maxHiddenDim() {
            int max = isMoe() ? 2 * expertFeedForwardLength : 0;
            for (int value : feedForwardLength) max = Math.max(max, value);
            return max;
        }
    }

    public record MoeWeights(
            MemoryView<MemorySegment> router,
            MemoryView<MemorySegment> routerScale,
            MemoryView<MemorySegment>[] gateUp,
            MemoryView<MemorySegment>[] down,
            MemoryView<MemorySegment> downScale,
            MemoryView<MemorySegment> postNorm1,
            MemoryView<MemorySegment> preNorm2,
            MemoryView<MemorySegment> postNorm2) {}

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> wq,
            MemoryView<MemorySegment> wk,
            MemoryView<MemorySegment> wv,
            MemoryView<MemorySegment> wo,
            MemoryView<MemorySegment> qNorm,
            MemoryView<MemorySegment> kNorm,
            MemoryView<MemorySegment> postAttnNorm,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> down,
            MemoryView<MemorySegment> up,
            MemoryView<MemorySegment> postFfnNorm,
            float outputScale,
            MemoryView<MemorySegment> inputGate,
            MemoryView<MemorySegment> projection,
            MemoryView<MemorySegment> postProjectionNorm,
            MoeWeights moe) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbeddings,
            LayerWeights[] layers,
            MemoryView<MemorySegment> finalNorm,
            RoPE.Schedule ropeFull,
            RoPE.Schedule ropeSwa,
            MemoryView<MemorySegment> classifier,
            MemoryView<MemorySegment> perLayerTokenEmbeddings,
            MemoryView<MemorySegment> perLayerModelProjection,
            MemoryView<MemorySegment> perLayerProjectionNorm) {}

    public static final class State extends ContextState {
        final MemoryView<MemorySegment> residual, normed, branchOut, attnOut, query;
        final MemoryView<MemorySegment> hidden, hidden2, logits;
        final MemoryView<MemorySegment> ropeCosFull, ropeSinFull, ropeCosSwa, ropeSinSwa;
        final FlashAttention.DecodeScratch decodeScratch =
                new FlashAttention.DecodeScratch(memoryArena());
        final MemoryView<MemorySegment>[] keyCache, valueCache, batchK, batchV;
        final MemoryView<MemorySegment> perLayerInputs, perLayerTokenRow, plGate, plProjection;
        final MemoryView<MemorySegment> moeShared,
                moeInput,
                moeRouterInput,
                moeRouter,
                moeOut,
                moeGather,
                moeExpertOut;
        // Per-expert packed gate|up at EXACTLY 2*expertFeedForwardLength wide: hidden is sized to
        // the model's max FFN width (dense layers can be wider), and the gelu-multiply between
        // gate|up and down addresses rows packed at the gateUp width.
        final MemoryView<MemorySegment> moeHidden;
        final int[] moeExpertCounts, moeRowTopE;
        final float[] moeRowTopP;
        final Moe.Routing moeRouting;

        // speculation scratch, lazily allocated from this state's arena and reused per
        // generation; freed with the state (see Gemma4Speculative.Scratch)
        Gemma4Speculative.Scratch specScratch;
        int[] lastTokens; // ids of the last ingested token batch (row->token, MTP draft seam)

        /** The speculation scratch's allocator (the state's own arena); State-internal seam. */
        MemoryAllocator<MemorySegment> specArena() {
            return memoryArena();
        }

        @SuppressWarnings("unchecked")
        State(
                Configuration c,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity > c.contextLength)
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds model contextLength "
                                + c.contextLength);
            int rows = batchCapacity, dim = c.embeddingLength;
            residual = Views.allocateF32(memoryArena(), rows, dim);
            normed = Views.allocateF32(memoryArena(), rows, dim);
            branchOut = Views.allocateF32(memoryArena(), rows, dim);
            attnOut = Views.allocateF32(memoryArena(), rows, c.maxQueryDim());
            query = Views.allocateF32(memoryArena(), rows, c.maxQueryDim());
            hidden = Views.allocateF32(memoryArena(), rows, c.maxHiddenDim());
            hidden2 = Views.allocateF32(memoryArena(), rows, c.maxHiddenDim());
            logits = Views.allocateF32(memoryArena(), 1, c.vocabularySize);
            ropeCosFull = Views.allocateF32(memoryArena(), rows, c.headSizeFull / 2);
            ropeSinFull = Views.allocateF32(memoryArena(), rows, c.headSizeFull / 2);
            ropeCosSwa = Views.allocateF32(memoryArena(), rows, c.headSizeSwa / 2);
            ropeSinSwa = Views.allocateF32(memoryArena(), rows, c.headSizeSwa / 2);
            keyCache = new MemoryView[c.ownKvLayers];
            valueCache = new MemoryView[c.ownKvLayers];
            batchK = new MemoryView[c.ownKvLayers];
            batchV = new MemoryView[c.ownKvLayers];
            for (int l = 0; l < c.ownKvLayers; l++) {
                int kvDim = c.kvDim(l);
                keyCache[l] =
                        Views.allocateF16(
                                memoryArena(), c.kvCachePositions(l, contextCapacity), kvDim);
                valueCache[l] =
                        Views.allocateF16(
                                memoryArena(), c.kvCachePositions(l, contextCapacity), kvDim);
                batchK[l] = Views.allocateF32(memoryArena(), rows, kvDim);
                batchV[l] = Views.allocateF32(memoryArena(), rows, kvDim);
            }
            int plDim = c.embeddingLengthPerLayer, plTotal = plDim * c.numberOfLayers;
            perLayerInputs = plDim == 0 ? null : Views.allocateF32(memoryArena(), rows, plTotal);
            perLayerTokenRow = plDim == 0 ? null : Views.allocateF32(memoryArena(), plTotal);
            plGate = plDim == 0 ? null : Views.allocateF32(memoryArena(), rows, plDim);
            plProjection = plDim == 0 ? null : Views.allocateF32(memoryArena(), rows, dim);
            if (c.isMoe()) {
                moeShared = Views.allocateF32(memoryArena(), rows, dim);
                moeInput = Views.allocateF32(memoryArena(), rows, dim);
                moeRouterInput = Views.allocateF32(memoryArena(), rows, dim);
                moeRouter = Views.allocateF32(memoryArena(), rows, c.expertCount);
                moeOut = Views.allocateF32(memoryArena(), rows, dim);
                moeGather = Views.allocateF32(memoryArena(), rows, dim);
                moeExpertOut = Views.allocateF32(memoryArena(), rows, dim);
                moeHidden = Views.allocateF32(memoryArena(), rows, 2 * c.expertFeedForwardLength);
                moeExpertCounts = new int[c.expertCount];
                moeRowTopE = new int[rows * c.expertUsedCount];
                moeRowTopP = new float[rows * c.expertUsedCount];
                moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts);
                moeRouting.topK = c.expertUsedCount;
                moeRouting.numExperts = c.expertCount;
            } else {
                moeShared = moeInput = moeRouterInput = moeRouter = null;
                moeOut = moeGather = moeExpertOut = null;
                moeHidden = null;
                moeExpertCounts = moeRowTopE = null;
                moeRowTopP = null;
                moeRouting = null;
            }
        }

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }

        @Override
        protected void clearHistory() {}
    }

    public static Gemma4 loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(channel, gguf, arena);
        }
    }

    /** Loads the text backbone and its paired Gemma 4 media sidecar into {@code arena}. */
    public static Gemma4 loadModel(Path textPath, Path mmprojPath, Arena arena) throws IOException {
        return loadModel(textPath, arena).withMedia(mmprojPath, arena);
    }

    /**
     * MTP load: the text model plus the {@code gemma4-assistant} draft sidecar, which enables
     * self-speculative decoding: the sidecar is attached and {@link #mtpDecoder} can mint a
     * per-generation draft decoder over it.
     */
    public static Gemma4 loadWithMtp(Path textGguf, Path mtpSidecar, Arena arena)
            throws IOException {
        return loadModel(textGguf, arena).attachMtp(mtpSidecar, arena);
    }

    /**
     * Attaches the {@code speculation} companion: the MTP draft sidecar, into {@code arena}. A
     * load-time wiring call, before the model is published - pass the WEIGHTS arena so the sidecar
     * dies with the backbone.
     *
     * <p>The PAIRING is enforced here, like the mmproj's: the sidecar consumes this backbone's
     * hidden state, so its {@code embedding_length_out} must equal this model's embedding width,
     * and its draft Q heads attend this backbone's KV rings (SWA drafts at layer {@code ownKv-2},
     * the full draft at {@code ownKv-1}), so the draft head sizes must equal the KV head sizes at
     * those layers - an E2B head on an E4B backbone refuses with both numbers rather than failing
     * in a GEMM or attending garbage later. (A same-geometry wrong head cannot be detected from
     * headers; it is still SAFE - verification keeps only tokens the backbone confirms - just slow,
     * visible as near-zero acceptance.)
     */
    public Gemma4 attachMtp(Path mtpSidecar, Arena arena) throws IOException {
        Gemma4Mtp sidecar =
                Gemma4Mtp.loadSidecar(mtpSidecar, configuration().vocabularySize(), arena);
        if (sidecar.configuration().backboneDim() != configuration().embeddingLength()) {
            throw new IllegalArgumentException(
                    mtpSidecar.getFileName()
                            + " drafts for a backbone with hidden width "
                            + sidecar.configuration().backboneDim()
                            + ", but this model's is "
                            + configuration().embeddingLength()
                            + " - it is the MTP head of a different gemma-4 size; use the sidecar"
                            + " published for this exact model");
        }
        int ownKv = configuration().ownKvLayers();
        int swaHead = ownKv >= 2 ? kvHeadSize(ownKv - 2) : -1;
        int fullHead = ownKv >= 1 ? kvHeadSize(ownKv - 1) : -1;
        if (ownKv < 2
                || sidecar.configuration().headSizeSWA() != swaHead
                || sidecar.configuration().headSizeFull() != fullHead) {
            throw new IllegalArgumentException(
                    mtpSidecar.getFileName()
                            + " drafts with head sizes "
                            + sidecar.configuration().headSizeSWA()
                            + "/"
                            + sidecar.configuration().headSizeFull()
                            + " (swa/full), but this backbone's KV heads at the shared layers are "
                            + swaHead
                            + "/"
                            + fullHead
                            + " - it is the MTP head of a different gemma-4 size; use the sidecar"
                            + " published for this exact model");
        }
        this.mtp = sidecar;
        return this;
    }

    private int kvHeadSize(int layer) {
        return configuration().kvDim(layer)
                / configuration().numberOfKeyValueHeadsPerLayer()[layer];
    }

    @Override
    public boolean speculationReady() {
        return mtp != null;
    }

    @Override
    public SpeculationResult speculate(
            State state,
            Sampler sampler,
            Constraints constraints,
            int depth,
            GenerationListener listener,
            SpeculationAudit audit) {
        int capacity = state.contextCapacity();
        int budget =
                constraints.maxTokens() == Constraints.UNLIMITED
                        ? capacity - state.position()
                        : Math.min(constraints.maxTokens(), capacity - state.position());
        long timeoutNanos = constraints.timeout().isZero() ? 0 : constraints.timeout().toNanos();
        return state.exclusively(
                () ->
                        Gemma4Speculative.generate(
                                this,
                                state,
                                budget,
                                timeoutNanos,
                                constraints.stopTokens(),
                                depth,
                                sampler,
                                listener,
                                audit));
    }

    /**
     * A fresh MTP draft forward over {@code allocator}, or null when no sidecar is loaded. The
     * decoder is pure per-generation scratch over the immutable {@link Gemma4Mtp} weights - mint
     * one per speculative generation in the state's arena (a model-level singleton was shared
     * mutable state: concurrent speculative decodes on two states would corrupt each other's draft
     * buffers). {@link Gemma4Speculative} is the decode loop over it.
     */
    Gemma4MtpDecoder mtpDecoder(MemoryAllocator<MemorySegment> allocator) {
        return mtp == null ? null : new Gemma4MtpDecoder(mtp, this, allocator);
    }

    /** Returns a model sharing this backbone's weights with a validated media sidecar attached. */
    public Gemma4 withMedia(Path mmprojPath, Arena arena) throws IOException {
        Objects.requireNonNull(mmprojPath, "mmprojPath");
        Objects.requireNonNull(arena, "arena");
        try (FileChannel channel = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, mmprojPath.toString());
            validateMediaPairing(mmprojPath, gguf, configuration.embeddingLength);
            String visionType = gguf.getStringOrDefault("clip.vision.projector_type", "");
            String audioType = gguf.getStringOrDefault("clip.audio.projector_type", "");
            Map<String, MemoryView<MemorySegment>> tensors =
                    ModelLoader.loadTensors(channel, gguf, arena);
            MediaProjector<Media.Image> visionEncoder =
                    switch (visionType) {
                        case "" -> null;
                        case "gemma4v" -> Gemma4Vision.loadModel(mmprojPath, gguf, tensors);
                        case "gemma4uv" -> Gemma4VisionUnified.loadModel(mmprojPath, gguf, tensors);
                        default ->
                                throw new IllegalArgumentException(
                                        "'"
                                                + mmprojPath.getFileName()
                                                + "' carries unsupported vision projector '"
                                                + visionType
                                                + "' (expected gemma4v or gemma4uv)");
                    };
            MediaProjector<Media.Audio> audioEncoder =
                    switch (audioType) {
                        case "" -> null;
                        case "gemma4a" ->
                                Gemma4Conformer.loadModel(mmprojPath, gguf, tensors, arena);
                        case "gemma4ua" -> Gemma4Audio.loadModel(mmprojPath, gguf, tensors);
                        default ->
                                throw new IllegalArgumentException(
                                        "'"
                                                + mmprojPath.getFileName()
                                                + "' carries unsupported audio projector '"
                                                + audioType
                                                + "' (expected gemma4a or gemma4ua)");
                    };
            return new Gemma4(configuration, tokenizer, weights, visionEncoder, audioEncoder);
        }
    }

    /** Header-only pairing check: fail before mapping a sidecar's tensors. */
    static void validateMediaPairing(Path path, GGUF gguf, int modelWidth) {
        String visionType = gguf.getStringOrDefault("clip.vision.projector_type", "");
        String audioType = gguf.getStringOrDefault("clip.audio.projector_type", "");
        if (visionType.isEmpty() && audioType.isEmpty())
            throw new IllegalArgumentException(
                    "'" + path.getFileName() + "' carries no media projector");
        int visionDim = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", 0);
        int audioDim = gguf.getValueOrDefault(int.class, "clip.audio.projection_dim", 0);
        if ((visionDim != 0 && visionDim != modelWidth)
                || (audioDim != 0 && audioDim != modelWidth))
            throw new IllegalArgumentException(
                    "'"
                            + path.getFileName()
                            + "' projector dimensions [vision="
                            + visionDim
                            + ", audio="
                            + audioDim
                            + "] do not match model width "
                            + modelWidth
                            + "; use the projector for the same Gemma 4 size");
    }

    public static Gemma4 loadModel(FileChannel channel, GGUF gguf, Arena arena) throws IOException {
        return loadModel(channel, gguf, arena, null);
    }

    public static Gemma4 loadModel(FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration c = loadConfiguration(gguf, tokenizer);
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new Gemma4(c, tokenizer, loadWeights(tensors, c));
    }

    private static Configuration loadConfiguration(GGUF gguf, Tokenizer tokenizer) {
        String arch = "gemma4";
        int layers = gguf.getValue(int.class, arch + ".block_count");
        int dim = gguf.getValue(int.class, arch + ".embedding_length");
        int heads = gguf.getValue(int.class, arch + ".attention.head_count");
        int fullHead = gguf.getValue(int.class, arch + ".attention.key_length");
        int swaHead = gguf.getValue(int.class, arch + ".attention.key_length_swa");
        int window = gguf.getValue(int.class, arch + ".attention.sliding_window");
        int[] ffn =
                scalarOrArray(gguf.getValue(Object.class, arch + ".feed_forward_length"), layers);
        boolean[] isSwa = new boolean[layers];
        Object pattern =
                gguf.getValueOrDefault(
                        Object.class, arch + ".attention.sliding_window_pattern", null);
        if (pattern instanceof boolean[] value && value.length == layers) isSwa = value;
        else
            for (int l = 0; l < layers; l++) {
                TensorEntry qNorm = gguf.getTensor("blk." + l + ".attn_q_norm.weight");
                isSwa[l] = qNorm != null ? elements(qNorm.shape()) == swaHead : l % 5 != 4;
            }
        Object kvRaw =
                gguf.getValueOrDefault(Object.class, arch + ".attention.head_count_kv", heads);
        int[] kvHeads = new int[layers];
        for (int l = 0; l < layers; l++) {
            TensorEntry wk = gguf.getTensor("blk." + l + ".attn_k.weight");
            int head = isSwa[l] ? swaHead : fullHead;
            kvHeads[l] =
                    wk != null
                            ? Math.toIntExact(wk.shape()[1]) / head
                            : kvRaw instanceof int[] values
                                    ? values[l]
                                    : ((Number) kvRaw).intValue();
        }
        int ownKv =
                layers - gguf.getValueOrDefault(int.class, arch + ".attention.shared_kv_layers", 0);
        Configuration c =
                new Configuration(
                        dim,
                        ffn,
                        layers,
                        heads,
                        kvHeads,
                        tokenizer.vocabulary().size(),
                        gguf.getValue(int.class, arch + ".context_length"),
                        gguf.getValueOrDefault(
                                float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-6f),
                        gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 1_000_000f),
                        gguf.getValueOrDefault(float.class, arch + ".rope.freq_base_swa", 10_000f),
                        fullHead,
                        swaHead,
                        window,
                        gguf.getValueOrDefault(float.class, arch + ".final_logit_softcapping", 0f),
                        isSwa,
                        ownKv,
                        gguf.getValueOrDefault(
                                int.class, arch + ".embedding_length_per_layer_input", 0),
                        gguf.getValueOrDefault(int.class, arch + ".expert_count", 0),
                        gguf.getValueOrDefault(int.class, arch + ".expert_used_count", 0),
                        gguf.getValueOrDefault(int.class, arch + ".expert_feed_forward_length", 0));
        for (int l = ownKv; l < layers; l++) {
            int source = c.kvSourceLayer(l);
            if (source < 0 || source >= ownKv || c.kvDim(l) != c.kvDim(source))
                throw new IllegalStateException(
                        "layer " + l + " cannot share KV with layer " + source);
        }
        return c;
    }

    private static int[] scalarOrArray(Object value, int n) {
        if (value instanceof int[] array) return array;
        int[] array = new int[n];
        Arrays.fill(array, ((Number) value).intValue());
        return array;
    }

    private static long elements(long[] shape) {
        long n = 1;
        for (long value : shape) n *= value;
        return n;
    }

    private static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        return Objects.requireNonNull(tensors.get(name), name);
    }

    private static MemoryView<MemorySegment> requireF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> value = require(tensors, name);
        Views.requireDatatype(value, DataType.FP32, name);
        return value;
    }

    private static MemoryView<MemorySegment> findF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> value = tensors.get(name);
        if (value != null) Views.requireDatatype(value, DataType.FP32, name);
        return value;
    }

    static Weights loadWeights(Map<String, MemoryView<MemorySegment>> tensors, Configuration c) {
        int n = c.numberOfLayers;
        float[] freqs = ModelLoader.ropeFreqFactors(tensors);
        RoPE.Schedule full =
                freqs == null
                        ? RoPE.plain(c.headSizeFull, c.ropeThetaFull)
                        : RoPE.withFreqFactors(c.headSizeFull, c.ropeThetaFull, freqs);
        RoPE.Schedule swa = RoPE.plain(c.headSizeSwa, c.ropeThetaSwa);
        boolean ple =
                c.embeddingLengthPerLayer > 0 && tensors.containsKey("per_layer_token_embd.weight");
        LayerWeights[] layers = new LayerWeights[n];
        for (int l = 0; l < n; l++) {
            String p = "blk." + l + ".";
            MemoryView<MemorySegment> scale = findF32(tensors, p + "layer_output_scale.weight");
            float outputScale = 1f;
            if (scale != null) {
                outputScale = Views.getFloat(scale, 0, p + "layer_output_scale.weight");
            }
            MoeWeights moe =
                    c.isMoe() && tensors.containsKey(p + "ffn_gate_inp.weight")
                            ? new MoeWeights(
                                    require(tensors, p + "ffn_gate_inp.weight"),
                                    requireF32(tensors, p + "ffn_gate_inp.scale"),
                                    Views.sliceLeadingAxis(
                                            require(tensors, p + "ffn_gate_up_exps.weight")),
                                    Views.sliceLeadingAxis(
                                            require(tensors, p + "ffn_down_exps.weight")),
                                    requireF32(tensors, p + "ffn_down_exps.scale"),
                                    requireF32(tensors, p + "post_ffw_norm_1.weight"),
                                    requireF32(tensors, p + "pre_ffw_norm_2.weight"),
                                    requireF32(tensors, p + "post_ffw_norm_2.weight"))
                            : null;
            layers[l] =
                    new LayerWeights(
                            requireF32(tensors, p + "attn_norm.weight"),
                            require(tensors, p + "attn_q.weight"),
                            tensors.get(p + "attn_k.weight"),
                            tensors.get(p + "attn_v.weight"),
                            require(tensors, p + "attn_output.weight"),
                            requireF32(tensors, p + "attn_q_norm.weight"),
                            c.hasKv(l)
                                    ? requireF32(tensors, p + "attn_k_norm.weight")
                                    : findF32(tensors, p + "attn_k_norm.weight"),
                            requireF32(tensors, p + "post_attention_norm.weight"),
                            requireF32(tensors, p + "ffn_norm.weight"),
                            require(tensors, p + "ffn_gate.weight"),
                            require(tensors, p + "ffn_down.weight"),
                            require(tensors, p + "ffn_up.weight"),
                            requireF32(tensors, p + "post_ffw_norm.weight"),
                            outputScale,
                            ple ? require(tensors, p + "inp_gate.weight") : null,
                            ple ? require(tensors, p + "proj.weight") : null,
                            ple ? requireF32(tensors, p + "post_norm.weight") : null,
                            moe);
        }
        MemoryView<MemorySegment> tokenEmbeddings = require(tensors, "token_embd.weight");
        return new Weights(
                tokenEmbeddings,
                layers,
                requireF32(tensors, "output_norm.weight"),
                full,
                swa,
                tensors.containsKey("output.weight")
                        ? require(tensors, "output.weight")
                        : tokenEmbeddings,
                ple ? require(tensors, "per_layer_token_embd.weight") : null,
                ple ? require(tensors, "per_layer_model_proj.weight") : null,
                ple ? requireF32(tensors, "per_layer_proj_norm.weight") : null);
    }
}
