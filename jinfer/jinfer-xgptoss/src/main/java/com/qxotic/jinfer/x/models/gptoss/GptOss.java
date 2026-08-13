package com.qxotic.jinfer.x.models.gptoss;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.Parallel;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.BaseState;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Config;
import com.qxotic.jinfer.x.boundary.LanguageModel;
import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.Activations;
import com.qxotic.jinfer.x.kernels.Convert;
import com.qxotic.jinfer.x.kernels.FlashAttention;
import com.qxotic.jinfer.x.kernels.MatMul;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Moe;
import com.qxotic.jinfer.x.kernels.Norms;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jinfer.x.kernels.RoPE;
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
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** GPT-OSS inference against the MemoryView boundary. */
public final class GptOss
        implements LanguageModel<GptOss.Configuration, GptOss.Weights, GptOss.State> {
    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    GptOss(Configuration configuration, Tokenizer tokenizer, Weights weights) {
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
    public Optional<StateCodec<State>> stateCodec() {
        return Optional.of(new GptOssStateCodec(configuration));
    }

    @Override
    public State newState(int contextCapacity, int batchCapacity, Arena arena) {
        return new State(configuration, contextCapacity, batchCapacity, arena);
    }

    @Override
    public void forward(State state, Batch batch) {
        int n = batch.count();
        if (n > state.batchCapacity)
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + state.batchCapacity);
        int from = state.position();
        if (from + n > state.contextCapacity)
            throw new IllegalArgumentException(
                    "ingest of "
                            + n
                            + " at position "
                            + from
                            + " exceeds contextCapacity "
                            + state.contextCapacity);
        switch (batch.input()) {
            case Batch.Input.Tokens t -> {
                int[] ids = t.ids();
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
            case Batch.Input.Sequences ignored ->
                    throw new UnsupportedOperationException(
                            "GPT-OSS is generative: packed sequences are not supported");
            case Batch.Input.Embeddings ignored ->
                    throw new UnsupportedOperationException(
                            "GPT-OSS is text-only: embedding input is not supported");
        }
        state.advance(n, batch.outputs());
    }

    @Override
    public MemoryView<?> head(State state, int output) {
        int dim = configuration.embeddingLength;
        int row = state.lastChunkLen - state.outputCount + output;
        return Parallel.onDecodePool(
                () -> {
                    tailAt(state, row);
                    Norms.rmsnorm(
                            state.normed,
                            0,
                            state.tail,
                            0,
                            weights.outputNorm,
                            dim,
                            configuration.rmsNormEps);
                    MatMul.gemv(
                            weights.outputWeight,
                            state.normed,
                            state.logits,
                            configuration.vocabularySize,
                            dim);
                    return state.logits;
                });
    }

    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        RoPE.fill(
                state.ropeCos,
                state.ropeSin,
                startPos,
                seqLen,
                configuration.headSize / 2,
                weights.rope);
        embed(state, tokens, tokenOffset, seqLen);
        int lastLayer = configuration.numberOfLayers - 1;
        for (int l = 0; l < lastLayer; l++) layer(state, l, startPos, seqLen);
        writeKv(state, lastLayer, startPos, seqLen);
    }

    private void embed(State state, int[] tokens, int tokenOffset, int seqLen) {
        Views.checkAlive(weights.tokenEmbeddings, "tokenEmbeddings");
        int dim = configuration.embeddingLength;
        for (int s = 0; s < seqLen; s++)
            Convert.copyToF32(
                    weights.tokenEmbeddings,
                    (long) tokens[tokenOffset + s] * dim,
                    state.residual,
                    (long) s * dim,
                    dim);
    }

    private void layer(State state, int l, int startPos, int seqLen) {
        attention(state, l, startPos, seqLen);
        moeFeedForward(state, l, state.residual, seqLen);
        if (Trace.ENABLED)
            Trace.sum("l_out-" + l, state.residual, seqLen * configuration.embeddingLength);
    }

    private void attention(State state, int l, int startPos, int seqLen) {
        Configuration c = configuration;
        int dim = c.embeddingLength, headSize = c.headSize, halfHead = headSize / 2;
        int heads = c.numberOfHeads, kvHeads = c.numberOfKeyValueHeads;
        int queryDim = c.queryDim(), kvDim = c.kvDim(), kvMul = heads / kvHeads;
        boolean swa = c.isSWA(l);
        LayerWeights layer = weights.layers[l];
        AttentionWeights attn = layer.attention;
        MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];

        Norms.rmsnormRows(state.normed, state.residual, layer.attnNorm, seqLen, dim, c.rmsNormEps);
        MatMul.gemm(attn.wq, state.normed, dim, state.query, queryDim, queryDim, seqLen, dim);
        Ops.addRowBiasInPlace(state.query, 0, attn.qBias, 0, seqLen, queryDim);
        Parallel.forRows(
                seqLen,
                s -> {
                    for (int h = 0; h < heads; h++)
                        RoPE.applyNeox(
                                state.query,
                                (long) s * queryDim + (long) h * headSize,
                                s,
                                state.ropeCos,
                                state.ropeSin,
                                halfHead);
                });

        MatMul.gemm(attn.wk, state.normed, dim, bK, kvDim, kvDim, seqLen, dim);
        MatMul.gemm(attn.wv, state.normed, dim, bV, kvDim, kvDim, seqLen, dim);
        Ops.addRowBiasInPlace(bK, 0, attn.kBias, 0, seqLen, kvDim);
        Ops.addRowBiasInPlace(bV, 0, attn.vBias, 0, seqLen, kvDim);
        Parallel.forRows(
                seqLen,
                s -> {
                    for (int h = 0; h < kvHeads; h++)
                        RoPE.applyNeox(
                                bK,
                                (long) s * kvDim + (long) h * headSize,
                                s,
                                state.ropeCos,
                                state.ropeSin,
                                halfHead);
                });

        float scale = 1f / (float) Math.sqrt(headSize);
        if (seqLen > 1)
            FlashAttention.slidingWindowPrefill(
                    state.query,
                    state.attnOut,
                    state.keyCache[l],
                    state.valueCache[l],
                    bK,
                    bV,
                    heads,
                    startPos,
                    seqLen,
                    headSize,
                    kvDim,
                    queryDim,
                    kvDim,
                    kvMul,
                    scale,
                    swa ? c.slidingWindow : 0,
                    swa ? c.slidingWindow - 1 : 0,
                    attn.sinks);
        else
            FlashAttention.flashDecode(
                    state.query,
                    state.attnOut,
                    state.keyCache[l],
                    state.valueCache[l],
                    bK,
                    bV,
                    heads,
                    startPos,
                    c.attentionStart(l, startPos),
                    headSize,
                    kvDim,
                    kvMul,
                    scale,
                    swa ? c.slidingWindow - 1 : 0,
                    attn.sinks,
                    state.decodeScratch);

        MatMul.gemm(attn.wo, state.attnOut, queryDim, state.branchOut, dim, dim, seqLen, queryDim);
        Ops.addRowBiasInPlace(state.branchOut, 0, attn.oBias, 0, seqLen, dim);
        Ops.addInPlace(state.residual, 0, state.branchOut, 0, seqLen * dim);
        commitKv(state, l, startPos, seqLen);
    }

    private void moeFeedForward(
            State state, int l, MemoryView<MemorySegment> residual, int seqLen) {
        Configuration c = configuration;
        int dim = c.embeddingLength, expertFf = c.expertFeedForwardLength;
        int experts = c.expertCount, topK = c.expertUsedCount;
        MoeFfnWeights moe = weights.layers[l].moe;

        Norms.rmsnormRows(
                state.normed, residual, weights.layers[l].postAttnNorm, seqLen, dim, c.rmsNormEps);
        MatMul.gemm(moe.router, state.normed, dim, state.moeRouter, experts, experts, seqLen, dim);
        Ops.addRowBiasInPlace(state.moeRouter, 0, moe.routerBias, 0, seqLen, experts);
        selectExperts(state, seqLen);

        state.moeRouting.seqLen = seqLen;
        Moe.dispatch(
                state.moeRouting,
                dim,
                state.normed,
                state.moeGather,
                state.moeExpertOut,
                state.moeOut,
                null,
                (e, n, gather, out) -> {
                    long gateUpOffset = (long) e * expertFf * dim;
                    MatMul.gemm(
                            moe.gateExps,
                            gateUpOffset,
                            gather,
                            dim,
                            state.hidden,
                            expertFf,
                            expertFf,
                            n,
                            dim);
                    MatMul.gemm(
                            moe.upExps,
                            gateUpOffset,
                            gather,
                            dim,
                            state.hidden2,
                            expertFf,
                            expertFf,
                            n,
                            dim);
                    Ops.addRowBiasInPlace(
                            state.hidden, 0, moe.gateBias, (long) e * expertFf, n, expertFf);
                    Ops.addRowBiasInPlace(
                            state.hidden2, 0, moe.upBias, (long) e * expertFf, n, expertFf);
                    Parallel.forRows(
                            n,
                            row ->
                                    Activations.clampedSwigluMultiply(
                                            state.hidden,
                                            row * expertFf,
                                            state.hidden2,
                                            row * expertFf,
                                            expertFf));
                    MatMul.gemm(
                            moe.downExps,
                            (long) e * dim * expertFf,
                            state.hidden,
                            expertFf,
                            out,
                            dim,
                            dim,
                            n,
                            expertFf);
                    Ops.addRowBiasInPlace(out, 0, moe.downBias, (long) e * dim, n, dim);
                });
        Ops.addInPlace(residual, 0, state.moeOut, 0, seqLen * dim);
    }

    private void selectExperts(State state, int seqLen) {
        int experts = configuration.expertCount, topK = configuration.expertUsedCount;
        Moe.selectTopK(
                state.moeRouter,
                seqLen,
                experts,
                topK,
                state.moeRowTopE,
                state.moeRowTopP,
                state.moeExpertCounts);
        for (int s = 0; s < seqLen; s++) {
            int base = s * topK;
            // softmax over the selected raw logits, × expertWeightsScale
            float max = Float.NEGATIVE_INFINITY;
            for (int k = 0; k < topK; k++) max = Math.max(max, state.moeRowTopP[base + k]);
            float sum = 0f;
            for (int k = 0; k < topK; k++) {
                float value = (float) Math.exp(state.moeRowTopP[base + k] - max);
                state.moeRowTopP[base + k] = value;
                sum += value;
            }
            float inverse = sum == 0f ? 0f : 1f / sum;
            for (int k = 0; k < topK; k++)
                state.moeRowTopP[base + k] =
                        state.moeRowTopP[base + k] * inverse * configuration.expertWeightsScale;
        }
    }

    private void writeKv(State state, int l, int startPos, int seqLen) {
        Configuration c = configuration;
        int dim = c.embeddingLength, headSize = c.headSize, halfHead = headSize / 2;
        int kvHeads = c.numberOfKeyValueHeads, kvDim = c.kvDim();
        LayerWeights layer = weights.layers[l];
        AttentionWeights attn = layer.attention;
        MemoryView<MemorySegment> bK = state.batchK[l], bV = state.batchV[l];

        Norms.rmsnormRows(state.normed, state.residual, layer.attnNorm, seqLen, dim, c.rmsNormEps);
        MatMul.gemm(attn.wk, state.normed, dim, bK, kvDim, kvDim, seqLen, dim);
        MatMul.gemm(attn.wv, state.normed, dim, bV, kvDim, kvDim, seqLen, dim);
        Ops.addRowBiasInPlace(bK, 0, attn.kBias, 0, seqLen, kvDim);
        Ops.addRowBiasInPlace(bV, 0, attn.vBias, 0, seqLen, kvDim);
        Parallel.forRows(
                seqLen,
                s -> {
                    for (int h = 0; h < kvHeads; h++)
                        RoPE.applyNeox(
                                bK,
                                (long) s * kvDim + (long) h * headSize,
                                s,
                                state.ropeCos,
                                state.ropeSin,
                                halfHead);
                });
        commitKv(state, l, startPos, seqLen);
    }

    private void commitKv(State state, int l, int startPos, int seqLen) {
        int kvDim = configuration.kvDim();
        for (int s = 0; s < seqLen; s++) {
            long position = configuration.kvCacheIndex(l, startPos + s);
            Convert.f32ToF16(
                    state.batchK[l], (long) s * kvDim, state.keyCache[l], position * kvDim, kvDim);
            Convert.f32ToF16(
                    state.batchV[l],
                    (long) s * kvDim,
                    state.valueCache[l],
                    position * kvDim,
                    kvDim);
        }
    }

    private void tailAt(State state, int row) {
        Configuration c = configuration;
        int l = c.numberOfLayers - 1;
        int dim = c.embeddingLength, headSize = c.headSize, halfHead = headSize / 2;
        int heads = c.numberOfHeads, queryDim = c.queryDim(), kvDim = c.kvDim();
        int kvMul = heads / c.numberOfKeyValueHeads;
        boolean swa = c.isSWA(l);
        LayerWeights layer = weights.layers[l];
        AttentionWeights attn = layer.attention;
        int startPos = state.position - state.lastChunkLen;
        int position = startPos + row;

        Norms.rmsnorm(
                state.tailScratch,
                0,
                state.residual,
                (long) row * dim,
                layer.attnNorm,
                dim,
                c.rmsNormEps);
        MatMul.gemm(attn.wq, state.tailScratch, dim, state.query, queryDim, queryDim, 1, dim);
        Ops.addRowBiasInPlace(state.query, 0, attn.qBias, 0, 1, queryDim);
        RoPE.fill(state.ropeCos, state.ropeSin, position, 1, halfHead, weights.rope);
        for (int h = 0; h < heads; h++)
            RoPE.applyNeox(
                    state.query, (long) h * headSize, 0, state.ropeCos, state.ropeSin, halfHead);
        FlashAttention.flashDecode(
                state.query,
                state.attnOut,
                state.keyCache[l],
                state.valueCache[l],
                null,
                null,
                heads,
                position,
                c.attentionStart(l, position),
                headSize,
                kvDim,
                kvMul,
                1f / (float) Math.sqrt(headSize),
                swa ? c.slidingWindow - 1 : 0,
                attn.sinks,
                state.decodeScratch);
        MatMul.gemm(attn.wo, state.attnOut, queryDim, state.tailScratch, dim, dim, 1, queryDim);
        Ops.addRowBiasInPlace(state.tailScratch, 0, attn.oBias, 0, 1, dim);
        Ops.addScaledInto(state.tail, state.residual, (long) row * dim, state.tailScratch, dim, 1f);
        moeFeedForward(state, l, state.tail, 1);
    }

    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int headSize,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            double ropeTheta,
            float ropeScalingFactor,
            int ropeOrigCtx,
            int slidingWindow,
            boolean[] swaMask,
            int expertCount,
            int expertUsedCount,
            int expertFeedForwardLength,
            float expertWeightsScale)
            implements Config {
        public Configuration {
            if (embeddingLength <= 0
                    || numberOfLayers <= 0
                    || numberOfHeads <= 0
                    || numberOfKeyValueHeads <= 0
                    || headSize <= 0
                    || vocabularySize <= 0
                    || contextLength <= 0
                    || ropeOrigCtx <= 0
                    || expertCount <= 0
                    || expertFeedForwardLength <= 0)
                throw new IllegalArgumentException("model dimensions must be positive");
            if (slidingWindow <= 0 || Integer.bitCount(slidingWindow) != 1)
                throw new IllegalArgumentException(
                        "slidingWindow must be a power of 2, got " + slidingWindow);
            if (numberOfHeads % numberOfKeyValueHeads != 0)
                throw new IllegalArgumentException("query heads must be divisible by KV heads");
            if ((headSize & 1) != 0)
                throw new IllegalArgumentException("headSize must be even, got " + headSize);
            if (expertUsedCount <= 0 || expertUsedCount > expertCount)
                throw new IllegalArgumentException(
                        "expertUsedCount must be in (0, expertCount], got " + expertUsedCount);
            if (swaMask == null || swaMask.length != numberOfLayers)
                throw new IllegalArgumentException("swaMask length must equal numberOfLayers");
        }

        int queryDim() {
            return numberOfHeads * headSize;
        }

        int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        boolean isSWA(int layer) {
            return swaMask[layer];
        }

        int kvCachePositions(int layer, int capacity) {
            return isSWA(layer) ? Math.min(capacity, slidingWindow) : capacity;
        }

        int kvCacheIndex(int layer, int position) {
            return isSWA(layer) ? position & (slidingWindow - 1) : position;
        }

        int attentionStart(int layer, int position) {
            return isSWA(layer) ? Math.max(0, position - slidingWindow + 1) : 0;
        }
    }

    public record AttentionWeights(
            MemoryView<MemorySegment> wq,
            MemoryView<MemorySegment> wk,
            MemoryView<MemorySegment> wv,
            MemoryView<MemorySegment> wo,
            MemoryView<MemorySegment> qBias,
            MemoryView<MemorySegment> kBias,
            MemoryView<MemorySegment> vBias,
            MemoryView<MemorySegment> oBias,
            MemoryView<MemorySegment> sinks) {}

    public record MoeFfnWeights(
            MemoryView<MemorySegment> router,
            MemoryView<MemorySegment> routerBias,
            MemoryView<MemorySegment> gateExps,
            MemoryView<MemorySegment> gateBias,
            MemoryView<MemorySegment> upExps,
            MemoryView<MemorySegment> upBias,
            MemoryView<MemorySegment> downExps,
            MemoryView<MemorySegment> downBias) {}

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> postAttnNorm,
            AttentionWeights attention,
            MoeFfnWeights moe) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbeddings,
            LayerWeights[] layers,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> outputWeight,
            RoPE.Schedule rope) {}

    public static final class State extends BaseState {
        final int contextCapacity, batchCapacity;
        final MemoryView<MemorySegment> residual,
                normed,
                branchOut,
                attnOut,
                query,
                logits,
                tail,
                tailScratch;
        final MemoryView<MemorySegment> ropeCos, ropeSin;
        final FlashAttention.DecodeScratch decodeScratch =
                new FlashAttention.DecodeScratch(memoryArena());
        final MemoryView<MemorySegment>[] keyCache, valueCache, batchK, batchV;
        final MemoryView<MemorySegment> moeRouter, moeGather, moeExpertOut, moeOut, hidden, hidden2;
        final int[] moeExpertCounts, moeRowTopE;
        final float[] moeRowTopP;
        final Moe.Routing moeRouting;

        @SuppressWarnings("unchecked")
        State(Configuration c, int contextCapacity, int batchCapacity, Arena arena) {
            super(arena);
            if (contextCapacity > c.contextLength)
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds model contextLength "
                                + c.contextLength);
            this.contextCapacity = contextCapacity;
            this.batchCapacity = Math.max(1, batchCapacity);
            int rows = this.batchCapacity, dim = c.embeddingLength;
            int queryDim = c.queryDim(), kvDim = c.kvDim();
            residual = Views.allocateF32(memoryArena(), rows * dim);
            normed = Views.allocateF32(memoryArena(), rows * dim);
            branchOut = Views.allocateF32(memoryArena(), rows * dim);
            attnOut = Views.allocateF32(memoryArena(), rows * queryDim);
            query = Views.allocateF32(memoryArena(), rows * queryDim);
            logits = Views.allocateF32(memoryArena(), c.vocabularySize);
            tail = Views.allocateF32(memoryArena(), dim);
            tailScratch = Views.allocateF32(memoryArena(), dim);
            ropeCos = Views.allocateF32(memoryArena(), rows * c.headSize / 2);
            ropeSin = Views.allocateF32(memoryArena(), rows * c.headSize / 2);
            hidden = Views.allocateF32(memoryArena(), rows * c.expertFeedForwardLength);
            hidden2 = Views.allocateF32(memoryArena(), rows * c.expertFeedForwardLength);
            moeRouter = Views.allocateF32(memoryArena(), rows * c.expertCount);
            moeGather = Views.allocateF32(memoryArena(), rows * dim);
            moeExpertOut = Views.allocateF32(memoryArena(), rows * dim);
            moeOut = Views.allocateF32(memoryArena(), rows * dim);
            moeExpertCounts = new int[c.expertCount];
            moeRowTopE = new int[rows * c.expertUsedCount];
            moeRowTopP = new float[rows * c.expertUsedCount];
            moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts);
            moeRouting.topK = c.expertUsedCount;
            moeRouting.numExperts = c.expertCount;
            keyCache = new MemoryView[c.numberOfLayers];
            valueCache = new MemoryView[c.numberOfLayers];
            batchK = new MemoryView[c.numberOfLayers];
            batchV = new MemoryView[c.numberOfLayers];
            for (int l = 0; l < c.numberOfLayers; l++) {
                int positions = c.kvCachePositions(l, contextCapacity);
                keyCache[l] = Views.allocateF16(memoryArena(), positions, kvDim);
                valueCache[l] = Views.allocateF16(memoryArena(), positions, kvDim);
                batchK[l] = Views.allocateF32(memoryArena(), rows * kvDim);
                batchV[l] = Views.allocateF32(memoryArena(), rows * kvDim);
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

        @Override
        public void reset() {
            resumeAt(0);
        }
    }

    public static GptOss loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(channel, gguf, arena);
        }
    }

    public static GptOss loadModel(FileChannel channel, GGUF gguf, Arena arena) throws IOException {
        return loadModel(channel, gguf, arena, null);
    }

    public static GptOss loadModel(FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration c = loadConfiguration(gguf, tokenizer);
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new GptOss(c, tokenizer, loadWeights(tensors, c));
    }

    private static Configuration loadConfiguration(GGUF gguf, Tokenizer tokenizer) {
        String arch = "gpt-oss";
        int layers = gguf.getValue(int.class, arch + ".block_count");
        int dim = gguf.getValue(int.class, arch + ".embedding_length");
        int heads = gguf.getValue(int.class, arch + ".attention.head_count");
        int context = gguf.getValue(int.class, arch + ".context_length");
        boolean[] swa = new boolean[layers];
        for (int l = 0; l < layers; l++) swa[l] = l % 2 == 0;
        return new Configuration(
                dim,
                layers,
                heads,
                gguf.getValue(int.class, arch + ".attention.head_count_kv"),
                gguf.getValueOrDefault(int.class, arch + ".attention.key_length", dim / heads),
                tokenizer.vocabulary().size(),
                context,
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f),
                gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 150000f),
                gguf.getValueOrDefault(float.class, arch + ".rope.scaling.factor", 1f),
                gguf.getValueOrDefault(
                        int.class, arch + ".rope.scaling.original_context_length", context),
                gguf.getValue(int.class, arch + ".attention.sliding_window"),
                swa,
                gguf.getValueOrDefault(int.class, arch + ".expert_count", 0),
                gguf.getValueOrDefault(int.class, arch + ".expert_used_count", 0),
                gguf.getValueOrDefault(int.class, arch + ".expert_feed_forward_length", 0),
                gguf.getValueOrDefault(float.class, arch + ".expert_weights_scale", 1f));
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

    static Weights loadWeights(Map<String, MemoryView<MemorySegment>> tensors, Configuration c) {
        LayerWeights[] layers = new LayerWeights[c.numberOfLayers];
        for (int l = 0; l < layers.length; l++) {
            String p = "blk." + l + ".";
            layers[l] =
                    new LayerWeights(
                            requireF32(tensors, p + "attn_norm.weight"),
                            requireF32(tensors, p + "post_attention_norm.weight"),
                            new AttentionWeights(
                                    require(tensors, p + "attn_q.weight"),
                                    require(tensors, p + "attn_k.weight"),
                                    require(tensors, p + "attn_v.weight"),
                                    require(tensors, p + "attn_output.weight"),
                                    requireF32(tensors, p + "attn_q.bias"),
                                    requireF32(tensors, p + "attn_k.bias"),
                                    requireF32(tensors, p + "attn_v.bias"),
                                    requireF32(tensors, p + "attn_output.bias"),
                                    requireF32(tensors, p + "attn_sinks.weight")),
                            new MoeFfnWeights(
                                    require(tensors, p + "ffn_gate_inp.weight"),
                                    requireF32(tensors, p + "ffn_gate_inp.bias"),
                                    require(tensors, p + "ffn_gate_exps.weight"),
                                    requireF32(tensors, p + "ffn_gate_exps.bias"),
                                    require(tensors, p + "ffn_up_exps.weight"),
                                    requireF32(tensors, p + "ffn_up_exps.bias"),
                                    require(tensors, p + "ffn_down_exps.weight"),
                                    requireF32(tensors, p + "ffn_down_exps.bias")));
        }
        MemoryView<MemorySegment> tokenEmbeddings = require(tensors, "token_embd.weight");
        return new Weights(
                tokenEmbeddings,
                layers,
                requireF32(tensors, "output_norm.weight"),
                tensors.containsKey("output.weight")
                        ? require(tensors, "output.weight")
                        : tokenEmbeddings,
                RoPE.yarn(
                        c.headSize,
                        c.ropeTheta,
                        c.ropeScalingFactor,
                        c.ropeOrigCtx,
                        32f,
                        1f,
                        1f,
                        1f));
    }
}
