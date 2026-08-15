package com.qxotic.jinfer.x.models.nemotronh;

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
import com.qxotic.jinfer.x.kernels.Convolutions;
import com.qxotic.jinfer.x.kernels.FlashAttention;
import com.qxotic.jinfer.x.kernels.Mamba2;
import com.qxotic.jinfer.x.kernels.MatMul;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Moe;
import com.qxotic.jinfer.x.kernels.Norms;
import com.qxotic.jinfer.x.kernels.Ops;
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
import java.util.Optional;

/** Text-only MemoryView port of the Nemotron-H SSM/attention/MoE decoder. */
public final class NemotronH
        implements LanguageModel<NemotronH.Configuration, NemotronH.Weights, NemotronH.State> {
    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    NemotronH(Configuration configuration, Tokenizer tokenizer, Weights weights) {
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
        return Optional.of(new NemotronHStateCodec(configuration));
    }

    @Override
    public State newState(int contextCapacity, int batchCapacity, Arena arena) {
        return new State(configuration, contextCapacity, batchCapacity, arena);
    }

    @Override
    public void forward(State state, Batch batch) {
        int rows = batch.count();
        if (rows <= 0)
            throw new IllegalArgumentException("Nemotron-H token batch must not be empty");
        if (rows > state.batchCapacity)
            throw new IllegalArgumentException(
                    "batch " + rows + " exceeds batchCapacity " + state.batchCapacity);
        int start = state.position();
        if (start + rows > state.contextCapacity)
            throw new IllegalArgumentException(
                    "ingest of "
                            + rows
                            + " at position "
                            + start
                            + " exceeds contextCapacity "
                            + state.contextCapacity);
        int[] tokens =
                switch (batch.input()) {
                    case Batch.Input.Tokens t -> t.ids();
                    case Batch.Input.Sequences ignored ->
                            throw new UnsupportedOperationException(
                                    "x Nemotron-H does not support packed sequences");
                    case Batch.Input.Embeddings ignored ->
                            throw new UnsupportedOperationException(
                                    "x Nemotron-H is text-only and does not support embedding"
                                            + " input");
                };
        for (int token : tokens)
            if (token < 0 || token >= configuration.vocabularySize)
                throw new IllegalArgumentException(
                        "token id " + token + " outside [0," + configuration.vocabularySize + ")");
        if (rows == 1)
            Parallel.onDecodePool(
                    () -> {
                        forward(state, tokens, start, rows);
                        return null;
                    });
        else forward(state, tokens, start, rows);
        state.advance(rows, batch.outputs());
    }

    void forward(State state, int[] tokens, int startPos, int rows) {
        Configuration c = configuration;
        Views.checkAlive(weights.tokenEmbedding, "tokenEmbedding");
        Convert.gatherToF32(
                weights.tokenEmbedding, tokens, 0, rows, state.residual, 0, c.embeddingLength);
        for (int layer = 0; layer < c.numberOfLayers; layer++) {
            Norms.rmsnormRows(
                    state.normed,
                    state.residual,
                    weights.attnNorm[layer],
                    rows,
                    c.embeddingLength,
                    c.rmsNormEps);
            switch (c.layerTypes[layer]) {
                case SSM -> ssm(state, layer, rows);
                case ATTENTION -> attention(state, layer, startPos, rows);
                case MOE -> moe(state, layer, rows);
            }
            Ops.addInPlace(state.residual, 0, state.branch, 0, rows * c.embeddingLength);
            if (Trace.ENABLED)
                Trace.sum(
                        "l" + layer + "-" + c.layerTypes[layer],
                        state.residual,
                        rows * c.embeddingLength);
        }
    }

    private void ssm(State s, int layer, int rows) {
        Configuration c = configuration;
        SsmWeights w = weights.ssm[layer];
        int dim = c.embeddingLength, inner = c.ssmInnerSize, channels = c.ssmConvChannels();
        int heads = c.ssmTimeStepRank, projection = c.ssmInProjSize();
        MatMul.gemm(w.inProj, s.normed, dim, s.ssmProjection, projection, projection, rows, dim);
        float[] dtValues = s.ssmDtValues;
        for (int row = 0; row < rows; row++) {
            long source = (long) row * projection;
            Convert.copyF32(s.ssmProjection, source, s.ssmZ, (long) row * inner, inner);
            Convert.copyF32(
                    s.ssmProjection, source + inner, s.ssmXbc, (long) row * channels, channels);
            for (int h = 0; h < heads; h++) {
                long index = (long) row * heads + h;
                float value =
                        Activations.softplus(
                                Views.getFloat(
                                                s.ssmProjection,
                                                source
                                                        + 2L * inner
                                                        + 2L * c.ssmGroupCount * c.ssmStateSize
                                                        + h,
                                                "ssmProjection")
                                        + Views.getFloat(w.dtBias, h, "dtBias"));
                dtValues[Math.toIntExact(index)] = value;
            }
        }
        Views.copyFromArray(s.ssmDt, 0, dtValues, 0, rows * heads, "ssmDt");
        Convolutions.causalDepthwiseSilu(
                s.ssmXbc,
                w.conv1d,
                w.conv1dBias,
                s.convState[layer],
                s.ssmConvOut,
                rows,
                channels,
                c.ssmConvKernel);
        Mamba2.scan(
                s.ssmConvOut,
                s.ssmZ,
                s.ssmDt,
                w.a,
                w.d,
                s.recurrentState[layer],
                s.ssmOutput,
                rows,
                inner,
                heads,
                c.ssmGroupCount,
                c.ssmStateSize);
        Mamba2.groupedRmsNorm(
                s.ssmOutput, w.norm, s.ssmTmp, rows, inner, c.ssmGroupCount, c.rmsNormEps);
        MatMul.gemm(w.outProj, s.ssmTmp, inner, s.branch, dim, dim, rows, inner);
    }

    private void attention(State s, int layer, int startPos, int rows) {
        Configuration c = configuration;
        AttentionWeights w = weights.attention[layer];
        int dim = c.embeddingLength, queryDim = c.queryDim(), kvDim = c.kvDim();
        MatMul.gemm(w.wq, s.normed, dim, s.q, queryDim, queryDim, rows, dim);
        MatMul.gemm(w.wk, s.normed, dim, s.k, kvDim, kvDim, rows, dim);
        MatMul.gemm(w.wv, s.normed, dim, s.v, kvDim, kvDim, rows, dim);
        for (int row = 0; row < rows; row++) {
            Convert.f32ToF16(
                    s.k,
                    (long) row * kvDim,
                    s.keyCache[layer],
                    (long) (startPos + row) * kvDim,
                    kvDim);
            Convert.f32ToF16(
                    s.v,
                    (long) row * kvDim,
                    s.valueCache[layer],
                    (long) (startPos + row) * kvDim,
                    kvDim);
        }
        FlashAttention.causalPrefill(
                s.q,
                s.attentionOut,
                s.keyCache[layer],
                s.valueCache[layer],
                c.numberOfHeads,
                startPos,
                rows,
                c.headSize,
                kvDim,
                queryDim,
                c.numberOfHeads / c.numberOfKeyValueHeads);
        MatMul.gemm(w.wo, s.attentionOut, queryDim, s.branch, dim, dim, rows, queryDim);
    }

    private void moe(State s, int layer, int rows) {
        Configuration c = configuration;
        MoeWeights w = weights.moe[layer];
        int dim = c.embeddingLength, experts = c.expertCount;
        int topK = Math.min(c.expertUsedCount, experts), expertFfn = c.expertFeedForwardLength;
        MatMul.gemm(w.router, s.normed, dim, s.moeRouter, experts, experts, rows, dim);
        Ops.mapInPlace(s.moeRouter, 0, rows * experts, Activations::sigmoid);
        Arrays.fill(s.moeExpertCounts, 0);
        for (int row = 0; row < rows; row++) {
            selectTopK(s, w.expProbsB, row, experts, topK);
            float sum = 0f;
            if (c.expertWeightsNorm)
                for (int k = 0; k < topK; k++) sum += s.moeRowTopP[row * topK + k];
            sum = Math.max(sum, 6.103515625e-5f);
            for (int k = 0; k < topK; k++) {
                int index = row * topK + k;
                if (c.expertWeightsNorm) s.moeRowTopP[index] /= sum;
                s.moeRowTopP[index] *= c.expertWeightsScale;
                s.moeExpertCounts[s.moeRowTopE[index]]++;
            }
        }
        Moe.Routing routing = s.moeRouting;
        routing.seqLen = rows;
        routing.topK = topK;
        routing.numExperts = experts;
        Moe.dispatch(
                routing,
                dim,
                s.normed,
                s.moeGather,
                s.moeDown,
                s.branch,
                null,
                (expert, n, gather, out) -> {
                    MatMul.gemm(
                            w.upExps,
                            (long) expert * expertFfn * dim,
                            gather,
                            dim,
                            s.moeHidden,
                            expertFfn,
                            expertFfn,
                            n,
                            dim);
                    Parallel.forRows(
                            n, row -> Activations.reluSqr(s.moeHidden, row * expertFfn, expertFfn));
                    MatMul.gemm(
                            w.downExps,
                            (long) expert * dim * expertFfn,
                            s.moeHidden,
                            expertFfn,
                            out,
                            dim,
                            dim,
                            n,
                            expertFfn);
                });
        if (w.upShared != null) {
            int shared = c.expertSharedFeedForwardLength;
            MatMul.gemm(w.upShared, s.normed, dim, s.sharedHidden, shared, shared, rows, dim);
            Parallel.forRows(
                    rows, row -> Activations.reluSqr(s.sharedHidden, row * shared, shared));
            MatMul.gemm(w.downShared, s.sharedHidden, shared, s.sharedOut, dim, dim, rows, shared);
            Ops.addInPlace(s.branch, 0, s.sharedOut, 0, rows * dim);
        }
    }

    private static void selectTopK(
            State s, MemoryView<MemorySegment> bias, int row, int experts, int topK) {
        long base = (long) row * experts;
        for (int k = 0; k < topK; k++) {
            int best = -1;
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int expert = 0; expert < experts; expert++) {
                boolean taken = false;
                for (int j = 0; j < k; j++)
                    if (s.moeRowTopE[row * topK + j] == expert) {
                        taken = true;
                        break;
                    }
                float score =
                        Views.getFloat(s.moeRouter, base + expert, "router")
                                + (bias == null ? 0f : Views.getFloat(bias, expert, "routerBias"));
                if (!taken && score > bestScore) {
                    bestScore = score;
                    best = expert;
                }
            }
            s.moeRowTopE[row * topK + k] = best;
            s.moeRowTopP[row * topK + k] = Views.getFloat(s.moeRouter, base + best, "router");
        }
    }

    @Override
    public MemoryView<?> head(State state, int output) {
        if (output < 0 || output >= state.outputCount)
            throw new IllegalArgumentException("output " + output + " outside retained outputs");
        Configuration c = configuration;
        int row = state.lastChunkLen - state.outputCount + output;
        return Parallel.onDecodePool(
                () -> {
                    Norms.rmsnorm(
                            state.normed,
                            0,
                            state.residual,
                            (long) row * c.embeddingLength,
                            weights.outputNorm,
                            c.embeddingLength,
                            c.rmsNormEps);
                    MatMul.gemv(
                            weights.outputWeight,
                            state.normed,
                            state.logits,
                            c.vocabularySize,
                            c.embeddingLength);
                    return state.logits;
                });
    }

    public enum LayerType {
        SSM,
        ATTENTION,
        MOE
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
            LayerType[] layerTypes,
            int ssmInnerSize,
            int ssmGroupCount,
            int ssmTimeStepRank,
            int ssmStateSize,
            int ssmConvKernel,
            int expertCount,
            int expertUsedCount,
            int expertFeedForwardLength,
            int expertSharedFeedForwardLength,
            boolean expertWeightsNorm,
            float expertWeightsScale)
            implements Config {
        int queryDim() {
            return numberOfHeads * headSize;
        }

        int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        int ssmConvChannels() {
            return ssmInnerSize + 2 * ssmGroupCount * ssmStateSize;
        }

        int ssmInProjSize() {
            return 2 * ssmInnerSize + 2 * ssmGroupCount * ssmStateSize + ssmTimeStepRank;
        }
    }

    public record AttentionWeights(
            MemoryView<MemorySegment> wq,
            MemoryView<MemorySegment> wk,
            MemoryView<MemorySegment> wv,
            MemoryView<MemorySegment> wo) {}

    public record SsmWeights(
            MemoryView<MemorySegment> inProj,
            MemoryView<MemorySegment> conv1d,
            MemoryView<MemorySegment> conv1dBias,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> d,
            MemoryView<MemorySegment> dtBias,
            MemoryView<MemorySegment> norm,
            MemoryView<MemorySegment> outProj) {}

    public record MoeWeights(
            MemoryView<MemorySegment> router,
            MemoryView<MemorySegment> expProbsB,
            MemoryView<MemorySegment> upExps,
            MemoryView<MemorySegment> downExps,
            MemoryView<MemorySegment> upShared,
            MemoryView<MemorySegment> downShared) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbedding,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> outputWeight,
            MemoryView<MemorySegment>[] attnNorm,
            AttentionWeights[] attention,
            SsmWeights[] ssm,
            MoeWeights[] moe) {}

    public static final class State extends BaseState {
        final int contextCapacity, batchCapacity;
        final MemoryView<MemorySegment> residual, normed, branch, logits;
        final MemoryView<MemorySegment> q, k, v, attentionOut;
        final MemoryView<MemorySegment> ssmProjection, ssmZ, ssmXbc, ssmDt, ssmConvOut;
        final MemoryView<MemorySegment> ssmOutput, ssmTmp;
        final float[] ssmDtValues;
        final MemoryView<MemorySegment> moeRouter, moeGather, moeDown, moeHidden;
        final MemoryView<MemorySegment> sharedHidden, sharedOut;
        final MemoryView<MemorySegment>[] keyCache, valueCache, convState, recurrentState;
        final int[] moeRowTopE, moeExpertCounts;
        final float[] moeRowTopP;
        final Moe.Routing moeRouting;

        @SuppressWarnings("unchecked")
        State(Configuration c, int contextCapacity, int batchCapacity, Arena arena) {
            super(arena);
            if (contextCapacity <= 0 || contextCapacity > c.contextLength)
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " outside [1,"
                                + c.contextLength
                                + "]");
            if (batchCapacity <= 0)
                throw new IllegalArgumentException("batchCapacity " + batchCapacity);
            this.contextCapacity = contextCapacity;
            this.batchCapacity = batchCapacity;
            int b = batchCapacity, dim = c.embeddingLength, qd = c.queryDim(), kvd = c.kvDim();
            int inner = c.ssmInnerSize, channels = c.ssmConvChannels();
            residual = Views.allocateF32(memoryArena(), b, dim);
            normed = Views.allocateF32(memoryArena(), b, dim);
            branch = Views.allocateF32(memoryArena(), b, dim);
            logits = Views.allocateF32(memoryArena(), c.vocabularySize);
            q = Views.allocateF32(memoryArena(), b, qd);
            k = Views.allocateF32(memoryArena(), b, kvd);
            v = Views.allocateF32(memoryArena(), b, kvd);
            attentionOut = Views.allocateF32(memoryArena(), b, qd);
            ssmProjection = Views.allocateF32(memoryArena(), b, Math.max(1, c.ssmInProjSize()));
            ssmZ = Views.allocateF32(memoryArena(), b, Math.max(1, inner));
            ssmXbc = Views.allocateF32(memoryArena(), b, Math.max(1, channels));
            ssmDt = Views.allocateF32(memoryArena(), b, Math.max(1, c.ssmTimeStepRank));
            ssmDtValues = new float[b * Math.max(1, c.ssmTimeStepRank)];
            ssmConvOut = Views.allocateF32(memoryArena(), b, Math.max(1, channels));
            ssmOutput = Views.allocateF32(memoryArena(), b, Math.max(1, inner));
            ssmTmp = Views.allocateF32(memoryArena(), b, Math.max(1, inner));
            int topK = Math.max(1, Math.min(c.expertUsedCount, c.expertCount));
            int experts = Math.max(1, c.expertCount),
                    expertFfn = Math.max(1, c.expertFeedForwardLength);
            int shared = Math.max(1, c.expertSharedFeedForwardLength);
            moeRouter = Views.allocateF32(memoryArena(), b, experts);
            moeGather = Views.allocateF32(memoryArena(), b, dim);
            moeDown = Views.allocateF32(memoryArena(), b, dim);
            moeHidden = Views.allocateF32(memoryArena(), b, expertFfn);
            sharedHidden = Views.allocateF32(memoryArena(), b, shared);
            sharedOut = Views.allocateF32(memoryArena(), b, dim);
            moeRowTopE = new int[b * topK];
            moeRowTopP = new float[b * topK];
            moeExpertCounts = new int[experts];
            moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts);
            keyCache = new MemoryView[c.numberOfLayers];
            valueCache = new MemoryView[c.numberOfLayers];
            convState = new MemoryView[c.numberOfLayers];
            recurrentState = new MemoryView[c.numberOfLayers];
            for (int layer = 0; layer < c.numberOfLayers; layer++) {
                switch (c.layerTypes[layer]) {
                    case ATTENTION -> {
                        keyCache[layer] = Views.allocateF16(memoryArena(), contextCapacity, kvd);
                        valueCache[layer] = Views.allocateF16(memoryArena(), contextCapacity, kvd);
                    }
                    case SSM -> {
                        convState[layer] =
                                Views.allocateF32(memoryArena(), c.ssmConvKernel - 1, channels);
                        recurrentState[layer] =
                                Views.allocateF32(memoryArena(), inner, c.ssmStateSize);
                    }
                    case MOE -> {}
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

        @Override
        public void reset() {
            resumeAt(0);
            for (MemoryView<MemorySegment> state : convState)
                if (state != null)
                    Ops.fillInPlace(state, 0, Math.toIntExact(state.logicalSize()), 0f);
            for (MemoryView<MemorySegment> state : recurrentState)
                if (state != null)
                    Ops.fillInPlace(state, 0, Math.toIntExact(state.logicalSize()), 0f);
        }
    }

    public static NemotronH loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(channel, gguf, arena);
        }
    }

    public static NemotronH loadModel(FileChannel channel, GGUF gguf, Arena arena)
            throws IOException {
        return loadModel(channel, gguf, arena, null);
    }

    public static NemotronH loadModel(
            FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer) throws IOException {
        String arch = gguf.getString("general.architecture");
        if (!arch.equals("nemotron_h") && !arch.equals("nemotron_h_moe"))
            throw new IllegalArgumentException("unsupported Nemotron-H architecture: " + arch);
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        int context = gguf.getValue(int.class, arch + ".context_length");
        int dim = gguf.getValue(int.class, arch + ".embedding_length");
        int layers = gguf.getValue(int.class, arch + ".block_count");
        int heads = gguf.getValue(int.class, arch + ".attention.head_count");
        int headSize =
                gguf.getValueOrDefault(int.class, arch + ".attention.key_length", dim / heads);
        float eps =
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        int[] kvByLayer = gguf.getValue(int[].class, arch + ".attention.head_count_kv");
        int[] ffnByLayer = gguf.getValue(int[].class, arch + ".feed_forward_length");
        LayerType[] types = new LayerType[layers];
        int kvHeads = 0;
        for (int layer = 0; layer < layers; layer++) {
            if (ffnByLayer[layer] > 0) types[layer] = LayerType.MOE;
            else if (kvByLayer[layer] > 0) {
                types[layer] = LayerType.ATTENTION;
                kvHeads = kvByLayer[layer];
            } else types[layer] = LayerType.SSM;
        }
        int inner = gguf.getValueOrDefault(int.class, arch + ".ssm.inner_size", 0);
        int groups = gguf.getValueOrDefault(int.class, arch + ".ssm.group_count", 0);
        int rank = gguf.getValueOrDefault(int.class, arch + ".ssm.time_step_rank", 0);
        int stateSize = gguf.getValueOrDefault(int.class, arch + ".ssm.state_size", 0);
        int kernel = gguf.getValueOrDefault(int.class, arch + ".ssm.conv_kernel", 0);
        int experts = gguf.getValueOrDefault(int.class, arch + ".expert_count", 0);
        int used = gguf.getValueOrDefault(int.class, arch + ".expert_used_count", 0);
        int expertFfn = gguf.getValueOrDefault(int.class, arch + ".expert_feed_forward_length", 0);
        int shared =
                gguf.getValueOrDefault(int.class, arch + ".expert_shared_feed_forward_length", 0);
        boolean normalize =
                gguf.getValueOrDefault(boolean.class, arch + ".expert_weights_norm", false);
        float scale = gguf.getValueOrDefault(float.class, arch + ".expert_weights_scale", 1f);
        if (dim <= 0
                || layers <= 0
                || heads <= 0
                || kvHeads <= 0
                || heads % kvHeads != 0
                || headSize <= 0
                || context <= 0
                || inner <= 0
                || rank <= 0
                || inner % rank != 0
                || groups <= 0
                || rank % groups != 0
                || stateSize <= 0
                || kernel <= 1
                || experts <= 0
                || used <= 0
                || used > experts
                || expertFfn <= 0)
            throw new IllegalArgumentException("inconsistent Nemotron-H dimensions");
        Configuration config =
                new Configuration(
                        dim,
                        layers,
                        heads,
                        kvHeads,
                        headSize,
                        tokenizer.vocabulary().size(),
                        context,
                        eps,
                        types,
                        inner,
                        groups,
                        rank,
                        stateSize,
                        kernel,
                        experts,
                        used,
                        expertFfn,
                        shared,
                        normalize,
                        scale);
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new NemotronH(config, tokenizer, loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, MemoryView<MemorySegment>> tensors, Configuration c) {
        int n = c.numberOfLayers;
        MemoryView<MemorySegment> embedding = require(tensors, "token_embd.weight");
        MemoryView<MemorySegment> output =
                tensors.containsKey("output.weight")
                        ? require(tensors, "output.weight")
                        : embedding;
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment>[] norms = new MemoryView[n];
        AttentionWeights[] attention = new AttentionWeights[n];
        SsmWeights[] ssm = new SsmWeights[n];
        MoeWeights[] moe = new MoeWeights[n];
        for (int layer = 0; layer < n; layer++) {
            String p = "blk." + layer + ".";
            norms[layer] = requireF32(tensors, p + "attn_norm.weight");
            switch (c.layerTypes[layer]) {
                case ATTENTION ->
                        attention[layer] =
                                new AttentionWeights(
                                        require(tensors, p + "attn_q.weight"),
                                        require(tensors, p + "attn_k.weight"),
                                        require(tensors, p + "attn_v.weight"),
                                        require(tensors, p + "attn_output.weight"));
                case SSM ->
                        ssm[layer] =
                                new SsmWeights(
                                        require(tensors, p + "ssm_in.weight"),
                                        requireF32(tensors, p + "ssm_conv1d.weight"),
                                        findF32(tensors, p + "ssm_conv1d.bias"),
                                        requireF32(tensors, p + "ssm_a"),
                                        requireF32(tensors, p + "ssm_d"),
                                        requireF32(tensors, p + "ssm_dt.bias"),
                                        requireF32(tensors, p + "ssm_norm.weight"),
                                        require(tensors, p + "ssm_out.weight"));
                case MOE ->
                        moe[layer] =
                                new MoeWeights(
                                        require(tensors, p + "ffn_gate_inp.weight"),
                                        findF32(tensors, p + "exp_probs_b.bias"),
                                        require(tensors, p + "ffn_up_exps.weight"),
                                        require(tensors, p + "ffn_down_exps.weight"),
                                        tensors.get(p + "ffn_up_shexp.weight"),
                                        tensors.get(p + "ffn_down_shexp.weight"));
            }
        }
        return new Weights(
                embedding,
                requireF32(tensors, "output_norm.weight"),
                output,
                norms,
                attention,
                ssm,
                moe);
    }

    private static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        return Objects.requireNonNull(tensors.get(name), name);
    }

    private static MemoryView<MemorySegment> requireF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> view = require(tensors, name);
        Views.requireDatatype(view, DataType.FP32, name);
        return view;
    }

    private static MemoryView<MemorySegment> findF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> view = tensors.get(name);
        if (view != null) Views.requireDatatype(view, DataType.FP32, name);
        return view;
    }
}
