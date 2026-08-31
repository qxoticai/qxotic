package com.qxotic.jinfer.models.bailingmoe3;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.MetadataValueType;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.KimiDeltaAttention;
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
import java.util.Optional;

/** Text-only BailingMoe3 decoder with KDA recurrent layers and compressed MLA layers. */
public final class BailingMoe3
        implements LanguageModel<BailingMoe3.Configuration, BailingMoe3.Weights, BailingMoe3.State>,
                SpeculativeDecoding<BailingMoe3.State> {
    private static final int SIGMOID_GATING = 2;

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    BailingMoe3(Configuration configuration, Tokenizer tokenizer, Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
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
    public Optional<CheckpointCodec<State>> checkpointCodec() {
        return Optional.of(new BailingMoe3CheckpointCodec(configuration));
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
        int rows = batch.count();
        if (rows <= 0 || rows > state.batchCapacity())
            throw new IllegalArgumentException("invalid BailingMoe3 batch size " + rows);
        int start = state.position();
        if (start + rows > state.contextCapacity())
            throw new IllegalArgumentException("batch exceeds the allocated context");
        int[] tokens =
                switch (batch.input()) {
                    case Batch.Input.Tokens t -> t.ids();
                    case Batch.Input.Sequences ignored ->
                            throw new UnsupportedOperationException(
                                    "packed sequences are unsupported");
                    case Batch.Input.Embeddings ignored ->
                            throw new UnsupportedOperationException(
                                    "embedding input is unsupported");
                };
        for (int token : tokens)
            if (token < 0 || token >= configuration.vocabularySize)
                throw new IllegalArgumentException("token id outside the vocabulary: " + token);
        forward(state, tokens, start, rows);
        state.advance(batch);
    }

    private void forward(State s, int[] tokens, int startPos, int rows) {
        Configuration c = configuration;
        Views.checkAlive(weights.tokenEmbedding, "tokenEmbedding");
        Convert.gatherToF32(
                weights.tokenEmbedding, tokens, 0, rows, s.residual, 0, c.embeddingLength);
        if (c.hasMtp()) mtpInputs(s, s.residual, rows);
        RoPE.fill(s.ropeCos, s.ropeSin, startPos, rows, weights.ropeHalf, weights.rope);
        for (int layer = 0; layer < c.numberOfLayers; layer++)
            decoderBlock(s, layer, startPos, rows);
        if (c.hasMtp()) synchronizeMtp(s, startPos, rows);
    }

    private void decoderBlock(State s, int layer, int startPos, int rows) {
        Configuration c = configuration;
        Norms.rmsnormRows(
                s.normed,
                s.residual,
                weights.attnNorm[layer],
                rows,
                c.embeddingLength,
                c.rmsNormEps);
        if (c.isAttention[layer]) mla(s, layer, startPos, rows);
        else kda(s, layer, rows);
        Ops.addInPlace(s.residual, 0, s.branch, 0, rows * c.embeddingLength);
        Norms.rmsnormRows(
                s.normed,
                s.residual,
                weights.ffnNorm[layer],
                rows,
                c.embeddingLength,
                c.rmsNormEps);
        if (layer < c.denseLeadingLayers) denseFfn(s, layer, rows);
        else moe(s, layer, rows);
        Ops.addInPlace(s.residual, 0, s.branch, 0, rows * c.embeddingLength);
        if (Trace.ENABLED) Trace.sum("l_out-" + layer, s.residual, rows * c.embeddingLength);
    }

    private void mtpInputs(State s, MemoryView<MemorySegment> embeddings, int rows) {
        Configuration c = configuration;
        int dim = c.embeddingLength;
        for (int row = 0; row < rows; row++)
            Norms.rmsnorm(
                    s.mtpConcat,
                    (long) row * 2 * dim,
                    embeddings,
                    (long) row * dim,
                    weights.nextn.embeddingNorm,
                    dim,
                    c.rmsNormEps);
    }

    private void synchronizeMtp(State s, int startPos, int rows) {
        Configuration c = configuration;
        int dim = c.embeddingLength;
        Norms.rmsnormRows(s.targetHidden, s.residual, weights.outputNorm, rows, dim, c.rmsNormEps);
        for (int row = 0; row < rows; row++) {
            MemoryView<MemorySegment> hidden = row == 0 ? s.pendingHidden : s.targetHidden;
            long hiddenOffset = row == 0 ? 0 : (long) (row - 1) * dim;
            Norms.rmsnorm(
                    s.mtpConcat,
                    (long) row * 2 * dim + dim,
                    hidden,
                    hiddenOffset,
                    weights.nextn.hiddenNorm,
                    dim,
                    c.rmsNormEps);
        }
        Convert.copyF32(s.targetHidden, (long) (rows - 1) * dim, s.pendingHidden, 0, dim);
        MatMul.gemm(weights.nextn.inputProjection, s.mtpConcat, s.residual, rows);
        decoderBlock(s, c.mtpLayer(), startPos, rows);
    }

    /** Fills {@code candidates[1..depth]} from the target seed in {@code candidates[0]}. */
    void draft(State state, int depth, int[] candidates) {
        MemoryView<MemorySegment> hidden = state.pendingHidden;
        int token = candidates[0];
        int position = state.position();
        for (int i = 1; i <= depth; i++) {
            draftOne(state, token, hidden, position + i - 1);
            token = Ops.argmax(state.logits, 0, configuration.vocabularySize);
            candidates[i] = token;
            hidden = state.normed;
        }
    }

    private void draftOne(State state, int token, MemoryView<MemorySegment> hidden, int position) {
        Configuration c = configuration;
        NextNWeights nextn = weights.nextn;
        int dim = c.embeddingLength;
        RoPE.fill(state.ropeCos, state.ropeSin, position, 1, weights.ropeHalf, weights.rope);
        Norms.rmsnorm(state.mtpConcat, dim, hidden, 0, nextn.hiddenNorm, dim, c.rmsNormEps);
        Views.checkAlive(weights.tokenEmbedding, "tokenEmbedding");
        Convert.copyToF32(weights.tokenEmbedding, (long) token * dim, state.normed, 0, dim);
        Norms.rmsnorm(state.mtpConcat, 0, state.normed, 0, nextn.embeddingNorm, dim, c.rmsNormEps);
        MatMul.gemv(nextn.inputProjection, state.mtpConcat, state.residual);
        decoderBlock(state, c.mtpLayer(), position, 1);
        Norms.rmsnorm(state.normed, 0, state.residual, 0, nextn.outputNorm, dim, c.rmsNormEps);
        Views.checkAlive(weights.outputWeight, "outputWeight");
        MatMul.gemv(weights.outputWeight, state.normed, state.logits);
    }

    private void kda(State s, int layer, int rows) {
        Configuration c = configuration;
        KdaWeights w = weights.kda[layer];
        int inner = c.kdaInnerSize();
        MatMul.gemm(w.q, s.normed, s.kdaQ, rows);
        MatMul.gemm(w.k, s.normed, s.kdaK, rows);
        MatMul.gemm(w.v, s.normed, s.kdaV, rows);
        Convolutions.causalDepthwiseSilu(
                s.kdaQ, w.qConv, s.qConvState[layer], s.kdaQConv, rows, inner, c.convKernel);
        Convolutions.causalDepthwiseSilu(
                s.kdaK, w.kConv, s.kConvState[layer], s.kdaKConv, rows, inner, c.convKernel);
        Convolutions.causalDepthwiseSilu(
                s.kdaV, w.vConv, s.vConvState[layer], s.kdaVConv, rows, inner, c.convKernel);
        MatMul.gemm(w.f, s.normed, s.kdaGateProjection, rows);
        MatMul.gemm(w.beta, s.normed, s.kdaBetaProjection, rows);
        KimiDeltaAttention.gates(
                s.kdaGateProjection,
                s.kdaBetaProjection,
                w.dtBias,
                w.a,
                s.kdaGate,
                s.kdaBeta,
                rows,
                c.numberOfHeads,
                c.kdaHeadDim,
                c.safeKdaGate,
                c.kdaGateLowerBound);
        KimiDeltaAttention.normalizeQk(
                s.kdaQConv, s.kdaKConv, rows, c.numberOfHeads, c.kdaHeadDim, c.rmsNormEps);
        KimiDeltaAttention.scan(
                s.kdaQConv,
                s.kdaKConv,
                s.kdaVConv,
                s.kdaGate,
                s.kdaBeta,
                s.recurrentState[layer],
                s.kdaOutput,
                s.kdaDecay,
                rows,
                c.numberOfHeads,
                c.kdaHeadDim);
        MatMul.gemm(w.outputGate, s.normed, s.kdaOutputGate, rows);
        KimiDeltaAttention.postNorm(
                s.kdaOutput,
                s.kdaOutputGate,
                w.outputNorm,
                s.kdaNormed,
                rows,
                c.numberOfHeads,
                c.kdaHeadDim,
                c.rmsNormEps);
        MatMul.gemm(w.output, s.kdaNormed, s.branch, rows);
        if (Trace.ENABLED) Trace.sum("kda_out-" + layer, s.branch, rows * c.embeddingLength);
    }

    private void mla(State s, int layer, int startPos, int rows) {
        Configuration c = configuration;
        MlaWeights w = weights.mla[layer];
        int heads = c.numberOfHeads, qk = c.mlaQkDim(), packed = c.mlaPackedDim();
        if (w.q != null) {
            MatMul.gemm(w.q, s.normed, s.mlaQAll, rows);
        } else {
            MatMul.gemm(w.qA, s.normed, s.mlaQA, rows);
            Norms.rmsnormRows(s.mlaQA, s.mlaQA, w.qANorm, rows, c.qLoraRank, c.rmsNormEps);
            MatMul.gemm(w.qB, s.mlaQA, s.mlaQAll, rows);
        }
        MatMul.gemm(w.kvA, s.normed, s.mlaKvAll, rows);
        for (int row = 0; row < rows; row++) {
            Norms.rmsnorm(
                    s.mlaLatent,
                    (long) row * c.kvLoraRank,
                    s.mlaKvAll,
                    (long) row * c.mlaCacheDim(),
                    w.kvANorm,
                    c.kvLoraRank,
                    c.rmsNormEps);
            Convert.copyF32(
                    s.mlaLatent,
                    (long) row * c.kvLoraRank,
                    s.mlaKvAll,
                    (long) row * c.mlaCacheDim(),
                    c.kvLoraRank);
            RoPE.applyNeox(
                    s.mlaKvAll,
                    (long) row * c.mlaCacheDim() + c.kvLoraRank,
                    row,
                    s.ropeCos,
                    s.ropeSin,
                    weights.ropeHalf);
            for (int head = 0; head < heads; head++)
                RoPE.applyNeox(
                        s.mlaQAll,
                        (long) row * heads * qk + (long) head * qk + c.qkNopeDim,
                        row,
                        s.ropeCos,
                        s.ropeSin,
                        weights.ropeHalf);
        }
        for (int head = 0; head < heads; head++) {
            for (int row = 0; row < rows; row++)
                Convert.copyF32(
                        s.mlaQAll,
                        (long) row * heads * qk + (long) head * qk,
                        s.mlaQNope,
                        (long) row * c.qkNopeDim,
                        c.qkNopeDim);
            MatMul.gemm(w.kB[head], s.mlaQNope, s.mlaQLatent, rows);
            for (int row = 0; row < rows; row++) {
                long packedHead = (long) row * packed + (long) head * c.mlaCacheDim();
                Convert.copyF32(
                        s.mlaQLatent,
                        (long) row * c.kvLoraRank,
                        s.mlaQPacked,
                        packedHead,
                        c.kvLoraRank);
                Convert.copyF32(
                        s.mlaQAll,
                        (long) row * heads * qk + (long) head * qk + c.qkNopeDim,
                        s.mlaQPacked,
                        packedHead + c.kvLoraRank,
                        c.ropeDimensionCount);
            }
        }
        for (int row = 0; row < rows; row++)
            Convert.f32ToF16(
                    s.mlaKvAll,
                    (long) row * c.mlaCacheDim(),
                    s.attentionCache[layer],
                    (long) (startPos + row) * c.mlaCacheDim(),
                    c.mlaCacheDim());
        FlashAttention.causalPrefill(
                s.mlaQPacked,
                s.mlaAttentionOut,
                s.attentionCache[layer],
                s.attentionCache[layer],
                heads,
                startPos,
                rows,
                c.mlaCacheDim(),
                c.mlaCacheDim(),
                packed,
                heads,
                1.0f / (float) Math.sqrt(qk));
        for (int head = 0; head < heads; head++) {
            for (int row = 0; row < rows; row++)
                Convert.copyF32(
                        s.mlaAttentionOut,
                        (long) row * packed + (long) head * c.mlaCacheDim(),
                        s.mlaLatent,
                        (long) row * c.kvLoraRank,
                        c.kvLoraRank);
            MatMul.gemm(w.vB[head], s.mlaLatent, s.mlaVHead, rows);
            for (int row = 0; row < rows; row++)
                Convert.copyF32(
                        s.mlaVHead,
                        (long) row * c.mlaValueHeadDim,
                        s.mlaProjected,
                        (long) row * heads * c.mlaValueHeadDim + (long) head * c.mlaValueHeadDim,
                        c.mlaValueHeadDim);
        }
        MatMul.gemm(w.gate, s.normed, s.mlaGate, rows);
        KimiDeltaAttention.headSigmoidMultiply(
                s.mlaProjected, s.mlaGate, rows, heads, c.mlaValueHeadDim);
        MatMul.gemm(w.output, s.mlaProjected, s.branch, rows);
        if (Trace.ENABLED) Trace.sum("mla_out-" + layer, s.branch, rows * c.embeddingLength);
    }

    private void denseFfn(State s, int layer, int rows) {
        FfnWeights w = weights.ffn[layer];
        MatMul.gemm(w.gate, s.normed, s.denseHidden, rows);
        MatMul.gemm(w.up, s.normed, s.denseHidden2, rows);
        Activations.siluMultiply(
                s.denseHidden, 0, s.denseHidden2, 0, rows * configuration.feedForwardLength);
        MatMul.gemm(w.down, s.denseHidden, s.branch, rows);
    }

    private void moe(State s, int layer, int rows) {
        Configuration c = configuration;
        MoeWeights w = weights.moe[layer];
        MatMul.gemm(w.router, s.normed, s.moeRouter, rows);
        Ops.mapInPlace(s.moeRouter, 0, rows * c.expertCount, Activations::sigmoid);
        Convert.copyF32(s.moeRouter, 0, s.moeSelection, 0, (long) rows * c.expertCount);
        Ops.addRowBiasInPlace(s.moeSelection, 0, w.selectionBias, 0, rows, c.expertCount);
        Moe.selectTopKGrouped(
                s.moeSelection,
                s.moeRouter,
                rows,
                c.expertCount,
                c.expertUsedCount,
                c.expertGroupCount,
                c.expertGroupCountUsed,
                s.moeRowTopE,
                s.moeRowTopP,
                s.moeExpertCounts,
                s.moeGroupScores,
                s.moeGroupMask);
        if (c.normalizeExpertWeights) Moe.normalizeTopP(s.moeRowTopP, rows, c.expertUsedCount);
        for (int i = 0; i < rows * c.expertUsedCount; i++) s.moeRowTopP[i] *= c.expertWeightsScale;
        Moe.Routing routing = s.moeRouting;
        routing.seqLen = rows;
        routing.topK = c.expertUsedCount;
        routing.numExperts = c.expertCount;
        Moe.dispatch(
                routing,
                c.embeddingLength,
                s.normed,
                s.moeGather,
                s.moeDown,
                s.branch,
                null,
                (expert, n, gather, out) -> {
                    MatMul.gemm(w.expertGate[expert], gather, s.moeHidden, n);
                    MatMul.gemm(w.expertUp[expert], gather, s.moeHidden2, n);
                    swiglu(
                            s.moeHidden,
                            s.moeHidden2,
                            n * c.expertFeedForwardLength,
                            c.expertSwiGluClamp[layer]);
                    MatMul.gemm(w.expertDown[expert], s.moeHidden, out, n);
                });
        MatMul.gemm(w.sharedGate, s.normed, s.sharedHidden, rows);
        MatMul.gemm(w.sharedUp, s.normed, s.sharedHidden2, rows);
        swiglu(
                s.sharedHidden,
                s.sharedHidden2,
                rows * c.sharedFeedForwardLength,
                c.sharedSwiGluClamp[layer]);
        MatMul.gemm(w.sharedDown, s.sharedHidden, s.sharedOut, rows);
        Ops.addInPlace(s.branch, 0, s.sharedOut, 0, rows * c.embeddingLength);
    }

    private static void swiglu(
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> up,
            int elements,
            float clamp) {
        if (!(clamp > 1e-6f)) {
            Activations.siluMultiply(gate, 0, up, 0, elements);
            return;
        }
        Ops.siluInPlace(gate, 0, elements);
        Ops.clampInPlace(gate, 0, elements, Float.NEGATIVE_INFINITY, clamp);
        Ops.clampInPlace(up, 0, elements, -clamp, clamp);
        Ops.multiplyInPlace(gate, 0, up, 0, elements);
    }

    @Override
    public MemoryView<?> logits(State state, int output) {
        return state.exclusively(
                () -> {
                    if (output < 0 || output >= state.outputCount())
                        throw new IllegalArgumentException("invalid output index " + output);
                    int row = state.lastBatchSize() - state.outputCount() + output;
                    Views.checkAlive(weights.outputWeight, "outputWeight");
                    if (configuration.hasMtp()) {
                        Convert.copyF32(
                                state.targetHidden,
                                (long) row * configuration.embeddingLength,
                                state.normed,
                                0,
                                configuration.embeddingLength);
                    } else {
                        Norms.rmsnorm(
                                state.normed,
                                0,
                                state.residual,
                                (long) row * configuration.embeddingLength,
                                weights.outputNorm,
                                configuration.embeddingLength,
                                configuration.rmsNormEps);
                    }
                    MatMul.gemv(weights.outputWeight, state.normed, state.logits);
                    if (Trace.ENABLED)
                        Trace.sum("result_output", state.logits, configuration.vocabularySize);
                    Reference.reachabilityFence(this);
                    return state.logits;
                });
    }

    void logitsAll(State state, MemoryView<MemorySegment> destination) {
        int dim = configuration.embeddingLength;
        int rows = state.outputCount();
        int first = state.lastBatchSize() - rows;
        Views.checkAlive(weights.outputWeight, "outputWeight");
        if (configuration.hasMtp()) {
            Convert.copyF32(state.targetHidden, (long) first * dim, state.normed, 0, rows * dim);
        } else {
            for (int row = 0; row < rows; row++)
                Norms.rmsnorm(
                        state.normed,
                        (long) row * dim,
                        state.residual,
                        (long) (first + row) * dim,
                        weights.outputNorm,
                        dim,
                        configuration.rmsNormEps);
        }
        MatMul.gemm(weights.outputWeight, state.normed, destination, rows);
        Reference.reachabilityFence(state);
    }

    @Override
    public boolean speculationReady() {
        return weights.nextn != null;
    }

    @Override
    public SpeculationResult speculate(
            State state,
            Sampler sampler,
            Constraints constraints,
            int depth,
            GenerationListener listener,
            SpeculationAudit audit) {
        long timeoutNanos = constraints.timeout().isZero() ? 0 : constraints.timeout().toNanos();
        return state.exclusively(
                () -> {
                    int remaining = state.contextCapacity() - state.position();
                    int budget =
                            constraints.maxTokens() == Constraints.UNLIMITED
                                    ? remaining
                                    : Math.min(constraints.maxTokens(), remaining);
                    return BailingMoe3Speculative.generate(
                            this,
                            state,
                            budget,
                            timeoutNanos,
                            constraints.stopTokens(),
                            depth,
                            sampler,
                            listener,
                            audit);
                });
    }

    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int nextnPredictLayers,
            int numberOfHeads,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            float ropeTheta,
            int ropeDimensionCount,
            boolean[] isAttention,
            int kdaHeadDim,
            int convKernel,
            boolean safeKdaGate,
            float kdaGateLowerBound,
            int qLoraRank,
            int kvLoraRank,
            int qkNopeDim,
            int mlaValueHeadDim,
            int feedForwardLength,
            int denseLeadingLayers,
            int expertCount,
            int expertUsedCount,
            int expertGroupCount,
            int expertGroupCountUsed,
            int expertFeedForwardLength,
            int sharedFeedForwardLength,
            boolean normalizeExpertWeights,
            float expertWeightsScale,
            float[] expertSwiGluClamp,
            float[] sharedSwiGluClamp)
            implements ContextConfiguration {
        int kdaInnerSize() {
            return numberOfHeads * kdaHeadDim;
        }

        int mlaQkDim() {
            return qkNopeDim + ropeDimensionCount;
        }

        int mlaCacheDim() {
            return kvLoraRank + ropeDimensionCount;
        }

        int mlaPackedDim() {
            return numberOfHeads * mlaCacheDim();
        }

        int storedLayers() {
            return numberOfLayers + nextnPredictLayers;
        }

        boolean hasMtp() {
            return nextnPredictLayers == 1;
        }

        int mtpLayer() {
            if (!hasMtp()) throw new IllegalStateException("BailingMoe3 model has no MTP layer");
            return numberOfLayers;
        }
    }

    public record KdaWeights(
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> k,
            MemoryView<MemorySegment> v,
            MemoryView<MemorySegment> qConv,
            MemoryView<MemorySegment> kConv,
            MemoryView<MemorySegment> vConv,
            MemoryView<MemorySegment> f,
            MemoryView<MemorySegment> beta,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> dtBias,
            MemoryView<MemorySegment> outputGate,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> output) {}

    public record MlaWeights(
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> qA,
            MemoryView<MemorySegment> qANorm,
            MemoryView<MemorySegment> qB,
            MemoryView<MemorySegment> kvA,
            MemoryView<MemorySegment> kvANorm,
            MemoryView<MemorySegment>[] kB,
            MemoryView<MemorySegment>[] vB,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> output) {}

    public record FfnWeights(
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> up,
            MemoryView<MemorySegment> down) {}

    public record MoeWeights(
            MemoryView<MemorySegment> router,
            MemoryView<MemorySegment> selectionBias,
            MemoryView<MemorySegment>[] expertGate,
            MemoryView<MemorySegment>[] expertUp,
            MemoryView<MemorySegment>[] expertDown,
            MemoryView<MemorySegment> sharedGate,
            MemoryView<MemorySegment> sharedUp,
            MemoryView<MemorySegment> sharedDown) {}

    public record NextNWeights(
            MemoryView<MemorySegment> embeddingNorm,
            MemoryView<MemorySegment> hiddenNorm,
            MemoryView<MemorySegment> inputProjection,
            MemoryView<MemorySegment> outputNorm) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbedding,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> outputWeight,
            MemoryView<MemorySegment>[] attnNorm,
            MemoryView<MemorySegment>[] ffnNorm,
            KdaWeights[] kda,
            MlaWeights[] mla,
            FfnWeights[] ffn,
            MoeWeights[] moe,
            NextNWeights nextn,
            RoPE.Schedule rope,
            int ropeHalf) {}

    public static final class State extends ContextState {
        final MemoryView<MemorySegment> residual, normed, branch, logits, ropeCos, ropeSin;
        final MemoryView<MemorySegment> kdaQ, kdaK, kdaV, kdaQConv, kdaKConv, kdaVConv;
        final MemoryView<MemorySegment> kdaGateProjection, kdaBetaProjection, kdaGate, kdaBeta;
        final MemoryView<MemorySegment> kdaOutput, kdaOutputGate, kdaNormed, kdaDecay;
        final MemoryView<MemorySegment> mlaQA, mlaQAll, mlaKvAll, mlaLatent, mlaQNope;
        final MemoryView<MemorySegment> mlaQLatent, mlaQPacked, mlaAttentionOut;
        final MemoryView<MemorySegment> mlaVHead, mlaProjected, mlaGate;
        final MemoryView<MemorySegment> denseHidden, denseHidden2;
        final MemoryView<MemorySegment> targetHidden, mtpConcat, pendingHidden;
        final MemoryView<MemorySegment> moeRouter, moeSelection, moeGather, moeDown;
        final MemoryView<MemorySegment> moeHidden,
                moeHidden2,
                sharedHidden,
                sharedHidden2,
                sharedOut;
        final MemoryView<MemorySegment>[] attentionCache, qConvState, kConvState, vConvState;
        final MemoryView<MemorySegment>[] recurrentState;
        final int[] moeExpertCounts, moeRowTopE;
        final float[] moeRowTopP, moeGroupScores;
        final boolean[] moeGroupMask;
        final Moe.Routing moeRouting;
        BailingMoe3Speculative.Scratch specScratch;

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
            if (contextCapacity <= 0 || contextCapacity > c.contextLength)
                throw new IllegalArgumentException("invalid context capacity " + contextCapacity);
            int b = batchCapacity, dim = c.embeddingLength, heads = c.numberOfHeads;
            int inner = c.kdaInnerSize(), qk = c.mlaQkDim(), packed = c.mlaPackedDim();
            residual = Views.allocateF32(memoryArena(), b, dim);
            normed = Views.allocateF32(memoryArena(), b, dim);
            branch = Views.allocateF32(memoryArena(), b, dim);
            logits = Views.allocateF32(memoryArena(), 1, c.vocabularySize);
            ropeCos = Views.allocateF32(memoryArena(), b, c.ropeDimensionCount / 2);
            ropeSin = Views.allocateF32(memoryArena(), b, c.ropeDimensionCount / 2);
            kdaQ = Views.allocateF32(memoryArena(), b, inner);
            kdaK = Views.allocateF32(memoryArena(), b, inner);
            kdaV = Views.allocateF32(memoryArena(), b, inner);
            kdaQConv = Views.allocateF32(memoryArena(), b, inner);
            kdaKConv = Views.allocateF32(memoryArena(), b, inner);
            kdaVConv = Views.allocateF32(memoryArena(), b, inner);
            kdaGateProjection = Views.allocateF32(memoryArena(), b, inner);
            kdaBetaProjection = Views.allocateF32(memoryArena(), b, heads);
            kdaGate = Views.allocateF32(memoryArena(), b, inner);
            kdaBeta = Views.allocateF32(memoryArena(), b, heads);
            kdaOutput = Views.allocateF32(memoryArena(), b, inner);
            kdaOutputGate = Views.allocateF32(memoryArena(), b, inner);
            kdaNormed = Views.allocateF32(memoryArena(), b, inner);
            kdaDecay = Views.allocateF32(memoryArena(), heads, c.kdaHeadDim);
            mlaQA = Views.allocateF32(memoryArena(), b, Math.max(1, c.qLoraRank));
            mlaQAll = Views.allocateF32(memoryArena(), b, heads * qk);
            mlaKvAll = Views.allocateF32(memoryArena(), b, c.mlaCacheDim());
            mlaLatent = Views.allocateF32(memoryArena(), b, c.kvLoraRank);
            mlaQNope = Views.allocateF32(memoryArena(), b, c.qkNopeDim);
            mlaQLatent = Views.allocateF32(memoryArena(), b, c.kvLoraRank);
            mlaQPacked = Views.allocateF32(memoryArena(), b, packed);
            mlaAttentionOut = Views.allocateF32(memoryArena(), b, packed);
            mlaVHead = Views.allocateF32(memoryArena(), b, c.mlaValueHeadDim);
            mlaProjected = Views.allocateF32(memoryArena(), b, heads * c.mlaValueHeadDim);
            mlaGate = Views.allocateF32(memoryArena(), b, heads);
            denseHidden = Views.allocateF32(memoryArena(), b, c.feedForwardLength);
            denseHidden2 = Views.allocateF32(memoryArena(), b, c.feedForwardLength);
            targetHidden = c.hasMtp() ? Views.allocateF32(memoryArena(), b, dim) : null;
            mtpConcat = c.hasMtp() ? Views.allocateF32(memoryArena(), b, 2, dim) : null;
            pendingHidden = c.hasMtp() ? Views.allocateF32(memoryArena(), 1, dim) : null;
            moeRouter = Views.allocateF32(memoryArena(), b, c.expertCount);
            moeSelection = Views.allocateF32(memoryArena(), b, c.expertCount);
            moeGather = Views.allocateF32(memoryArena(), b, dim);
            moeDown = Views.allocateF32(memoryArena(), b, dim);
            moeHidden = Views.allocateF32(memoryArena(), b, c.expertFeedForwardLength);
            moeHidden2 = Views.allocateF32(memoryArena(), b, c.expertFeedForwardLength);
            sharedHidden = Views.allocateF32(memoryArena(), b, c.sharedFeedForwardLength);
            sharedHidden2 = Views.allocateF32(memoryArena(), b, c.sharedFeedForwardLength);
            sharedOut = Views.allocateF32(memoryArena(), b, dim);
            attentionCache = new MemoryView[c.storedLayers()];
            qConvState = new MemoryView[c.storedLayers()];
            kConvState = new MemoryView[c.storedLayers()];
            vConvState = new MemoryView[c.storedLayers()];
            recurrentState = new MemoryView[c.storedLayers()];
            for (int layer = 0; layer < c.storedLayers(); layer++) {
                if (c.isAttention[layer]) {
                    attentionCache[layer] =
                            Views.allocateF16(memoryArena(), contextCapacity, c.mlaCacheDim());
                } else {
                    qConvState[layer] = Views.allocateF32(memoryArena(), c.convKernel - 1, inner);
                    kConvState[layer] = Views.allocateF32(memoryArena(), c.convKernel - 1, inner);
                    vConvState[layer] = Views.allocateF32(memoryArena(), c.convKernel - 1, inner);
                    recurrentState[layer] =
                            Views.allocateF32(memoryArena(), heads, c.kdaHeadDim, c.kdaHeadDim);
                }
            }
            moeExpertCounts = new int[c.expertCount];
            moeRowTopE = new int[b * c.expertUsedCount];
            moeRowTopP = new float[b * c.expertUsedCount];
            moeGroupScores = new float[c.expertGroupCount];
            moeGroupMask = new boolean[c.expertGroupCount];
            moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts);
            clearHistory();
        }

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }

        @Override
        protected void clearHistory() {
            clear(qConvState);
            clear(kConvState);
            clear(vConvState);
            clear(recurrentState);
            if (pendingHidden != null)
                Ops.fillInPlace(pendingHidden, 0, Math.toIntExact(pendingHidden.logicalSize()), 0f);
        }

        private static void clear(MemoryView<MemorySegment>[] states) {
            for (MemoryView<MemorySegment> state : states)
                if (state != null)
                    Ops.fillInPlace(state, 0, Math.toIntExact(state.logicalSize()), 0f);
        }
    }

    public static BailingMoe3 loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(channel, gguf, arena);
        }
    }

    public static BailingMoe3 loadModel(FileChannel channel, GGUF gguf, Arena arena)
            throws IOException {
        return loadModel(channel, gguf, arena, null);
    }

    public static BailingMoe3 loadModel(
            FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer) throws IOException {
        String arch = gguf.getString("general.architecture");
        if (!"bailingmoe3".equals(arch))
            throw new IllegalArgumentException("unsupported architecture: " + arch);
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration c = loadConfiguration(gguf, tokenizer, arch);
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new BailingMoe3(
                c, tokenizer, loadWeights(tensors, c, buildRope(gguf, arch, c, tensors)));
    }

    private static Configuration loadConfiguration(GGUF gguf, Tokenizer tokenizer, String arch) {
        int storedLayers = gguf.getValue(int.class, arch + ".block_count");
        int nextnLayers = gguf.getValueOrDefault(int.class, arch + ".nextn_predict_layers", 0);
        require(nextnLayers >= 0 && nextnLayers <= 1, "at most one NextN layer is supported");
        int layers = storedLayers - nextnLayers;
        int heads = gguf.getValue(int.class, arch + ".attention.head_count");
        int[] kvHeads = gguf.getValue(int[].class, arch + ".attention.head_count_kv");
        require(
                kvHeads.length == storedLayers,
                "attention.head_count_kv must have one entry per stored layer");
        require(
                gguf.getValue(int.class, arch + ".expert_gating_func") == SIGMOID_GATING,
                "only sigmoid expert gating is supported");
        boolean[] attention = new boolean[storedLayers];
        for (int i = 0; i < storedLayers; i++) attention[i] = kvHeads[i] > 0;
        if (nextnLayers == 1) attention[layers] = true;
        int expertFeedForwardLength =
                gguf.getValue(int.class, arch + ".expert_feed_forward_length");
        int sharedExpertCount = gguf.getValueOrDefault(int.class, arch + ".expert_shared_count", 1);
        Configuration c =
                new Configuration(
                        gguf.getValue(int.class, arch + ".embedding_length"),
                        layers,
                        nextnLayers,
                        heads,
                        tokenizer.vocabulary().size(),
                        gguf.getValue(int.class, arch + ".context_length"),
                        gguf.getValue(float.class, arch + ".attention.layer_norm_rms_epsilon"),
                        gguf.getValue(float.class, arch + ".rope.freq_base"),
                        gguf.getValue(int.class, arch + ".rope.dimension_count"),
                        attention,
                        gguf.getValue(int.class, arch + ".kda.head_dim"),
                        gguf.getValue(int.class, arch + ".ssm.conv_kernel"),
                        gguf.getValueOrDefault(boolean.class, arch + ".kda.safe_gate", true),
                        gguf.getValue(float.class, arch + ".kda.gate_lower_bound"),
                        gguf.getValueOrDefault(int.class, arch + ".attention.q_lora_rank", 0),
                        gguf.getValue(int.class, arch + ".attention.kv_lora_rank"),
                        gguf.getValue(int.class, arch + ".attention.key_length_mla")
                                - gguf.getValue(int.class, arch + ".rope.dimension_count"),
                        gguf.getValue(int.class, arch + ".attention.value_length_mla"),
                        gguf.getValue(int.class, arch + ".feed_forward_length"),
                        gguf.getValue(int.class, arch + ".leading_dense_block_count"),
                        gguf.getValue(int.class, arch + ".expert_count"),
                        gguf.getValue(int.class, arch + ".expert_used_count"),
                        gguf.getValue(int.class, arch + ".expert_group_count"),
                        gguf.getValue(int.class, arch + ".expert_group_used_count"),
                        expertFeedForwardLength,
                        gguf.getValueOrDefault(
                                int.class,
                                arch + ".expert_shared_feed_forward_length",
                                Math.multiplyExact(
                                        expertFeedForwardLength, Math.max(1, sharedExpertCount))),
                        gguf.getValue(boolean.class, arch + ".expert_weights_norm"),
                        gguf.getValue(float.class, arch + ".expert_weights_scale"),
                        layerFloats(gguf, arch + ".swiglu_clamp_exp", storedLayers),
                        layerFloats(gguf, arch + ".swiglu_clamp_shexp", storedLayers));
        validate(c);
        require(
                gguf.getValueOrDefault(int.class, arch + ".vocab_size", c.vocabularySize)
                        == c.vocabularySize,
                "tokenizer vocabulary does not match the model");
        return c;
    }

    private static void validate(Configuration c) {
        require(
                c.embeddingLength > 0
                        && c.numberOfLayers > 0
                        && c.numberOfHeads > 0
                        && c.vocabularySize > 0
                        && c.contextLength > 0,
                "invalid core dimensions");
        require(
                c.rmsNormEps > 0f
                        && Float.isFinite(c.rmsNormEps)
                        && c.ropeTheta > 0f
                        && Float.isFinite(c.ropeTheta)
                        && c.ropeDimensionCount > 0
                        && (c.ropeDimensionCount & 1) == 0,
                "invalid normalization or RoPE metadata");
        require(
                c.kdaHeadDim > 0
                        && c.convKernel > 1
                        && c.kdaGateLowerBound < 0f
                        && Float.isFinite(c.kdaGateLowerBound),
                "invalid KDA metadata");
        require(
                c.qLoraRank >= 0 && c.kvLoraRank > 0 && c.qkNopeDim > 0 && c.mlaValueHeadDim > 0,
                "invalid or unsupported MLA dimensions");
        require(
                c.feedForwardLength > 0
                        && c.denseLeadingLayers >= 0
                        && c.denseLeadingLayers <= c.numberOfLayers,
                "invalid dense FFN dimensions");
        require(
                c.expertCount > 0
                        && c.expertUsedCount > 0
                        && c.expertUsedCount <= c.expertCount
                        && c.expertGroupCount > 0
                        && c.expertCount % c.expertGroupCount == 0
                        && c.expertGroupCountUsed > 0
                        && c.expertGroupCountUsed <= c.expertGroupCount
                        && c.expertUsedCount
                                <= c.expertGroupCountUsed * (c.expertCount / c.expertGroupCount)
                        && c.expertFeedForwardLength > 0
                        && c.sharedFeedForwardLength > 0
                        && c.expertWeightsScale > 0f
                        && Float.isFinite(c.expertWeightsScale),
                "invalid MoE dimensions");
    }

    static float[] layerFloats(GGUF gguf, String key, int layers) {
        float[] values = new float[layers];
        if (!gguf.containsKey(key)) return values;
        if (gguf.getType(key) == MetadataValueType.ARRAY) {
            float[] stored = gguf.getValue(float[].class, key);
            require(stored.length == layers, key + " must have one entry per stored layer");
            return stored;
        }
        float value = gguf.getValue(float.class, key);
        Arrays.fill(values, value);
        return values;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("BailingMoe3: " + message);
    }

    static RoPE.Schedule buildRope(
            GGUF gguf,
            String arch,
            Configuration c,
            Map<String, MemoryView<MemorySegment>> tensors) {
        String type = gguf.getValueOrDefault(String.class, arch + ".rope.scaling.type", "");
        RoPE.Schedule rope;
        if (type.isEmpty() || type.equals("none")) {
            rope = RoPE.plain(c.ropeDimensionCount, c.ropeTheta);
        } else if (type.equals("yarn")) {
            float factor = gguf.getValue(float.class, arch + ".rope.scaling.factor");
            int original = gguf.getValue(int.class, arch + ".rope.scaling.original_context_length");
            float fast =
                    gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_fast", 32f);
            float slow =
                    gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_slow", 1f);
            float logMultiplier =
                    gguf.getValueOrDefault(
                            float.class, arch + ".rope.scaling.yarn_log_multiplier", 0f);
            float ln = (float) Math.log(factor);
            float mscale = factor <= 1f ? 1f : 1f + 0.1f * logMultiplier * ln;
            rope =
                    RoPE.yarn(
                            c.ropeDimensionCount,
                            c.ropeTheta,
                            factor,
                            original,
                            fast,
                            slow,
                            1f,
                            1f / mscale);
        } else {
            throw new IllegalArgumentException(
                    "BailingMoe3: unsupported rope.scaling.type '" + type + "'");
        }
        Optional<float[]> factors = ModelLoader.ropeFreqFactors(tensors);
        if (factors.isPresent()) {
            require(
                    type.isEmpty() || type.equals("none"),
                    "rope_freqs.weight cannot be combined with " + type + " scaling");
            require(
                    factors.get().length == c.ropeDimensionCount / 2,
                    "rope_freqs.weight has the wrong length");
            rope = RoPE.withFreqFactors(c.ropeDimensionCount, c.ropeTheta, factors.get());
        }
        return rope;
    }

    @SuppressWarnings("unchecked")
    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors, Configuration c, RoPE.Schedule rope) {
        int n = c.storedLayers();
        MemoryView<MemorySegment>[] attnNorm = new MemoryView[n], ffnNorm = new MemoryView[n];
        KdaWeights[] kda = new KdaWeights[n];
        MlaWeights[] mla = new MlaWeights[n];
        FfnWeights[] ffn = new FfnWeights[n];
        MoeWeights[] moe = new MoeWeights[n];
        for (int layer = 0; layer < n; layer++) {
            String p = "blk." + layer + ".";
            attnNorm[layer] = ModelLoader.requireF32(tensors, p + "attn_norm.weight");
            ffnNorm[layer] = ModelLoader.requireF32(tensors, p + "ffn_norm.weight");
            if (c.isAttention[layer]) {
                MemoryView<MemorySegment> q =
                        c.qLoraRank == 0 ? ModelLoader.require(tensors, p + "attn_q.weight") : null;
                mla[layer] =
                        new MlaWeights(
                                q,
                                c.qLoraRank > 0
                                        ? ModelLoader.require(tensors, p + "attn_q_a.weight")
                                        : null,
                                c.qLoraRank > 0
                                        ? ModelLoader.requireF32(
                                                tensors, p + "attn_q_a_norm.weight")
                                        : null,
                                c.qLoraRank > 0
                                        ? ModelLoader.require(tensors, p + "attn_q_b.weight")
                                        : null,
                                ModelLoader.require(tensors, p + "attn_kv_a_mqa.weight"),
                                ModelLoader.requireF32(tensors, p + "attn_kv_a_norm.weight"),
                                Views.sliceLeadingAxis(
                                        ModelLoader.require(tensors, p + "attn_k_b.weight")),
                                Views.sliceLeadingAxis(
                                        ModelLoader.require(tensors, p + "attn_v_b.weight")),
                                ModelLoader.require(tensors, p + "attn_gate.weight"),
                                ModelLoader.require(tensors, p + "attn_output.weight"));
            } else {
                kda[layer] =
                        new KdaWeights(
                                ModelLoader.require(tensors, p + "attn_q.weight"),
                                ModelLoader.require(tensors, p + "attn_k.weight"),
                                ModelLoader.require(tensors, p + "attn_v.weight"),
                                ModelLoader.requireF32(tensors, p + "ssm_conv1d_q.weight"),
                                ModelLoader.requireF32(tensors, p + "ssm_conv1d_k.weight"),
                                ModelLoader.requireF32(tensors, p + "ssm_conv1d_v.weight"),
                                ModelLoader.require(tensors, p + "ssm_f_a.weight"),
                                ModelLoader.require(tensors, p + "ssm_beta.weight"),
                                ModelLoader.requireF32(tensors, p + "ssm_a"),
                                ModelLoader.requireF32(tensors, p + "ssm_dt.bias"),
                                ModelLoader.require(tensors, p + "ssm_g_a.weight"),
                                ModelLoader.requireF32(tensors, p + "ssm_norm.weight"),
                                ModelLoader.require(tensors, p + "attn_output.weight"));
            }
            if (layer < c.denseLeadingLayers) {
                ffn[layer] =
                        new FfnWeights(
                                ModelLoader.require(tensors, p + "ffn_gate.weight"),
                                ModelLoader.require(tensors, p + "ffn_up.weight"),
                                ModelLoader.require(tensors, p + "ffn_down.weight"));
            } else {
                moe[layer] =
                        new MoeWeights(
                                ModelLoader.require(tensors, p + "ffn_gate_inp.weight"),
                                ModelLoader.requireF32(tensors, p + "exp_probs_b.bias"),
                                Views.sliceLeadingAxis(
                                        ModelLoader.require(tensors, p + "ffn_gate_exps.weight")),
                                Views.sliceLeadingAxis(
                                        ModelLoader.require(tensors, p + "ffn_up_exps.weight")),
                                Views.sliceLeadingAxis(
                                        ModelLoader.require(tensors, p + "ffn_down_exps.weight")),
                                ModelLoader.require(tensors, p + "ffn_gate_shexp.weight"),
                                ModelLoader.require(tensors, p + "ffn_up_shexp.weight"),
                                ModelLoader.require(tensors, p + "ffn_down_shexp.weight"));
            }
        }
        MemoryView<MemorySegment> embedding = ModelLoader.require(tensors, "token_embd.weight");
        NextNWeights nextn =
                c.hasMtp()
                        ? new NextNWeights(
                                ModelLoader.requireF32(
                                        tensors, "blk." + c.mtpLayer() + ".nextn.enorm.weight"),
                                ModelLoader.requireF32(
                                        tensors, "blk." + c.mtpLayer() + ".nextn.hnorm.weight"),
                                ModelLoader.require(
                                        tensors, "blk." + c.mtpLayer() + ".nextn.eh_proj.weight"),
                                ModelLoader.requireF32(
                                        tensors,
                                        "blk." + c.mtpLayer() + ".layer_output_norm.weight"))
                        : null;
        return new Weights(
                embedding,
                ModelLoader.requireF32(tensors, "output_norm.weight"),
                ModelLoader.find(tensors, "output.weight").orElse(embedding),
                attnNorm,
                ffnNorm,
                kda,
                mla,
                ffn,
                moe,
                nextn,
                rope,
                c.ropeDimensionCount / 2);
    }
}
