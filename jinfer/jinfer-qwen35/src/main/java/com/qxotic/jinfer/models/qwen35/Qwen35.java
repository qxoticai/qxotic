package com.qxotic.jinfer.models.qwen35;

import com.qxotic.format.gguf.GGUF;
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
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.GatedDeltaNet;
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
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * MemoryView port of the hybrid Qwen3.5 gated-delta/full-attention decoder, optionally with the
 * Qwen3-VL vision tower attached ({@link #withMedia}).
 */
public final class Qwen35
        implements LanguageModel<Qwen35.Configuration, Qwen35.Weights, Qwen35.State>,
                SpeculativeDecoding<Qwen35.State>,
                Multimodal {
    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;
    private final MediaProjector<Media.Image> vision;

    Qwen35(Configuration configuration, Tokenizer tokenizer, Weights weights) {
        this(configuration, tokenizer, weights, null);
    }

    private Qwen35(
            Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            MediaProjector<Media.Image> vision) {
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
    public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
        if (modality == Media.Image.class && vision != null)
            return Optional.of((MediaProjector<R>) vision);
        return Optional.empty();
    }

    /**
     * Returns a model sharing this backbone's weights with a validated vision sidecar attached.
     *
     * <p>ponytail: the sidecar borrows {@code arena}, so callers close it once after the model -
     * never before, and never the sidecar separately.
     */
    public Qwen35 withMedia(Path mmprojPath, Arena arena) throws IOException {
        Objects.requireNonNull(mmprojPath, "mmprojPath");
        Objects.requireNonNull(arena, "arena");
        try (FileChannel channel = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, mmprojPath.toString());
            // ponytail: this is the only check the backbone owns; the tower's own geometry
            // (projector type included) is validated inside Qwen35Vision.loadModel.
            int projected = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", 0);
            if (projected != configuration.embeddingLength)
                throw new IllegalArgumentException(
                        "'"
                                + mmprojPath.getFileName()
                                + "' projector width "
                                + projected
                                + " does not match model width "
                                + configuration.embeddingLength);
            Qwen35Vision tower =
                    Qwen35Vision.loadModel(
                            mmprojPath, gguf, ModelLoader.loadTensors(channel, gguf, arena));
            return new Qwen35(configuration, tokenizer, weights, tower);
        }
    }

    @Override
    public Optional<CheckpointCodec<State>> checkpointCodec() {
        return Optional.of(new Qwen35CheckpointCodec(configuration));
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
        if (rows <= 0) throw new IllegalArgumentException("Qwen3.5 token batch must not be empty");
        if (rows > state.batchCapacity())
            throw new IllegalArgumentException(
                    "batch " + rows + " exceeds batchCapacity " + state.batchCapacity());
        int start = state.position();
        if (start + rows > state.contextCapacity())
            throw new IllegalArgumentException(
                    "ingest of "
                            + rows
                            + " at position "
                            + start
                            + " exceeds contextCapacity "
                            + state.contextCapacity());
        int[] tokens =
                switch (batch.input()) {
                    case Batch.Input.Tokens t -> t.ids();
                    case Batch.Input.Sequences ignored ->
                            throw new UnsupportedOperationException(
                                    "Qwen3.5 does not support packed sequences");
                    case Batch.Input.Embeddings e -> {
                        if (vision == null)
                            throw new UnsupportedOperationException(
                                    "no media encoder loaded (attach --with media=<mmproj.gguf>)");
                        if (e.rows().shape().flatAt(1) != configuration.embeddingLength)
                            throw new IllegalArgumentException(
                                    "embedding width "
                                            + e.rows().shape().flatAt(1)
                                            + " != model width "
                                            + configuration.embeddingLength);
                        MemoryView<MemorySegment> rowsView =
                                Views.castToSegmentBacked(e.rows(), "embedding rows");
                        if (rows == 1)
                            Parallel.onDecodePool(
                                    () -> {
                                        forwardMedia(state, rowsView, start, 1);
                                        return null;
                                    });
                        else forwardMedia(state, rowsView, start, rows);
                        state.advance(batch);
                        yield null;
                    }
                };
        if (tokens == null) return; // embedding batch already forwarded
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
        state.advance(batch);
    }

    /**
     * Projected media rows: copy into the residual stream, then the decoder blocks (prefill-style
     * batch; the draft-head synchronize path is token-only and skipped).
     */
    private void forwardMedia(
            State state, MemoryView<MemorySegment> rowsView, int startPos, int rows) {
        Configuration c = configuration;
        if (weights.rope != null)
            RoPE.fill(state.ropeCos, state.ropeSin, startPos, rows, weights.ropeHalf, weights.rope);
        Convert.copyF32(rowsView, 0, state.residual, 0, (long) rows * c.embeddingLength);
        for (int layer = 0; layer < c.numberOfLayers; layer++)
            decoderBlock(state, layer, startPos, rows);
    }

    private void forward(State state, int[] tokens, int startPos, int rows) {
        Configuration c = configuration;
        if (weights.rope != null)
            RoPE.fill(state.ropeCos, state.ropeSin, startPos, rows, weights.ropeHalf, weights.rope);
        Views.checkAlive(weights.tokenEmbedding, "tokenEmbedding");
        Convert.gatherToF32(
                weights.tokenEmbedding, tokens, 0, rows, state.residual, 0, c.embeddingLength);
        for (int layer = 0; layer < c.numberOfLayers; layer++)
            decoderBlock(state, layer, startPos, rows);
        if (c.hasMtp()) {
            Norms.rmsnormRows(
                    state.targetHidden,
                    state.residual,
                    weights.outputNorm,
                    rows,
                    c.embeddingLength,
                    c.rmsNormEps);
            synchronizeMtp(state, tokens, startPos, rows);
        }
    }

    private void decoderBlock(State state, int layer, int startPos, int rows) {
        Configuration c = configuration;
        Norms.rmsnormRows(
                state.normed,
                state.residual,
                weights.attnNorm[layer],
                rows,
                c.embeddingLength,
                c.rmsNormEps);
        if (c.isFullAttention[layer]) attention(state, layer, startPos, rows);
        else delta(state, layer, rows);
        Ops.addInPlace(state.residual, 0, state.branch, 0, rows * c.embeddingLength);
        Norms.rmsnormRows(
                state.normed,
                state.residual,
                weights.postAttentionNorm[layer],
                rows,
                c.embeddingLength,
                c.rmsNormEps);
        if (c.isMoE()) moe(state, layer, rows);
        else denseFfn(state, layer, rows);
        Ops.addInPlace(state.residual, 0, state.branch, 0, rows * c.embeddingLength);
        if (Trace.ENABLED) Trace.sum("l_out-" + layer, state.residual, rows * c.embeddingLength);
    }

    /** Keeps the embedded MTP block's KV prefix aligned with every committed target token. */
    private void synchronizeMtp(State state, int[] tokens, int startPos, int rows) {
        Configuration c = configuration;
        NextNWeights nextn = weights.nextn;
        int dim = c.embeddingLength;
        for (int row = 0; row < rows; row++) {
            long concat = (long) row * 2 * dim;
            Convert.copyToF32(
                    nextn.tokenEmbedding,
                    (long) tokens[row] * dim,
                    state.normed,
                    (long) row * dim,
                    dim);
            Norms.rmsnorm(
                    state.mtpConcat,
                    concat,
                    state.normed,
                    (long) row * dim,
                    nextn.embeddingNorm,
                    dim,
                    c.rmsNormEps);
            MemoryView<MemorySegment> hidden = row == 0 ? state.pendingHidden : state.targetHidden;
            long hiddenOffset = row == 0 ? 0 : (long) (row - 1) * dim;
            Norms.rmsnorm(
                    state.mtpConcat,
                    concat + dim,
                    hidden,
                    hiddenOffset,
                    nextn.hiddenNorm,
                    dim,
                    c.rmsNormEps);
        }
        Convert.copyF32(state.targetHidden, (long) (rows - 1) * dim, state.pendingHidden, 0, dim);
        MatMul.gemm(nextn.inputProjection, state.mtpConcat, state.residual, rows);
        decoderBlock(state, c.mtpLayer(), startPos, rows);
    }

    /** Fills {@code candidates[1..depth]} from the exact target seed in {@code candidates[0]}. */
    void draft(State state, int depth, int[] candidates) {
        Parallel.onDecodePool(
                () -> {
                    MemoryView<MemorySegment> hidden = state.pendingHidden;
                    int token = candidates[0];
                    int position = state.position();
                    for (int i = 1; i <= depth; i++) {
                        draftOne(state, token, hidden, position + i - 1);
                        token = Ops.argmax(state.logits, 0, configuration.vocabularySize);
                        candidates[i] = token;
                        hidden = state.normed;
                    }
                    return null;
                });
    }

    private void draftOne(State state, int token, MemoryView<MemorySegment> hidden, int position) {
        Configuration c = configuration;
        NextNWeights nextn = weights.nextn;
        int dim = c.embeddingLength;
        if (weights.rope != null)
            RoPE.fill(state.ropeCos, state.ropeSin, position, 1, weights.ropeHalf, weights.rope);
        Norms.rmsnorm(state.mtpConcat, dim, hidden, 0, nextn.hiddenNorm, dim, c.rmsNormEps);
        Convert.copyToF32(nextn.tokenEmbedding, (long) token * dim, state.normed, 0, dim);
        Norms.rmsnorm(state.mtpConcat, 0, state.normed, 0, nextn.embeddingNorm, dim, c.rmsNormEps);
        MatMul.gemv(nextn.inputProjection, state.mtpConcat, state.residual);
        decoderBlock(state, c.mtpLayer(), position, 1);
        Norms.rmsnorm(state.normed, 0, state.residual, 0, nextn.outputNorm, dim, c.rmsNormEps);
        MatMul.gemv(nextn.outputWeight, state.normed, state.logits);
    }

    private void attention(State s, int layer, int startPos, int rows) {
        Configuration c = configuration;
        int dim = c.embeddingLength, qDim = c.queryDim(), kvDim = c.kvDim();
        MatMul.gemm(weights.attnQ[layer], s.normed, s.packedQ, rows);
        MatMul.gemm(weights.attnK[layer], s.normed, s.k, rows);
        MatMul.gemm(weights.attnV[layer], s.normed, s.v, rows);
        GatedDeltaNet.unpackAttentionQGate(
                s.packedQ, s.q, s.attentionGate, rows, c.numberOfHeads, c.headSize);
        Parallel.forRows(
                rows,
                row -> {
                    for (int h = 0; h < c.numberOfHeads; h++) {
                        long off = (long) row * qDim + (long) h * c.headSize;
                        Norms.rmsnorm(
                                s.q,
                                off,
                                s.q,
                                off,
                                weights.attnQNorm[layer],
                                c.headSize,
                                c.rmsNormEps);
                        if (weights.rope != null)
                            RoPE.applyInterleaved(
                                    s.q, off, row, s.ropeCos, s.ropeSin, weights.ropeHalf);
                    }
                    for (int h = 0; h < c.numberOfKeyValueHeads; h++) {
                        long off = (long) row * kvDim + (long) h * c.headSize;
                        Norms.rmsnorm(
                                s.k,
                                off,
                                s.k,
                                off,
                                weights.attnKNorm[layer],
                                c.headSize,
                                c.rmsNormEps);
                        if (weights.rope != null)
                            RoPE.applyInterleaved(
                                    s.k, off, row, s.ropeCos, s.ropeSin, weights.ropeHalf);
                    }
                });
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
                qDim,
                c.numberOfHeads / c.numberOfKeyValueHeads);
        GatedDeltaNet.sigmoidMultiply(s.attentionOut, s.attentionGate, rows * qDim);
        MatMul.gemm(weights.attnOutput[layer], s.attentionOut, s.branch, rows);
    }

    private void delta(State s, int layer, int rows) {
        Configuration c = configuration;
        int dim = c.embeddingLength, inner = c.ssmInnerSize, heads = c.ssmTimeStepRank;
        int headDim = c.headVDim(), channels = c.convChannels();
        MatMul.gemm(weights.attnQkv[layer], s.normed, s.ssmQkv, rows);
        MatMul.gemm(weights.attnGate[layer], s.normed, s.z, rows);
        MatMul.gemm(weights.ssmAlpha[layer], s.normed, s.alpha, rows);
        MatMul.gemm(weights.ssmBeta[layer], s.normed, s.betaProjection, rows);
        Convolutions.causalDepthwiseSilu(
                s.ssmQkv,
                weights.ssmConv1d[layer],
                s.convState[layer],
                s.convOut,
                rows,
                channels,
                c.ssmConvKernel);
        GatedDeltaNet.prepareQkv(
                s.convOut,
                s.qGroup,
                s.kGroup,
                s.ssmQ,
                s.ssmK,
                s.ssmV,
                rows,
                channels,
                c.ssmGroupCount,
                heads,
                headDim,
                c.rmsNormEps);
        GatedDeltaNet.gates(
                s.alpha,
                s.betaProjection,
                weights.ssmDtBias[layer],
                weights.ssmA[layer],
                s.ssmGate,
                s.ssmBeta,
                rows,
                heads);
        GatedDeltaNet.scan(
                s.ssmQ,
                s.ssmK,
                s.ssmV,
                s.ssmGate,
                s.ssmBeta,
                s.recurrentState[layer],
                s.ssmOutput,
                s.ssmSk,
                s.ssmDelta,
                rows,
                heads,
                headDim);
        GatedDeltaNet.postNorm(
                s.ssmOutput,
                s.z,
                weights.ssmNorm[layer],
                s.ssmTmp,
                rows,
                heads,
                headDim,
                c.rmsNormEps);
        MatMul.gemm(weights.ssmOut[layer], s.ssmTmp, s.branch, rows);
    }

    private void denseFfn(State s, int layer, int rows) {
        Configuration c = configuration;
        MatMul.gemm(weights.ffnGate[layer], s.normed, s.hidden, rows);
        MatMul.gemm(weights.ffnUp[layer], s.normed, s.hidden2, rows);
        Activations.siluMultiply(s.hidden, 0, s.hidden2, 0, rows * c.hiddenDim);
        MatMul.gemm(weights.ffnDown[layer], s.hidden, s.branch, rows);
    }

    private void moe(State s, int layer, int rows) {
        Configuration c = configuration;
        int dim = c.embeddingLength, experts = c.expertCount;
        int topK = Math.min(c.expertUsedCount, experts), expertFfn = c.expertFeedForwardLength;
        MatMul.gemm(weights.moeRouter[layer], s.normed, s.moeRouter, rows);
        for (int row = 0; row < rows; row++)
            Ops.softmaxInPlace(s.moeRouter, (long) row * experts, experts);
        Moe.selectTopK(
                s.moeRouter, rows, experts, topK, s.moeRowTopE, s.moeRowTopP, s.moeExpertCounts);
        for (int row = 0; row < rows; row++) {
            float sum = 0f;
            for (int k = 0; k < topK; k++) sum += s.moeRowTopP[row * topK + k];
            for (int k = 0; k < topK; k++) s.moeRowTopP[row * topK + k] /= sum;
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
                    MatMul.gemm(weights.moeExpertGate[layer][expert], gather, s.moeHidden, n);
                    MatMul.gemm(weights.moeExpertUp[layer][expert], gather, s.moeHidden2, n);
                    Activations.siluMultiply(s.moeHidden, 0, s.moeHidden2, 0, n * expertFfn);
                    MatMul.gemm(weights.moeExpertDown[layer][expert], s.moeHidden, out, n);
                });
        if (c.expertSharedFeedForwardLength > 0 && weights.moeSharedGate[layer] != null) {
            int shared = c.expertSharedFeedForwardLength;
            MatMul.gemm(weights.moeSharedGate[layer], s.normed, s.sharedGate, rows);
            MatMul.gemm(weights.moeSharedUp[layer], s.normed, s.sharedUp, rows);
            Activations.siluMultiply(s.sharedGate, 0, s.sharedUp, 0, rows * shared);
            MatMul.gemm(weights.moeSharedDown[layer], s.sharedGate, s.sharedOut, rows);
            if (weights.moeSharedInputGate[layer] == null) {
                Ops.addInPlace(s.branch, 0, s.sharedOut, 0, rows * dim);
            } else {
                MatMul.gemm(weights.moeSharedInputGate[layer], s.normed, s.sharedScale, rows);
                // sigmoid scalars read on the OWNING thread (checked access; a confined arena
                // would reject reads from forRows workers), the saxpy rows stay parallel
                float[] scales = new float[rows];
                for (int row = 0; row < rows; row++)
                    scales[row] =
                            Activations.sigmoid(Views.getFloat(s.sharedScale, row, "sharedScale"));
                Parallel.forRows(
                        rows,
                        row ->
                                Ops.saxpyInPlace(
                                        s.branch,
                                        (long) row * dim,
                                        s.sharedOut,
                                        (long) row * dim,
                                        dim,
                                        scales[row]));
            }
        }
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
                    if (configuration.hasMtp()) {
                        Convert.copyF32(state.targetHidden, (long) row * dim, state.normed, 0, dim);
                    } else {
                        Norms.rmsnorm(
                                state.normed,
                                0,
                                state.residual,
                                (long) row * dim,
                                weights.outputNorm,
                                dim,
                                configuration.rmsNormEps);
                    }
                    MatMul.gemv(weights.outputWeight, state.normed, state.logits);
                    return state.logits;
                });
    }

    void logitsAll(State state, MemoryView<MemorySegment> destination) {
        int dim = configuration.embeddingLength;
        int rows = state.outputCount();
        int first = state.lastBatchSize() - rows;
        Parallel.onDecodePool(
                () -> {
                    if (configuration.hasMtp()) {
                        Convert.copyF32(
                                state.targetHidden,
                                (long) first * dim,
                                state.normed,
                                0,
                                rows * dim);
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
                    return null;
                });
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
        int capacity = state.contextCapacity();
        int budget =
                constraints.maxTokens() == Constraints.UNLIMITED
                        ? capacity - state.position()
                        : Math.min(constraints.maxTokens(), capacity - state.position());
        long timeoutNanos = constraints.timeout().isZero() ? 0 : constraints.timeout().toNanos();
        return state.exclusively(
                () ->
                        Qwen35Speculative.generate(
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

    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int nextnPredictLayers,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int headSize,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            float ropeTheta,
            int ropeDimensionCount,
            int hiddenDim,
            boolean[] isFullAttention,
            int ssmInnerSize,
            int ssmGroupCount,
            int ssmTimeStepRank,
            int ssmStateSize,
            int ssmConvKernel,
            int expertCount,
            int expertUsedCount,
            int expertFeedForwardLength,
            int expertSharedFeedForwardLength)
            implements ContextConfiguration {
        int queryDim() {
            return numberOfHeads * headSize;
        }

        int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        int headVDim() {
            return ssmInnerSize / ssmTimeStepRank;
        }

        int convChannels() {
            return ssmInnerSize + 2 * ssmGroupCount * ssmStateSize;
        }

        boolean isMoE() {
            return expertCount > 0;
        }

        int storedLayers() {
            return numberOfLayers + nextnPredictLayers;
        }

        boolean hasMtp() {
            return nextnPredictLayers == 1;
        }

        int mtpLayer() {
            if (!hasMtp()) throw new IllegalStateException("Qwen3.5 model has no MTP layer");
            return numberOfLayers;
        }
    }

    public record NextNWeights(
            MemoryView<MemorySegment> tokenEmbedding,
            MemoryView<MemorySegment> embeddingNorm,
            MemoryView<MemorySegment> hiddenNorm,
            MemoryView<MemorySegment> inputProjection,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> outputWeight) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbedding,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> outputWeight,
            MemoryView<MemorySegment>[] attnNorm,
            MemoryView<MemorySegment>[] postAttentionNorm,
            MemoryView<MemorySegment>[] attnQ,
            MemoryView<MemorySegment>[] attnK,
            MemoryView<MemorySegment>[] attnV,
            MemoryView<MemorySegment>[] attnOutput,
            MemoryView<MemorySegment>[] attnQNorm,
            MemoryView<MemorySegment>[] attnKNorm,
            MemoryView<MemorySegment>[] attnQkv,
            MemoryView<MemorySegment>[] attnGate,
            MemoryView<MemorySegment>[] ssmAlpha,
            MemoryView<MemorySegment>[] ssmBeta,
            MemoryView<MemorySegment>[] ssmOut,
            MemoryView<MemorySegment>[] ssmConv1d,
            MemoryView<MemorySegment>[] ssmA,
            MemoryView<MemorySegment>[] ssmDtBias,
            MemoryView<MemorySegment>[] ssmNorm,
            MemoryView<MemorySegment>[] ffnGate,
            MemoryView<MemorySegment>[] ffnUp,
            MemoryView<MemorySegment>[] ffnDown,
            MemoryView<MemorySegment>[] moeRouter,
            MemoryView<MemorySegment>[][] moeExpertGate,
            MemoryView<MemorySegment>[][] moeExpertUp,
            MemoryView<MemorySegment>[][] moeExpertDown,
            MemoryView<MemorySegment>[] moeSharedGate,
            MemoryView<MemorySegment>[] moeSharedUp,
            MemoryView<MemorySegment>[] moeSharedDown,
            MemoryView<MemorySegment>[] moeSharedInputGate,
            RoPE.Schedule rope,
            int ropeHalf,
            NextNWeights nextn) {}

    public static final class State extends ContextState {
        final MemoryView<MemorySegment> residual, normed, branch, logits;
        final MemoryView<MemorySegment> targetHidden, mtpConcat, pendingHidden;
        final MemoryView<MemorySegment> packedQ,
                q,
                k,
                v,
                attentionGate,
                attentionOut,
                ropeCos,
                ropeSin;
        final MemoryView<MemorySegment> ssmQkv, convOut, qGroup, kGroup, ssmQ, ssmK, ssmV;
        final MemoryView<MemorySegment> z, alpha, betaProjection, ssmGate, ssmBeta;
        final MemoryView<MemorySegment> ssmOutput, ssmTmp, ssmSk, ssmDelta;
        final MemoryView<MemorySegment> hidden, hidden2;
        final MemoryView<MemorySegment>[] keyCache, valueCache, convState, recurrentState;
        final MemoryView<MemorySegment> moeRouter, moeGather, moeDown;
        // Per-expert gate/up at EXACTLY expertFeedForwardLength wide: hidden/hidden2 are sized to
        // max(dense, expert) FFN width, and the silu-multiply between gate/up and down addresses
        // rows packed at the expert width.
        final MemoryView<MemorySegment> moeHidden, moeHidden2;
        final MemoryView<MemorySegment> sharedGate, sharedUp, sharedOut, sharedScale;
        final int[] moeExpertCounts, moeRowTopE;
        final float[] moeRowTopP;
        final Moe.Routing moeRouting;
        Qwen35Speculative.Scratch specScratch;

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
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " outside [1,"
                                + c.contextLength
                                + "]");
            if (batchCapacity <= 0)
                throw new IllegalArgumentException("batchCapacity " + batchCapacity);
            int b = batchCapacity, dim = c.embeddingLength, qd = c.queryDim(), kvd = c.kvDim();
            int hd = c.headVDim(), heads = c.ssmTimeStepRank, channels = c.convChannels();
            int maxHidden = Math.max(c.hiddenDim, c.expertFeedForwardLength);
            residual = Views.allocateF32(memoryArena(), b, dim);
            normed = Views.allocateF32(memoryArena(), b, dim);
            branch = Views.allocateF32(memoryArena(), b, dim);
            logits = Views.allocateF32(memoryArena(), 1, c.vocabularySize);
            targetHidden = c.hasMtp() ? Views.allocateF32(memoryArena(), b, dim) : null;
            mtpConcat = c.hasMtp() ? Views.allocateF32(memoryArena(), b, 2L * dim) : null;
            pendingHidden = c.hasMtp() ? Views.allocateF32(memoryArena(), dim) : null;
            packedQ = Views.allocateF32(memoryArena(), b, 2 * qd);
            q = Views.allocateF32(memoryArena(), b, qd);
            k = Views.allocateF32(memoryArena(), b, kvd);
            v = Views.allocateF32(memoryArena(), b, kvd);
            attentionGate = Views.allocateF32(memoryArena(), b, qd);
            attentionOut = Views.allocateF32(memoryArena(), b, qd);
            int ropeLanes =
                    Math.max(1, Math.max(0, Math.min(c.ropeDimensionCount, c.headSize) & ~1) / 2);
            ropeCos = Views.allocateF32(memoryArena(), b, ropeLanes);
            ropeSin = Views.allocateF32(memoryArena(), b, ropeLanes);
            ssmQkv = Views.allocateF32(memoryArena(), b, channels);
            convOut = Views.allocateF32(memoryArena(), b, channels);
            qGroup = Views.allocateF32(memoryArena(), b, c.ssmGroupCount, hd);
            kGroup = Views.allocateF32(memoryArena(), b, c.ssmGroupCount, hd);
            ssmQ = Views.allocateF32(memoryArena(), b, heads, hd);
            ssmK = Views.allocateF32(memoryArena(), b, heads, hd);
            ssmV = Views.allocateF32(memoryArena(), b, heads, hd);
            z = Views.allocateF32(memoryArena(), b, c.ssmInnerSize);
            alpha = Views.allocateF32(memoryArena(), b, heads);
            betaProjection = Views.allocateF32(memoryArena(), b, heads);
            ssmGate = Views.allocateF32(memoryArena(), b, heads);
            ssmBeta = Views.allocateF32(memoryArena(), b, heads);
            ssmOutput = Views.allocateF32(memoryArena(), b, heads, hd);
            ssmTmp = Views.allocateF32(memoryArena(), b, c.ssmInnerSize);
            ssmSk = Views.allocateF32(memoryArena(), heads, hd);
            ssmDelta = Views.allocateF32(memoryArena(), heads, hd);
            hidden = Views.allocateF32(memoryArena(), b, Math.max(1, maxHidden));
            hidden2 = Views.allocateF32(memoryArena(), b, Math.max(1, maxHidden));
            keyCache = new MemoryView[c.storedLayers()];
            valueCache = new MemoryView[c.storedLayers()];
            convState = new MemoryView[c.storedLayers()];
            recurrentState = new MemoryView[c.storedLayers()];
            for (int l = 0; l < c.storedLayers(); l++) {
                if (c.isFullAttention[l]) {
                    keyCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvd);
                    valueCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvd);
                } else {
                    convState[l] = Views.allocateF32(memoryArena(), c.ssmConvKernel - 1, channels);
                    recurrentState[l] = Views.allocateF32(memoryArena(), heads, hd, hd);
                }
            }
            if (c.isMoE()) {
                int topK = Math.min(c.expertUsedCount, c.expertCount);
                moeRouter = Views.allocateF32(memoryArena(), b, c.expertCount);
                moeGather = Views.allocateF32(memoryArena(), b, dim);
                moeDown = Views.allocateF32(memoryArena(), b, dim);
                moeHidden = Views.allocateF32(memoryArena(), b, c.expertFeedForwardLength);
                moeHidden2 = Views.allocateF32(memoryArena(), b, c.expertFeedForwardLength);
                int shared = Math.max(1, c.expertSharedFeedForwardLength);
                sharedGate = Views.allocateF32(memoryArena(), b, shared);
                sharedUp = Views.allocateF32(memoryArena(), b, shared);
                sharedOut = Views.allocateF32(memoryArena(), b, dim);
                sharedScale = Views.allocateF32(memoryArena(), b, 1);
                moeExpertCounts = new int[c.expertCount];
                moeRowTopE = new int[b * topK];
                moeRowTopP = new float[b * topK];
                moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts);
            } else {
                moeRouter = moeGather = moeDown = null;
                moeHidden = moeHidden2 = null;
                sharedGate = sharedUp = sharedOut = sharedScale = null;
                moeExpertCounts = moeRowTopE = null;
                moeRowTopP = null;
                moeRouting = null;
            }
        }

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }

        @Override
        protected void clearHistory() {
            for (MemoryView<MemorySegment> state : convState)
                if (state != null)
                    Ops.fillInPlace(state, 0, Math.toIntExact(state.logicalSize()), 0f);
            for (MemoryView<MemorySegment> state : recurrentState)
                if (state != null)
                    Ops.fillInPlace(state, 0, Math.toIntExact(state.logicalSize()), 0f);
            if (pendingHidden != null)
                Ops.fillInPlace(pendingHidden, 0, Math.toIntExact(pendingHidden.logicalSize()), 0f);
        }
    }

    public static Qwen35 loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(channel, gguf, arena);
        }
    }

    public static Qwen35 loadModel(FileChannel channel, GGUF gguf, Arena arena) throws IOException {
        return loadModel(channel, gguf, arena, null);
    }

    public static Qwen35 loadModel(FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        String arch = gguf.getString("general.architecture");
        if (!arch.equals("qwen35") && !arch.equals("qwen35moe"))
            throw new IllegalArgumentException("unsupported Qwen3.5 architecture: " + arch);
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        int context = gguf.getValue(int.class, arch + ".context_length");
        int dim = gguf.getValue(int.class, arch + ".embedding_length");
        int storedLayers = gguf.getValue(int.class, arch + ".block_count");
        int nextnLayers = gguf.getValueOrDefault(int.class, arch + ".nextn_predict_layers", 0);
        if (nextnLayers < 0 || nextnLayers > 1)
            throw new IllegalArgumentException(
                    "unsupported "
                            + arch
                            + ".nextn_predict_layers="
                            + nextnLayers
                            + " (expected 0 or 1)");
        int layers = storedLayers - nextnLayers;
        int heads = gguf.getValue(int.class, arch + ".attention.head_count");
        int kvHeads = gguf.getValue(int.class, arch + ".attention.head_count_kv");
        int headSize =
                gguf.getValueOrDefault(int.class, arch + ".attention.key_length", dim / heads);
        float eps =
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        float theta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 1_000_000f);
        int ropeDim = gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);
        int interval = gguf.getValueOrDefault(int.class, arch + ".full_attention_interval", 4);
        int hidden = gguf.getValueOrDefault(int.class, arch + ".feed_forward_length", 0);
        int inner = gguf.getValueOrDefault(int.class, arch + ".ssm.inner_size", 0);
        int groups = gguf.getValueOrDefault(int.class, arch + ".ssm.group_count", 0);
        int rank = gguf.getValueOrDefault(int.class, arch + ".ssm.time_step_rank", 0);
        int stateSize = gguf.getValueOrDefault(int.class, arch + ".ssm.state_size", 0);
        int convKernel = gguf.getValueOrDefault(int.class, arch + ".ssm.conv_kernel", 0);
        int expertCount = gguf.getValueOrDefault(int.class, arch + ".expert_count", 0);
        int expertUsed = gguf.getValueOrDefault(int.class, arch + ".expert_used_count", 0);
        int expertFfn = gguf.getValueOrDefault(int.class, arch + ".expert_feed_forward_length", 0);
        int sharedFfn =
                gguf.getValueOrDefault(int.class, arch + ".expert_shared_feed_forward_length", 0);
        if (dim <= 0 || layers <= 0 || heads <= 0 || kvHeads <= 0 || headSize <= 0 || context <= 0)
            throw new IllegalArgumentException("invalid Qwen3.5 dimensions in GGUF metadata");
        if (heads % kvHeads != 0
                || inner <= 0
                || groups <= 0
                || rank <= 0
                || inner % rank != 0
                || stateSize != inner / rank
                || convKernel <= 0
                || interval <= 0)
            throw new IllegalArgumentException("inconsistent Qwen3.5 attention/SSM dimensions");
        if ((expertCount == 0 && hidden <= 0)
                || (expertCount > 0
                        && (expertUsed <= 0 || expertUsed > expertCount || expertFfn <= 0)))
            throw new IllegalArgumentException("inconsistent Qwen3.5 FFN dimensions");
        boolean[] full = new boolean[storedLayers];
        for (int i = 0; i < layers; i++) full[i] = (i + 1) % interval == 0;
        if (nextnLayers == 1) full[layers] = true;
        Configuration config =
                new Configuration(
                        dim,
                        layers,
                        nextnLayers,
                        heads,
                        kvHeads,
                        headSize,
                        tokenizer.vocabulary().size(),
                        context,
                        eps,
                        theta,
                        ropeDim,
                        hidden,
                        full,
                        inner,
                        groups,
                        rank,
                        stateSize,
                        convKernel,
                        expertCount,
                        expertUsed,
                        expertFfn,
                        sharedFfn);
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new Qwen35(config, tokenizer, loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, MemoryView<MemorySegment>> tensors, Configuration c) {
        if (!c.hasMtp() && tensors.keySet().stream().anyMatch(name -> name.contains(".nextn.")))
            throw new IllegalArgumentException(
                    "Qwen3.5 GGUF contains nextn tensors but declares nextn_predict_layers=0");
        int n = c.storedLayers();
        MemoryView<MemorySegment> embedding = require(tensors, "token_embd.weight");
        MemoryView<MemorySegment> outputNorm = requireF32(tensors, "output_norm.weight");
        MemoryView<MemorySegment> output =
                tensors.containsKey("output.weight")
                        ? require(tensors, "output.weight")
                        : embedding;
        int ropeDim = Math.max(0, Math.min(c.ropeDimensionCount, c.headSize) & ~1);
        NextNWeights nextn = null;
        if (c.hasMtp()) {
            String block = p(c.mtpLayer()) + "nextn.";
            nextn =
                    new NextNWeights(
                            tensors.getOrDefault(block + "embed_tokens.weight", embedding),
                            requireF32(tensors, block + "enorm.weight"),
                            requireF32(tensors, block + "hnorm.weight"),
                            require(tensors, block + "eh_proj.weight"),
                            tensors.containsKey(block + "shared_head_norm.weight")
                                    ? requireF32(tensors, block + "shared_head_norm.weight")
                                    : outputNorm,
                            tensors.getOrDefault(block + "shared_head_head.weight", output));
        }
        boolean moe = c.isMoE(), shared = moe && c.expertSharedFeedForwardLength > 0;
        return new Weights(
                embedding,
                outputNorm,
                output,
                array(n, i -> requireF32(tensors, p(i) + "attn_norm.weight")),
                array(n, i -> requireF32(tensors, p(i) + "post_attention_norm.weight")),
                array(n, i -> requireIf(tensors, c.isFullAttention[i], p(i) + "attn_q.weight")),
                array(n, i -> requireIf(tensors, c.isFullAttention[i], p(i) + "attn_k.weight")),
                array(n, i -> requireIf(tensors, c.isFullAttention[i], p(i) + "attn_v.weight")),
                array(
                        n,
                        i -> requireIf(tensors, c.isFullAttention[i], p(i) + "attn_output.weight")),
                array(
                        n,
                        i ->
                                requireF32If(
                                        tensors,
                                        c.isFullAttention[i],
                                        p(i) + "attn_q_norm.weight")),
                array(
                        n,
                        i ->
                                requireF32If(
                                        tensors,
                                        c.isFullAttention[i],
                                        p(i) + "attn_k_norm.weight")),
                array(n, i -> requireIf(tensors, !c.isFullAttention[i], p(i) + "attn_qkv.weight")),
                array(n, i -> requireIf(tensors, !c.isFullAttention[i], p(i) + "attn_gate.weight")),
                array(n, i -> requireIf(tensors, !c.isFullAttention[i], p(i) + "ssm_alpha.weight")),
                array(n, i -> requireIf(tensors, !c.isFullAttention[i], p(i) + "ssm_beta.weight")),
                array(n, i -> requireIf(tensors, !c.isFullAttention[i], p(i) + "ssm_out.weight")),
                array(
                        n,
                        i ->
                                requireF32If(
                                        tensors,
                                        !c.isFullAttention[i],
                                        p(i) + "ssm_conv1d.weight")),
                array(n, i -> requireF32If(tensors, !c.isFullAttention[i], p(i) + "ssm_a")),
                array(n, i -> requireF32If(tensors, !c.isFullAttention[i], p(i) + "ssm_dt.bias")),
                array(
                        n,
                        i ->
                                requireF32If(
                                        tensors, !c.isFullAttention[i], p(i) + "ssm_norm.weight")),
                array(n, i -> requireIf(tensors, !moe, p(i) + "ffn_gate.weight")),
                array(n, i -> requireIf(tensors, !moe, p(i) + "ffn_up.weight")),
                array(n, i -> requireIf(tensors, !moe, p(i) + "ffn_down.weight")),
                array(
                        n,
                        i ->
                                requireFirstIf(
                                        tensors,
                                        moe,
                                        p(i) + "ffn_gate_inp.weight",
                                        p(i) + "ffn_router.weight")),
                array2(
                        n,
                        i -> {
                            MemoryView<MemorySegment> v =
                                    requireIf(tensors, moe, p(i) + "ffn_gate_exps.weight");
                            return v == null ? null : Views.sliceLeadingAxis(v);
                        }),
                array2(
                        n,
                        i -> {
                            MemoryView<MemorySegment> v =
                                    requireIf(tensors, moe, p(i) + "ffn_up_exps.weight");
                            return v == null ? null : Views.sliceLeadingAxis(v);
                        }),
                array2(
                        n,
                        i -> {
                            MemoryView<MemorySegment> v =
                                    requireIf(tensors, moe, p(i) + "ffn_down_exps.weight");
                            return v == null ? null : Views.sliceLeadingAxis(v);
                        }),
                array(
                        n,
                        i ->
                                requireFirstIf(
                                        tensors,
                                        shared,
                                        p(i) + "ffn_gate_shexp.weight",
                                        p(i) + "ffn_shared_expert_gate.weight")),
                array(
                        n,
                        i ->
                                requireFirstIf(
                                        tensors,
                                        shared,
                                        p(i) + "ffn_up_shexp.weight",
                                        p(i) + "ffn_shared_expert_up.weight")),
                array(
                        n,
                        i ->
                                requireFirstIf(
                                        tensors,
                                        shared,
                                        p(i) + "ffn_down_shexp.weight",
                                        p(i) + "ffn_shared_expert_down.weight")),
                array(
                        n,
                        i ->
                                requireFirstIf(
                                        tensors,
                                        shared,
                                        p(i) + "ffn_gate_inp_shexp.weight",
                                        p(i) + "ffn_shared_expert_gate_inp.weight")),
                ropeDim == 0 ? null : RoPE.plain(ropeDim, c.ropeTheta),
                ropeDim / 2,
                nextn);
    }

    private interface ViewAt {
        MemoryView<MemorySegment> get(int i);
    }

    private static MemoryView<MemorySegment>[] array(int n, ViewAt supplier) {
        MemoryView<MemorySegment>[] out = new MemoryView[n];
        for (int i = 0; i < n; i++) out[i] = supplier.get(i);
        return out;
    }

    private interface ViewArrayAt {
        MemoryView<MemorySegment>[] get(int i);
    }

    private static MemoryView<MemorySegment>[][] array2(int n, ViewArrayAt supplier) {
        MemoryView<MemorySegment>[][] out = new MemoryView[n][];
        for (int i = 0; i < n; i++) out[i] = supplier.get(i);
        return out;
    }

    private static String p(int i) {
        return "blk." + i + ".";
    }

    private static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> view = tensors.get(name);
        if (view == null) throw new IllegalArgumentException("Qwen3.5 GGUF is missing " + name);
        return view;
    }

    private static MemoryView<MemorySegment> requireF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> view = require(tensors, name);
        Views.requireDatatype(view, DataType.FP32, name);
        return view;
    }

    private static MemoryView<MemorySegment> requireIf(
            Map<String, MemoryView<MemorySegment>> tensors, boolean required, String name) {
        return required ? require(tensors, name) : null;
    }

    private static MemoryView<MemorySegment> requireF32If(
            Map<String, MemoryView<MemorySegment>> tensors, boolean required, String name) {
        return required ? requireF32(tensors, name) : null;
    }

    private static MemoryView<MemorySegment> requireFirstIf(
            Map<String, MemoryView<MemorySegment>> tensors,
            boolean required,
            String first,
            String second) {
        if (!required) return null;
        MemoryView<MemorySegment> view = ModelLoader.firstPresent(tensors, first, second);
        if (view == null) throw new IllegalArgumentException("Qwen3.5 GGUF is missing " + first);
        return view;
    }
}
