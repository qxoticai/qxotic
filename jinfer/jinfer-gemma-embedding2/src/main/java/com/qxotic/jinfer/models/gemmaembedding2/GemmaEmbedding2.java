package com.qxotic.jinfer.models.gemmaembedding2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.EmbeddingModel;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.PromptWriter;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jinfer.kernels.Trace;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jinfer.models.gemma4.Gemma4;
import com.qxotic.jota.Shape;
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
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Google EmbeddingGemma 2 ({@code general.architecture = gemma-embedding2}): a bidirectional Gemma
 * 4 text encoder pooled into one vector per input. Five sliding layers in six attend within a
 * SYMMETRIC window ({@code |i - j| <= sliding_window / 2}), the sixth to the whole input; Q/K are
 * RMS-normed per head and NeoX-rotated, V is RMS-normed without weight, attention is unscaled.
 * Every layer adds a per-layer input projected from the input rows (no per-layer token table) and
 * ends with a learned scalar. The final-normed rows are mean-pooled, projected to {@code
 * embedding_length_out} and L2-normalized (llama.cpp {@code gemma-embedding2}, pooling MEAN).
 *
 * <p>Rows are STAGED by {@link #ingest}: token rows scaled by {@code sqrt(width)}, media rows (a
 * Gemma 4 mmproj's towers, {@link #withMedia}) as projected. The forward runs over the whole staged
 * input when an embedding is projected, since every row attends to rows after it. {@link #embedAll}
 * packs whole text sequences instead, each attending only to itself.
 */
public final class GemmaEmbedding2
        implements EmbeddingModel<
                        GemmaEmbedding2.Configuration,
                        GemmaEmbedding2.Weights,
                        GemmaEmbedding2.State>,
                Multimodal {
    static final String ARCHITECTURE = "gemma-embedding2";

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;
    private final Gemma4.Towers towers;

    /** The reference video processor's {@code max_soft_tokens}: a frame's budget (a still: 280). */
    static final int VIDEO_FRAME_TOKENS = 140;

    final int bos, eos; // every input is framed <bos> ... <eos>

    private GemmaEmbedding2(
            Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            Gemma4.Towers towers,
            int bos,
            int eos) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
        this.towers = towers;
        this.bos = bos;
        this.eos = eos;
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

    /** A model sharing these weights with the towers of {@code mmprojPath} attached. */
    public GemmaEmbedding2 withMedia(Path mmprojPath, Arena arena) throws IOException {
        return new GemmaEmbedding2(
                configuration,
                tokenizer,
                weights,
                Gemma4.loadTowers(mmprojPath, configuration.embeddingLength, arena),
                bos,
                eos);
    }

    @Override
    public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
        return towers.projector(modality);
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

    // === Staging ===

    @Override
    public void ingest(State state, Batch batch) {
        state.exclusively(() -> stage(state, batch));
        Reference.reachabilityFence(this);
    }

    private void stage(State state, Batch batch) {
        int rows = batch.count(), at = state.position(), dim = configuration.embeddingLength;
        if (rows <= 0 || rows > state.batchCapacity())
            throw new IllegalArgumentException("invalid EmbeddingGemma 2 batch size " + rows);
        if (at + rows > state.contextCapacity())
            throw new IllegalArgumentException(
                    "ingest of "
                            + rows
                            + " at "
                            + at
                            + " exceeds contextCapacity "
                            + state.contextCapacity());
        switch (batch.input()) {
            case Batch.Input.Tokens t -> stageTokens(state, t.ids(), 0, rows, at);
            case Batch.Input.Embeddings e -> {
                if (e.rows().shape().flatAt(1) != dim)
                    throw new IllegalArgumentException(
                            "embedding width " + e.rows().shape().flatAt(1) + " != model " + dim);
                Convert.copyF32(
                        Views.castToSegmentBacked(e.rows(), "embedding rows"),
                        0,
                        state.input,
                        (long) at * dim,
                        (long) rows * dim);
            }
            case Batch.Input.Sequences ignored ->
                    throw new UnsupportedOperationException(
                            "EmbeddingGemma 2 embeds packed sequences through embedAll");
        }
        state.advance(batch);
    }

    /** Token rows into the staged input at row {@code at}, scaled by {@code sqrt(width)}. */
    private void stageTokens(State state, int[] ids, int from, int rows, int at) {
        int dim = configuration.embeddingLength;
        for (int i = from; i < from + rows; i++)
            if (ids[i] < 0 || ids[i] >= configuration.vocabularySize)
                throw new IllegalArgumentException("token id outside the vocabulary: " + ids[i]);
        Views.checkAlive(weights.tokenEmbedding, "tokenEmbedding");
        Convert.gatherToF32(
                weights.tokenEmbedding, ids, from, rows, state.input, (long) at * dim, dim);
        Ops.multiplyInPlace(state.input, (long) at * dim, rows * dim, (float) Math.sqrt(dim));
    }

    // === Embedding ===

    /**
     * Embeds one interleaved input: {@link String} text and {@link Media.Image}/{@link
     * Media.Audio}/{@link Media.Video} parts in order, framed as the reference processor frames
     * them - {@code <bos>}, the text's tokens, each medium's projected rows between {@code
     * <|image>}/{@code <image|>} (or {@code <|audio>}/{@code <audio|>}), {@code <eos>}. A video is
     * its frames back to back, each an image block sized at {@link #VIDEO_FRAME_TOKENS}; which
     * frames is the caller's sampling (the reference takes one a second, at most 32, see {@code
     * VideoCodec#every}). The state is reset first; the embedding is a state-owned view, live until
     * {@code consumer} returns. Returns the positions the input took.
     */
    @Override
    public int embed(State state, List<?> parts, Consumer<MemoryView<?>> consumer) {
        Objects.requireNonNull(parts, "parts");
        Objects.requireNonNull(consumer, "consumer");
        int[] positions = new int[1];
        try {
            state.exclusively(
                    () -> {
                        state.reset();
                        PromptWriter out =
                                new PromptWriter(
                                        tokenizer, state.batchCapacity(), b -> stage(state, b));
                        out.id(bos);
                        for (Object part : parts)
                            switch (part) {
                                case String text -> out.text(text);
                                case Media.Image image ->
                                        media(
                                                out,
                                                image,
                                                tower(projector(Media.Image.class), "image"));
                                case Media.Audio audio ->
                                        media(
                                                out,
                                                audio,
                                                tower(projector(Media.Audio.class), "audio"));
                                case Media.Video video -> {
                                    var frames = tower(towers.frames(VIDEO_FRAME_TOKENS), "video");
                                    for (Media.Video.Frame frame : video.frames())
                                        media(out, frame.image(), frames);
                                }
                                case null, default ->
                                        throw new IllegalArgumentException(
                                                "parts are String or Media.Image, Media.Audio,"
                                                        + " Media.Video, not "
                                                        + part);
                            }
                        out.id(eos).finish();
                        positions[0] = state.position();
                        consumer.accept(embedding(state));
                    });
        } finally {
            Reference.reachabilityFence(this);
        }
        return positions[0];
    }

    private <R extends Media> void media(PromptWriter out, R media, MediaProjector<R> projector) {
        boolean audio = media instanceof Media.Audio;
        out.id(SpecialTokens.require(tokenizer, audio ? "<|audio>" : "<|image>"))
                .media(media, null, projector, true)
                .id(SpecialTokens.require(tokenizer, audio ? "<audio|>" : "<image|>"));
    }

    private static <R extends Media> MediaProjector<R> tower(
            Optional<MediaProjector<R>> projector, String medium) {
        return projector.orElseThrow(
                () ->
                        new UnsupportedOperationException(
                                "no tower for "
                                        + medium
                                        + " input attached: load the mmproj with withMedia"));
    }

    /** A whole packed stream: every length positive, summing to the token count. */
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

    /** The whole staged input's embedding: every retained output projects the same mean pool. */
    @Override
    public void projectEmbedding(State state, int outputIndex, Consumer<MemoryView<?>> consumer) {
        Objects.requireNonNull(consumer, "consumer");
        try {
            state.exclusively(
                    () -> {
                        if (outputIndex < 0 || outputIndex >= state.outputCount())
                            throw new IllegalArgumentException(
                                    "output "
                                            + outputIndex
                                            + " outside [0,"
                                            + state.outputCount()
                                            + ")");
                        consumer.accept(embedding(state));
                    });
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    /** The staged rows as one sequence, encoded and pooled. */
    private MemoryView<MemorySegment> embedding(State state) {
        int rows = state.position();
        forward(state, new int[] {rows});
        return pool(state, 0, rows);
    }

    /**
     * Packs whole sequences into context-sized groups, one forward per group, each sequence
     * attending only to itself; delivers one embedding per sequence in input order.
     */
    @Override
    public void embedAll(
            State state, Batch.Input.Sequences sequences, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(sink, "sink");
        requireComplete(sequences);
        int[] lengths = sequences.seqLen(), ids = sequences.tokens().ids();
        for (int i = 0; i < lengths.length; i++)
            if (lengths[i] > state.contextCapacity())
                throw new IllegalArgumentException(
                        "sequence "
                                + i
                                + " has "
                                + lengths[i]
                                + " tokens, over contextCapacity "
                                + state.contextCapacity());
        try {
            state.exclusively(
                    () -> {
                        state.reset();
                        for (int first = 0, at = 0; first < lengths.length; ) {
                            int last = first, rows = 0;
                            while (last < lengths.length
                                    && rows + lengths[last] <= state.contextCapacity())
                                rows += lengths[last++];
                            stageTokens(state, ids, at, rows, 0);
                            forward(state, Arrays.copyOfRange(lengths, first, last));
                            for (int i = first, start = 0; i < last; start += lengths[i++])
                                sink.accept(pool(state, start, lengths[i]));
                            at += rows;
                            first = last;
                        }
                    });
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    /** Final-norm, mean-pool and project {@code rows} rows from {@code start}, L2-normalized. */
    private MemoryView<MemorySegment> pool(State state, int start, int rows) {
        int dim = configuration.embeddingLength, out = configuration.embeddingLengthOut;
        Norms.rmsnormRows(
                state.normed,
                state.residual.slice(0, start, start + rows),
                weights.outputNorm,
                rows,
                dim,
                configuration.rmsNormEps);
        Ops.fillInPlace(state.pooled, 0, dim, 0f);
        for (int r = 0; r < rows; r++)
            Ops.addInPlace(state.pooled, 0, state.normed, (long) r * dim, dim);
        Ops.multiplyInPlace(state.pooled, 0, dim, 1f / rows);
        MatMul.gemv(weights.output, state.pooled, state.embedding);
        float squares = Norms.sumOfSquares(state.embedding, 0, out);
        Ops.multiplyInPlace(
                state.embedding, 0, out, squares > 0 ? (float) (1 / Math.sqrt(squares)) : 0f);
        return state.embedding;
    }

    // === Forward ===

    /**
     * The encoder over the staged rows, cut into independent segments of {@code lengths}: RoPE
     * positions restart and attention stays inside each one.
     */
    private void forward(State state, int[] lengths) {
        Configuration c = configuration;
        int rows = 0, radius = c.slidingWindow / 2, longest = 0;
        for (int length : lengths) {
            for (int i = rows, end = rows + length; i < end; i++) {
                state.first[i] = rows;
                state.end[i] = end;
                state.windowFirst[i] = Math.max(rows, i - radius);
                state.windowEnd[i] = Math.min(end, i + radius + 1);
            }
            rows += length;
            longest = Math.max(longest, length);
        }
        state.ropeUpTo(longest);
        Convert.copyF32(state.input, 0, state.residual, 0, (long) rows * c.embeddingLength);
        for (int l = 0; l < c.numberOfLayers; l++) {
            attention(state, l, rows);
            feedForward(state, l, rows);
            perLayerInput(state, l, rows);
            if (Trace.ENABLED) Trace.sum("l_out-" + l, state.residual, rows * c.embeddingLength);
        }
    }

    private void attention(State state, int l, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[l];
        boolean swa = c.isSwa[l];
        int headSize = c.headSize(l), kvHeads = c.numberOfKeyValueHeads[l];
        MemoryView<MemorySegment> cos = swa ? state.ropeCosSwa : state.ropeCosFull;
        MemoryView<MemorySegment> sin = swa ? state.ropeSinSwa : state.ropeSinFull;

        Norms.rmsnormRows(
                state.normed, state.residual, w.attnNorm, rows, c.embeddingLength, c.rmsNormEps);
        MatMul.gemm(w.query, state.normed, state.query, rows);
        MatMul.gemm(w.key, state.normed, state.key, rows);
        MatMul.gemm(w.value, state.normed, state.value, rows);
        Parallel.forLoop(
                rows,
                r -> {
                    int position = r - state.first[r];
                    for (int h = 0; h < c.numberOfHeads; h++) {
                        long at = (long) r * state.queryStride + (long) h * headSize;
                        Norms.rmsnorm(
                                state.query,
                                at,
                                state.query,
                                at,
                                w.queryNorm,
                                headSize,
                                c.rmsNormEps);
                        RoPE.applyNeox(state.query, at, position, cos, sin, headSize / 2);
                    }
                    for (int h = 0; h < kvHeads; h++) {
                        long at = (long) r * state.kvStride + (long) h * headSize;
                        Norms.rmsnorm(
                                state.key, at, state.key, at, w.keyNorm, headSize, c.rmsNormEps);
                        RoPE.applyNeox(state.key, at, position, cos, sin, headSize / 2);
                        Norms.rmsnormNoWeight(
                                state.value, at, state.value, at, headSize, c.rmsNormEps);
                    }
                });
        // the heads' outputs packed at this layer's width, over the shared buffer
        int width = c.numberOfHeads * headSize;
        MemoryView<MemorySegment> out =
                state.attentionOut.slice(0, 0, (long) rows * width).view(Shape.flat(rows, width));
        GemmAttention.prefill(
                state.query,
                state.queryStride,
                state.key,
                state.value,
                state.kvStride,
                out,
                rows,
                c.numberOfHeads,
                kvHeads,
                headSize,
                swa ? state.windowFirst : state.first,
                swa ? state.windowEnd : state.end,
                state.attentionScratch);
        MatMul.gemm(w.output, out, state.branch, rows);
        normThenAdd(state, state.branch, w.postAttnNorm, rows, 1f);
    }

    private void feedForward(State state, int l, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[l];
        int hidden = c.feedForwardLength;
        Norms.rmsnormRows(
                state.normed, state.residual, w.ffnNorm, rows, c.embeddingLength, c.rmsNormEps);
        MatMul.gemm(w.gate, state.normed, state.hidden, rows);
        MatMul.gemm(w.up, state.normed, state.hidden2, rows);
        Parallel.forLoop(
                rows,
                r ->
                        Activations.geluMultiply(
                                state.hidden, r * hidden, state.hidden2, r * hidden, hidden));
        MatMul.gemm(w.down, state.hidden, state.branch, rows);
        normThenAdd(state, state.branch, w.postFfnNorm, rows, 1f);
    }

    /**
     * The layer's slice of the per-layer model projection of the INPUT rows, scaled by {@code
     * 1/sqrt(width)} and RMS-normed, gates {@code gelu(inp_gate x)}; projected back, post-normed
     * and added.
     */
    private void perLayerInput(State state, int l, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[l];
        int dim = c.embeddingLength, plDim = c.embeddingLengthPerLayer;
        MatMul.gemm(
                weights.perLayerModelProjection,
                (long) l * plDim * dim,
                state.input,
                dim,
                state.perLayerInput,
                plDim,
                plDim,
                rows,
                dim);
        MatMul.gemm(w.inputGate, state.residual, state.perLayerGate, rows);
        float scale = (float) (1 / Math.sqrt(dim));
        Parallel.forLoop(
                rows,
                r -> {
                    long at = (long) r * plDim;
                    Ops.multiplyInPlace(state.perLayerInput, at, plDim, scale);
                    Norms.rmsnorm(
                            state.perLayerInput,
                            at,
                            state.perLayerInput,
                            at,
                            weights.perLayerProjectionNorm,
                            plDim,
                            c.rmsNormEps);
                    Activations.geluMultiply(
                            state.perLayerGate, (int) at, state.perLayerInput, (int) at, plDim);
                });
        MatMul.gemm(w.projection, state.perLayerGate, state.branch, rows);
        normThenAdd(state, state.branch, w.postProjectionNorm, rows, w.outputScale);
    }

    /** A branch's close: post-norm it, add it to the residual, scale the sum. */
    private void normThenAdd(
            State state,
            MemoryView<MemorySegment> branch,
            MemoryView<MemorySegment> norm,
            int rows,
            float scale) {
        int dim = configuration.embeddingLength;
        Parallel.forLoop(
                rows,
                r -> {
                    long at = (long) r * dim;
                    Norms.rmsnorm(branch, at, branch, at, norm, dim, configuration.rmsNormEps);
                    Ops.addInPlace(state.residual, at, branch, at, dim);
                    if (scale != 1f) Ops.multiplyInPlace(state.residual, at, dim, scale);
                });
    }

    // === Configuration ===

    /**
     * The checkpoint's shape. {@code slidingWindow} is llama.cpp's {@code n_swa}, the full width of
     * the symmetric window; full and sliding layers have their own head size and rope base.
     */
    public record Configuration(
            int embeddingLength,
            int feedForwardLength,
            int numberOfLayers,
            int numberOfHeads,
            int[] numberOfKeyValueHeads,
            int headSizeFull,
            int headSizeSwa,
            int vocabularySize,
            int maxContextLength,
            float rmsNormEps,
            double ropeThetaFull,
            double ropeThetaSwa,
            int slidingWindow,
            boolean[] isSwa,
            int embeddingLengthPerLayer,
            int embeddingLengthOut)
            implements ContextConfiguration {

        int headSize(int layer) {
            return isSwa[layer] ? headSizeSwa : headSizeFull;
        }
    }

    // === Weights ===

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> queryNorm,
            MemoryView<MemorySegment> keyNorm,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> postAttnNorm,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> up,
            MemoryView<MemorySegment> down,
            MemoryView<MemorySegment> postFfnNorm,
            MemoryView<MemorySegment> inputGate,
            MemoryView<MemorySegment> projection,
            MemoryView<MemorySegment> postProjectionNorm,
            float outputScale) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbedding,
            MemoryView<MemorySegment> perLayerModelProjection,
            MemoryView<MemorySegment> perLayerProjectionNorm,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> output,
            LayerWeights[] layers) {}

    // === State ===

    /**
     * Staged input rows and the encoder's scratch, all sized to the context: one pass sees all.
     * Rows sit at the widest layer's strides; RoPE tables cover every position once.
     */
    public static final class State extends ContextState {
        final MemoryView<MemorySegment> input, residual, normed, branch;
        final MemoryView<MemorySegment> query, attentionOut, key, value, hidden, hidden2;
        final MemoryView<MemorySegment> perLayerInput, perLayerGate, pooled, embedding;
        final GemmAttention.Scratch attentionScratch;
        final MemoryView<MemorySegment> ropeCosFull, ropeSinFull, ropeCosSwa, ropeSinSwa;
        final int queryStride, kvStride;
        // per row: its segment's keys, then those within the window
        final int[] first, end, windowFirst, windowEnd;
        private final RoPE.Schedule ropeFull, ropeSwa;
        private int ropeRows; // positions the tables cover so far

        State(
                Configuration c,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity > c.maxContextLength)
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds the model's maxContextLength "
                                + c.maxContextLength);
            int rows = contextCapacity, dim = c.embeddingLength;
            int q = 0, kv = 0;
            for (int l = 0; l < c.numberOfLayers; l++) {
                q = Math.max(q, c.numberOfHeads * c.headSize(l));
                kv = Math.max(kv, c.numberOfKeyValueHeads[l] * c.headSize(l));
            }
            queryStride = q;
            kvStride = kv;
            input = Views.allocateF32(memoryArena(), rows, dim);
            residual = Views.allocateF32(memoryArena(), rows, dim);
            normed = Views.allocateF32(memoryArena(), rows, dim);
            branch = Views.allocateF32(memoryArena(), rows, dim);
            query = Views.allocateF32(memoryArena(), rows, q);
            attentionOut = Views.allocateF32(memoryArena(), (long) rows * q); // per-layer width
            key = Views.allocateF32(memoryArena(), rows, kv);
            value = Views.allocateF32(memoryArena(), rows, kv);
            hidden = Views.allocateF32(memoryArena(), rows, c.feedForwardLength);
            hidden2 = Views.allocateF32(memoryArena(), rows, c.feedForwardLength);
            perLayerInput = Views.allocateF32(memoryArena(), rows, c.embeddingLengthPerLayer);
            perLayerGate = Views.allocateF32(memoryArena(), rows, c.embeddingLengthPerLayer);
            attentionScratch = GemmAttention.Scratch.allocate(memoryArena(), rows, kv, q);
            pooled = Views.allocateF32(memoryArena(), 1, dim);
            embedding = Views.allocateF32(memoryArena(), 1, c.embeddingLengthOut);
            ropeCosFull = Views.allocateF32(memoryArena(), rows, c.headSizeFull / 2);
            ropeSinFull = Views.allocateF32(memoryArena(), rows, c.headSizeFull / 2);
            ropeCosSwa = Views.allocateF32(memoryArena(), rows, c.headSizeSwa / 2);
            ropeSinSwa = Views.allocateF32(memoryArena(), rows, c.headSizeSwa / 2);
            ropeFull = RoPE.plain(c.headSizeFull, c.ropeThetaFull);
            ropeSwa = RoPE.plain(c.headSizeSwa, c.ropeThetaSwa);
            first = new int[rows];
            end = new int[rows];
            windowFirst = new int[rows];
            windowEnd = new int[rows];
        }

        /** Grows the RoPE tables to cover positions {@code [0, positions)}. */
        void ropeUpTo(int positions) {
            if (positions <= ropeRows) return;
            int from = ropeRows, n = positions - from;
            RoPE.fill(
                    ropeCosFull.slice(0, from, positions),
                    ropeSinFull.slice(0, from, positions),
                    from,
                    n,
                    (int) ropeCosFull.shape().flatAt(1),
                    ropeFull);
            RoPE.fill(
                    ropeCosSwa.slice(0, from, positions),
                    ropeSinSwa.slice(0, from, positions),
                    from,
                    n,
                    (int) ropeCosSwa.shape().flatAt(1),
                    ropeSwa);
            ropeRows = positions;
        }

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }

        /** Staged rows past the cursor are dead: the next ingest overwrites them. */
        @Override
        protected void clearHistory() {}
    }

    // === Loading ===

    public static GemmaEmbedding2 loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return loadModel(channel, ModelLoader.readGguf(channel, path.toString()), arena, null);
        }
    }

    /**
     * Loads an EmbeddingGemma 2 GGUF over an already-parsed header; {@code tokenizer} null builds
     * the GGUF's own.
     */
    public static GemmaEmbedding2 loadModel(
            FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer) throws IOException {
        String arch = gguf.getString("general.architecture");
        if (!ARCHITECTURE.equals(arch))
            throw new IllegalArgumentException("unsupported architecture: " + arch);
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration c = loadConfiguration(gguf, tokenizer.vocabulary().size());
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        return new GemmaEmbedding2(
                c,
                tokenizer,
                loadWeights(tensors, c),
                Gemma4.Towers.NONE,
                gguf.getValue(int.class, "tokenizer.ggml.bos_token_id"),
                gguf.getValue(int.class, "tokenizer.ggml.eos_token_id"));
    }

    /** Reads what llama.cpp reads, and refuses what its graph does not implement. */
    static Configuration loadConfiguration(GGUF gguf, int vocabularySize) {
        String a = ARCHITECTURE + ".";
        int layers = gguf.getValue(int.class, a + "block_count");
        int heads = gguf.getValue(int.class, a + "attention.head_count");
        int headFull = gguf.getValue(int.class, a + "attention.key_length");
        int headSwa = gguf.getValue(int.class, a + "attention.key_length_swa");
        Object kv = gguf.getValue(Object.class, a + "attention.head_count_kv");
        int[] kvHeads = new int[layers];
        if (kv instanceof int[] perLayer) kvHeads = perLayer.clone();
        else Arrays.fill(kvHeads, ((Number) kv).intValue());
        Configuration c =
                new Configuration(
                        gguf.getValue(int.class, a + "embedding_length"),
                        gguf.getValue(int.class, a + "feed_forward_length"),
                        layers,
                        heads,
                        kvHeads,
                        headFull,
                        headSwa,
                        vocabularySize,
                        gguf.getValue(int.class, a + "context_length"),
                        gguf.getValueOrDefault(
                                float.class, a + "attention.layer_norm_rms_epsilon", 1e-6f),
                        gguf.getValue(float.class, a + "rope.freq_base"),
                        gguf.getValueOrDefault(float.class, a + "rope.freq_base_swa", 10_000f),
                        gguf.getValue(int.class, a + "attention.sliding_window"),
                        swaLayers(gguf, layers),
                        gguf.getValue(int.class, a + "embedding_length_per_layer_input"),
                        gguf.getValue(int.class, a + "embedding_length_out"));
        require(
                kvHeads.length == layers && c.isSwa.length == layers,
                "one KV head count and one window flag per layer");
        for (int l = 0; l < layers; l++)
            require(
                    kvHeads[l] > 0 && heads % kvHeads[l] == 0,
                    "layer " + l + ": " + kvHeads[l] + " KV heads for " + heads);
        require(
                gguf.getValueOrDefault(int.class, a + "attention.value_length", headFull)
                                == headFull
                        && gguf.getValueOrDefault(
                                        int.class, a + "attention.value_length_swa", headSwa)
                                == headSwa
                        && gguf.getValueOrDefault(int.class, a + "rope.dimension_count", headFull)
                                == headFull
                        && gguf.getValueOrDefault(
                                        int.class, a + "rope.dimension_count_swa", headSwa)
                                == headSwa,
                "key, value and rotary widths must equal the head size");
        require(
                !gguf.getValueOrDefault(boolean.class, a + "attention.causal", false)
                        && gguf.getValueOrDefault(int.class, a + "pooling_type", 1) == 1,
                "only the bidirectional mean-pooled encoder is implemented");
        require(
                gguf.getValueOrDefault(int.class, a + "attention.shared_kv_layers", 0) == 0,
                "shared KV layers are not implemented");
        return c;
    }

    /**
     * Which layers slide, as llama.cpp reads {@code attention.sliding_window_pattern}: one flag per
     * layer, or a period {@code n} (layer {@code il} slides unless it is the last of its period; 0
     * slides every layer) defaulting to 6.
     */
    private static boolean[] swaLayers(GGUF gguf, int layers) {
        Object pattern =
                gguf.getValueOrDefault(
                        Object.class, ARCHITECTURE + ".attention.sliding_window_pattern", 6);
        if (pattern instanceof boolean[] perLayer) return perLayer.clone();
        int period = ((Number) pattern).intValue();
        boolean[] swa = new boolean[layers];
        for (int l = 0; l < layers; l++) swa[l] = period == 0 || l % period < period - 1;
        return swa;
    }

    static Weights loadWeights(Map<String, MemoryView<MemorySegment>> tensors, Configuration c) {
        LayerWeights[] layers = new LayerWeights[c.numberOfLayers];
        for (int l = 0; l < layers.length; l++) {
            String p = "blk." + l + ".";
            layers[l] =
                    new LayerWeights(
                            ModelLoader.requireF32(tensors, p + "attn_norm.weight"),
                            ModelLoader.require(tensors, p + "attn_q.weight"),
                            ModelLoader.require(tensors, p + "attn_k.weight"),
                            ModelLoader.require(tensors, p + "attn_v.weight"),
                            ModelLoader.requireF32(tensors, p + "attn_q_norm.weight"),
                            ModelLoader.requireF32(tensors, p + "attn_k_norm.weight"),
                            ModelLoader.require(tensors, p + "attn_output.weight"),
                            ModelLoader.requireF32(tensors, p + "post_attention_norm.weight"),
                            ModelLoader.requireF32(tensors, p + "ffn_norm.weight"),
                            ModelLoader.require(tensors, p + "ffn_gate.weight"),
                            ModelLoader.require(tensors, p + "ffn_up.weight"),
                            ModelLoader.require(tensors, p + "ffn_down.weight"),
                            ModelLoader.requireF32(tensors, p + "post_ffw_norm.weight"),
                            ModelLoader.require(tensors, p + "inp_gate.weight"),
                            ModelLoader.require(tensors, p + "proj.weight"),
                            ModelLoader.requireF32(tensors, p + "post_norm.weight"),
                            Views.getFloat(
                                    ModelLoader.requireF32(
                                            tensors, p + "layer_output_scale.weight"),
                                    0,
                                    p + "layer_output_scale.weight"));
        }
        return new Weights(
                ModelLoader.require(tensors, "token_embd.weight"),
                ModelLoader.require(tensors, "per_layer_model_proj.weight"),
                ModelLoader.requireF32(tensors, "per_layer_proj_norm.weight"),
                ModelLoader.requireF32(tensors, "output_norm.weight"),
                ModelLoader.require(tensors, "output.weight"),
                layers);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("EmbeddingGemma 2: " + message);
    }
}
