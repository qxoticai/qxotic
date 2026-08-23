// Qwen3 against the MemoryView boundary. A standard decoder-only Llama-family
// transformer: RMSNorm + grouped-query attention with per-head q/k RMS-norm (QK-norm), NeoX
// (split-half) rotary, SwiGLU FFN. No conv, no MoE, no gated attention, no embedding norm. Ties
// the LM head to the embedding table when output.weight is absent; embedding checkpoints pool
// the LAST row and L2-normalize; reranker checkpoints answer through the tied head ({yes,no}).
// Weights/state/KV are MemoryView<MemorySegment>; GEMM/GEMV use MatMul's shared public entry
// points.
//
// Packed Sequences are ISOLATED: RoPE positions restart per sequence and a sequence attends only
// to its own tokens (its KV slice: earlier chunks from the cache, this chunk from the F32 batch),
// never a packed neighbour's - the projection/FFN GEMMs stay batched over the whole chunk, only
// positions and attention are segmented. A sequence may span chunks; the KV carry lives in the
// global cache layout (row = absolute stream position). Tokens ingest is the degenerate one-piece
// case of the same law (the sequence started at row 0, the cursor is its prior).
package com.qxotic.jinfer.models.qwen3;

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
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jinfer.kernels.Trace;
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
import java.util.function.Consumer;

public final class Qwen3
        implements LanguageModel<Qwen3.Configuration, Qwen3.Weights, Qwen3.State>,
                EmbeddingModel<Qwen3.Configuration, Qwen3.Weights, Qwen3.State> {

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    Qwen3(Configuration configuration, Tokenizer tokenizer, Weights weights) {
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
        return Optional.of(new Qwen3CheckpointCodec(configuration));
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
    public void ingest(State s, Batch batch) {
        s.exclusively(() -> forward(s, batch));
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
        int[] ids;
        int nPieces;
        switch (batch.input()) {
            case Batch.Input.Tokens t -> {
                ids = t.ids();
                // the degenerate one-piece case: this sequence started at row 0, the cursor is
                // its prior - within-sequence positions ARE the absolute ones
                for (int r = 0; r < n; r++) s.posOf[r] = from + r;
                s.pieceRow0[0] = 0;
                s.pieceLen[0] = n;
                s.pieceKv[0] = 0;
                s.piecePrior[0] = from;
                nPieces = 1;
            }
            case Batch.Input.Sequences seq -> {
                ids = seq.tokens().ids();
                nPieces = cutPieces(seq.seqLen(), from, n, s);
            }
            case Batch.Input.Embeddings ignored ->
                    throw new UnsupportedOperationException(
                            "Qwen3 does not support embedding inputs");
        }
        if (n == 1)
            Parallel.runDecodeStep(
                    () -> {
                        forward(s, ids, from, n, nPieces);
                        return null;
                    });
        else forward(s, ids, from, n, nPieces);
        s.advance(batch);
    }

    /**
     * Group the chunk rows {@code [cs, cs+n)} into per-sequence pieces {row0, len, kvStart (global
     * cache row of the sequence's start), prior (its rows already cached before this chunk)}, and
     * fill {@code posOf} with each row's WITHIN-SEQUENCE position (restarts per sequence). {@code
     * fullSeqLen} is the FULL stream layout; a sequence may span chunks.
     */
    private static int cutPieces(int[] fullSeqLen, int cs, int n, State s) {
        int nPieces = 0, gStart = 0, j = 0;
        while (j < fullSeqLen.length && gStart + fullSeqLen[j] <= cs) {
            gStart += fullSeqLen[j];
            j++;
        }
        for (int r = 0; r < n; ) {
            while (j < fullSeqLen.length && cs + r >= gStart + fullSeqLen[j]) {
                gStart += fullSeqLen[j];
                j++;
            }
            int row0 = r, prior = (cs + r) - gStart, kvStart = gStart;
            int seqEnd = gStart + fullSeqLen[j];
            for (; r < n && cs + r < seqEnd; r++) s.posOf[r] = (cs + r) - gStart;
            s.pieceRow0[nPieces] = row0;
            s.pieceLen[nPieces] = r - row0;
            s.pieceKv[nPieces] = kvStart;
            s.piecePrior[nPieces] = prior;
            nPieces++;
        }
        return nPieces;
    }

    /** Causal chunk forward over rows {@code [startPos, startPos+seqLen)} of the stream. */
    private void forward(State state, int[] tokens, int startPos, int seqLen, int nPieces) {
        // ONCE for the batch: an angle never depends on the layer. Not a range: each sequence
        // restarts at position 0, so the tables are filled from per-row positions
        RoPE.fill(
                state.ropeCos,
                state.ropeSin,
                state.posOf,
                seqLen,
                configuration.ropeDim / 2,
                weights.rope());
        embedTokens(state, tokens, seqLen);
        for (int l = 0; l < configuration.numberOfLayers; l++) {
            attention(state, l, startPos, seqLen, nPieces);
            feedForward(state, l, seqLen);
            if (Trace.ENABLED)
                Trace.sum("l_out-" + l, state.residual, seqLen * configuration.embeddingLength);
        }
    }

    /** Token-embedding lookup into the residual stream (no scaling). */
    private void embedTokens(State state, int[] tokens, int seqLen) {
        Views.checkAlive(weights.tokenEmbeddings, "tokenEmbeddings"); // fail-fast on freed weights
        int dim = configuration.embeddingLength;
        // per-row dispatch via Convert.copyToF32 (the cost profile of the old per-row
        // virtual copyTo it replaces); the batched gather-dequant - dispatch hoisted once per
        // table - is a planned, separately-benchmarked commit, not a polish
        for (int s = 0; s < seqLen; s++) {
            Convert.copyToF32(
                    weights.tokenEmbeddings,
                    (long) tokens[s] * dim,
                    state.residual,
                    (long) s * dim,
                    dim);
        }
    }

    // --- attention (GQA) ---

    /**
     * Pre-norm GQA: per-head Q/K RMS-norm + NeoX RoPE (no V-norm), per-sequence isolated causal
     * attention with {@code scale = 1/sqrt(headSize)}, output projection, added to the residual.
     * One piece = one flash pass over the sequence's OWN KV slice: prior rows from the cache, this
     * chunk's from the F32 batch.
     */
    private void attention(State state, int l, int startPos, int seqLen, int nPieces) {
        Configuration config = configuration;
        int headSize = config.headSize;
        attentionProject(state, l, seqLen);
        commitKv(state, l, startPos, seqLen);
        float scale = 1.0f / (float) Math.sqrt(headSize);
        if (nPieces == 1 && state.pieceKv[0] == 0) {
            // the Tokens law (or a single sequence from the stream's start): the whole chunk is
            // one piece whose cache slice is the cache itself
            if (seqLen > 1) {
                FlashAttention.slidingWindowPrefill(
                        state.query,
                        state.attnOut,
                        state.keyCache[l],
                        state.valueCache[l],
                        state.batchK,
                        state.batchV,
                        config.numberOfHeads,
                        startPos,
                        seqLen,
                        headSize,
                        config.kvDim,
                        config.queryDim,
                        config.kvDim,
                        config.kvMul,
                        scale,
                        0,
                        0,
                        null);
            } else {
                FlashAttention.flashDecode(
                        state.query,
                        state.attnOut,
                        state.keyCache[l],
                        state.valueCache[l],
                        state.batchK,
                        state.batchV,
                        config.numberOfHeads,
                        startPos,
                        0,
                        headSize,
                        config.kvDim,
                        config.kvMul,
                        scale,
                        0,
                        null,
                        state.decodeScratch);
            }
        } else {
            for (int p = 0; p < nPieces; p++) {
                int r0 = state.pieceRow0[p], sl = state.pieceLen[p];
                int kvStart = state.pieceKv[p], prior = state.piecePrior[p];
                // jota's slice(dim, from, to-exclusive): rows r0 .. r0+sl of the flat scratch
                MemoryView<MemorySegment> qP =
                        state.query.slice(
                                0, (long) r0 * config.queryDim, (long) (r0 + sl) * config.queryDim);
                MemoryView<MemorySegment> oP =
                        state.attnOut.slice(
                                0, (long) r0 * config.queryDim, (long) (r0 + sl) * config.queryDim);
                MemoryView<MemorySegment> bKP =
                        state.batchK.slice(
                                0, (long) r0 * config.kvDim, (long) (r0 + sl) * config.kvDim);
                MemoryView<MemorySegment> bVP =
                        state.batchV.slice(
                                0, (long) r0 * config.kvDim, (long) (r0 + sl) * config.kvDim);
                // the sequence's own earlier rows: a slice at its global start (a packed
                // neighbour's rows are outside it); the cache is [rows, kvDim] - slice by ROWS.
                // prior == 0: nothing is read, pass the cache
                MemoryView<MemorySegment> cKP =
                        prior > 0
                                ? state.keyCache[l].slice(
                                        0, (long) kvStart, (long) (kvStart + prior))
                                : state.keyCache[l];
                MemoryView<MemorySegment> cVP =
                        prior > 0
                                ? state.valueCache[l].slice(
                                        0, (long) kvStart, (long) (kvStart + prior))
                                : state.valueCache[l];
                FlashAttention.slidingWindowPrefill(
                        qP,
                        oP,
                        cKP,
                        cVP,
                        bKP,
                        bVP,
                        config.numberOfHeads,
                        prior,
                        sl,
                        headSize,
                        config.kvDim,
                        config.queryDim,
                        config.kvDim,
                        config.kvMul,
                        scale,
                        0,
                        0,
                        null);
            }
        }
        attentionFinish(state, l, seqLen);
    }

    /**
     * The shared head of both attention paths: pre-norm, Q/K/V projections into {@code
     * query}/{@code batchK}/{@code batchV}, per-head QK RMS-norm + NeoX RoPE. The old port's {@code
     * Ops.scaleRows(headNormScale)} after the norm is dropped: headNormScale is 1.0f (hardcoded),
     * an identity.
     */
    private void attentionProject(State state, int l, int seqLen) {
        Configuration config = configuration;
        int dim = config.embeddingLength;
        LayerWeights lw = weights.layers[l];

        Norms.rmsnormRows(
                state.normed, state.residual, lw.attnNorm(), seqLen, dim, config.rmsNormEps);
        MatMul.gemm(lw.wq(), state.normed, state.query, seqLen);
        headNormRope(
                state, state.query, config.queryDim, config.numberOfHeads, lw.attnQNorm(), seqLen);
        MatMul.gemm(lw.wk(), state.normed, state.batchK, seqLen);
        MatMul.gemm(lw.wv(), state.normed, state.batchV, seqLen);
        headNormRope(
                state,
                state.batchK,
                config.kvDim,
                config.numberOfKeyValueHeads,
                lw.attnKNorm(),
                seqLen);
    }

    /** The shared tail: output projection, added to the residual. */
    private void attentionFinish(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength;
        MatMul.gemm(weights.layers[l].wo(), state.attnOut, state.branchOut, seqLen);
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
        int headSize = configuration.headSize, halfHeadSize = configuration.ropeDim / 2;
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

    /** Pre-norm SwiGLU FFN added to the residual. */
    private void feedForward(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        LayerWeights lw = weights.layers[l];
        Norms.rmsnormRows(
                state.normed, state.residual, lw.ffnNorm(), seqLen, dim, configuration.rmsNormEps);
        MatMul.gemm(lw.w1(), state.normed, state.hidden, seqLen);
        MatMul.gemm(lw.w3(), state.normed, state.hidden2, seqLen);
        Parallel.forLoop(
                seqLen,
                s ->
                        Activations.siluMultiply(
                                state.hidden,
                                s * hiddenDim,
                                state.hidden2,
                                s * hiddenDim,
                                hiddenDim));
        MatMul.gemm(lw.w2(), state.hidden, state.normed, seqLen);
        Ops.addInPlace(state.residual, 0, state.normed, 0, seqLen * dim);
    }

    /**
     * Write the chunk's K/V into layer {@code l}'s (linear) cache - called INSIDE the layer,
     * because {@code batchK}/{@code batchV} are single buffers reused by every layer: a commit
     * deferred to the end of the forward would write the LAST layer's values into every layer's
     * cache. LFM2 can defer it because it keeps per-layer batch buffers.
     */
    private void commitKv(State state, int l, int startPos, int seqLen) {
        int kvDim = configuration.kvDim;
        for (int s = 0; s < seqLen; s++) {
            long kvPos = startPos + s;
            Convert.f32ToF16(
                    state.batchK, (long) s * kvDim, state.keyCache[l], kvPos * kvDim, kvDim);
            Convert.f32ToF16(
                    state.batchV, (long) s * kvDim, state.valueCache[l], kvPos * kvDim, kvDim);
        }
    }

    // --- heads ---

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
        return Parallel.runDecodeStep(
                () -> {
                    Norms.rmsnorm(
                            s.normed,
                            0,
                            s.residual,
                            (long) row * dim,
                            weights.finalNorm(),
                            dim,
                            configuration.rmsNormEps);
                    MatMul.gemv(weights.wcls(), s.normed, s.logits);
                    return s.logits;
                });
    }

    /**
     * The raw LM logit of ONE token at the last ingested row, via the TIED token-embedding head
     * (Qwen3 small variants tie the LM head; reranker GGUFs carry no separate output.weight). A
     * reranker reads its two verdict tokens with two of these - two dot products, where a
     * generative head would project the whole vocabulary (~155 MB streamed at Q8) to reach them.
     * Holds the state and fences the model, like every other public entry point that runs kernels;
     * {@link #targetedHead} is the unfenced seam.
     */
    public float logit(State s, int token) {
        float out = s.exclusively(() -> targetedHead(s, token));
        Reference.reachabilityFence(this);
        return out;
    }

    private float targetedHead(State s, int token) {
        if (token < 0 || token >= configuration.vocabularySize())
            throw new IllegalArgumentException(
                    "token " + token + " outside [0," + configuration.vocabularySize() + ")");
        if (s.outputCount() == 0) throw new IllegalStateException("state has no retained output");
        int dim = configuration.embeddingLength;
        // the last retained row IS the last row of the chunk just ingested
        int row = s.lastBatchSize() - 1;
        Norms.rmsnorm(
                s.normed,
                0,
                s.residual,
                (long) row * dim,
                weights.finalNorm(),
                dim,
                configuration.rmsNormEps);
        // one row of the tied head: c[0] = tokenEmbeddings[token] . normed
        MatMul.gemm(
                weights.tokenEmbeddings,
                (long) token * dim,
                s.normed,
                dim,
                s.logits,
                configuration.vocabularySize,
                1,
                1,
                dim);
        return Views.getFloat(s.logits, 0, "logits");
    }

    /**
     * The sentence embedding: final-norm the retained row, then L2-normalize. LAST pooling means
     * {@code outputIndex} addresses the sequence's final row.
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
        MemoryView<MemorySegment> out = s.pooled;
        Norms.rmsnorm(
                out,
                0,
                s.residual,
                (long) row * dim,
                weights.finalNorm(),
                dim,
                configuration.rmsNormEps);
        float ss = Norms.sumOfSquares(out, 0, dim);
        float inv = ss > 0 ? (float) (1.0 / Math.sqrt(ss)) : 0f;
        Ops.mapInPlace(out, 0, dim, v -> v * inv);
        return out;
    }

    private static void requireOutput(State state, int output) {
        if (output < 0 || output >= state.outputCount())
            throw new IllegalArgumentException(
                    "output " + output + " outside [0," + state.outputCount() + ")");
    }

    // === Configuration ===

    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int vocabularySize,
            int contextLength,
            int hiddenDim,
            float rmsNormEps,
            float ropeTheta,
            int headSize,
            int ropeDim,
            int queryDim,
            int kvDim,
            int kvMul)
            implements ContextConfiguration {}

    // === State ===

    public static final class State extends ContextState {

        /** The residual stream every block adds back into. */
        final MemoryView<MemorySegment> residual;

        /**
         * Pre-norm output - the input of EVERY projection; second life as the FFN down-proj
         * destination before the residual add.
         */
        final MemoryView<MemorySegment> normed;

        /** The attention branch's output: o_proj destination, added to the residual. */
        final MemoryView<MemorySegment> branchOut;

        /** Flash-attention result, all heads concatenated, pre-o_proj. */
        final MemoryView<MemorySegment> attnOut;

        /** Q projection (per-head normed + roped in place). */
        final MemoryView<MemorySegment> query;

        /** FFN gate projection; post silu-multiply the gated hidden. */
        final MemoryView<MemorySegment> hidden;

        /** FFN up projection (the multiplicand consumed by siluMultiply). */
        final MemoryView<MemorySegment> hidden2;

        /** The LM head's output buffer. */
        final MemoryView<MemorySegment> logits;

        /** The pooled (final-normed, L2-normalized) embedding row - a REUSED buffer. */
        final MemoryView<MemorySegment> pooled;

        final MemoryView<MemorySegment> ropeCos, ropeSin;
        final FlashAttention.DecodeScratch decodeScratch;
        final MemoryView<MemorySegment> batchK, batchV; // this chunk's K/V (uniform kvDim)
        final MemoryView<MemorySegment>[] keyCache, valueCache; // per layer

        // per-row within-sequence RoPE positions + the chunk's per-sequence piece descriptors
        // (row0, len, kvStart, prior): ONE allocation each, refilled per forward (pieces <= rows)
        final int[] posOf, pieceRow0, pieceLen, pieceKv, piecePrior;

        State(
                Configuration config,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity > config.contextLength()) {
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds model contextLength "
                                + config.contextLength());
            }
            int c = batchCapacity;
            int dim = config.embeddingLength;
            this.residual = Views.allocateF32(memoryArena(), c, dim);
            this.normed = Views.allocateF32(memoryArena(), c, dim);
            this.branchOut = Views.allocateF32(memoryArena(), c, dim);
            this.attnOut = Views.allocateF32(memoryArena(), c, config.queryDim);
            this.query = Views.allocateF32(memoryArena(), c, config.queryDim);
            this.hidden = Views.allocateF32(memoryArena(), c, config.hiddenDim);
            this.hidden2 = Views.allocateF32(memoryArena(), c, config.hiddenDim);
            this.logits = Views.allocateF32(memoryArena(), 1, config.vocabularySize);
            this.pooled = Views.allocateF32(memoryArena(), 1, dim);
            // rotary values for the batch about to be ingested: sized by BATCH, never context
            this.ropeCos = Views.allocateF32(memoryArena(), c, config.ropeDim / 2);
            this.ropeSin = Views.allocateF32(memoryArena(), c, config.ropeDim / 2);
            this.decodeScratch = new FlashAttention.DecodeScratch(memoryArena());
            this.batchK = Views.allocateF32(memoryArena(), c, config.kvDim);
            this.batchV = Views.allocateF32(memoryArena(), c, config.kvDim);
            int n = config.numberOfLayers;
            this.keyCache = new MemoryView[n];
            this.valueCache = new MemoryView[n];
            for (int l = 0; l < n; l++) {
                keyCache[l] = Views.allocateF16(memoryArena(), contextCapacity, config.kvDim);
                valueCache[l] = Views.allocateF16(memoryArena(), contextCapacity, config.kvDim);
            }
            this.posOf = new int[c];
            this.pieceRow0 = new int[c];
            this.pieceLen = new int[c];
            this.pieceKv = new int[c];
            this.piecePrior = new int[c];
        }

        /**
         * Recycles this allocation for a fresh sequence: cursor to 0. Pure attention carries
         * nothing but KV across positions, and stale rows beyond the cursor are attention-masked
         * (then overwritten) - a cursor move suffices, nothing to zero.
         */
        @Override
        protected void clearHistory() {}

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }
    }

    // === Weights ===

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> wq,
            MemoryView<MemorySegment> wk,
            MemoryView<MemorySegment> wv,
            MemoryView<MemorySegment> wo,
            MemoryView<MemorySegment> attnQNorm,
            MemoryView<MemorySegment> attnKNorm,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> w1,
            MemoryView<MemorySegment> w2,
            MemoryView<MemorySegment> w3) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbeddings,
            LayerWeights[] layers,
            MemoryView<MemorySegment> finalNorm,
            RoPE.Schedule rope,
            MemoryView<MemorySegment> wcls) {}

    public static Qwen3 loadModel(Path ggufPath, Arena arena) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, arena);
        }
    }

    public static Qwen3 loadModel(FileChannel fileChannel, GGUF gguf, Arena arena)
            throws IOException {
        return loadModel(fileChannel, gguf, arena, null);
    }

    public static Qwen3 loadModel(
            FileChannel fileChannel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration config = loadConfiguration(gguf, tokenizer);
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(fileChannel, gguf, arena);
        return new Qwen3(config, tokenizer, loadWeights(tensors, config));
    }

    private static Configuration loadConfiguration(GGUF gguf, Tokenizer tokenizer) {
        String arch = "qwen3";
        int dim = gguf.getValueOrDefault(int.class, arch + ".embedding_length", 0);
        int nLayers = gguf.getValueOrDefault(int.class, arch + ".block_count", 0);
        int nHeads = gguf.getValueOrDefault(int.class, arch + ".attention.head_count", 0);
        int nKvHeads = gguf.getValueOrDefault(int.class, arch + ".attention.head_count_kv", nHeads);
        int contextLength = gguf.getValueOrDefault(int.class, arch + ".context_length", 0);
        int hiddenDim = gguf.getValueOrDefault(int.class, arch + ".feed_forward_length", 0);
        float rmsNormEps =
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10000f);
        int headSize =
                gguf.getValueOrDefault(int.class, arch + ".attention.key_length", dim / nHeads);
        int ropeDim = gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);
        int queryDim = nHeads * headSize;
        int kvDim = nKvHeads * headSize;
        int kvMul = nHeads / nKvHeads;
        return new Configuration(
                dim,
                nLayers,
                nHeads,
                nKvHeads,
                tokenizer.vocabulary().size(),
                contextLength,
                hiddenDim,
                rmsNormEps,
                ropeTheta,
                headSize,
                ropeDim,
                queryDim,
                kvDim,
                kvMul);
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors, Configuration config) {
        int n = config.numberOfLayers;
        // rope_freqs.weight, when a converter ships it, overrides the computed schedule (the old
        // fromMeta's tensor arm; the metadata-array fallback never fired for Qwen3 and is dropped)
        RoPE.Schedule rope =
                ModelLoader.ropeFreqFactors(tensors)
                        .map(
                                freqs ->
                                        RoPE.withFreqFactors(
                                                config.headSize, config.ropeTheta, freqs))
                        .orElseGet(() -> RoPE.plain(config.headSize, config.ropeTheta));

        MemoryView<MemorySegment> tokenEmbeddings =
                ModelLoader.require(tensors, "token_embd.weight");
        MemoryView<MemorySegment> wcls =
                ModelLoader.find(tensors, "output.weight").orElse(tokenEmbeddings);
        MemoryView<MemorySegment> finalNorm = ModelLoader.requireF32(tensors, "output_norm.weight");

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            layers[i] =
                    new LayerWeights(
                            ModelLoader.requireF32(tensors, p + "attn_norm.weight"),
                            ModelLoader.require(tensors, p + "attn_q.weight"),
                            ModelLoader.require(tensors, p + "attn_k.weight"),
                            ModelLoader.require(tensors, p + "attn_v.weight"),
                            ModelLoader.require(tensors, p + "attn_output.weight"),
                            ModelLoader.requireF32(tensors, p + "attn_q_norm.weight"),
                            ModelLoader.requireF32(tensors, p + "attn_k_norm.weight"),
                            ModelLoader.requireF32(tensors, p + "ffn_norm.weight"),
                            ModelLoader.require(tensors, p + "ffn_gate.weight"),
                            ModelLoader.require(tensors, p + "ffn_down.weight"),
                            ModelLoader.require(tensors, p + "ffn_up.weight"));
        }
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls);
    }
}
