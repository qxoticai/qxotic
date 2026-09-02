// Llama's RoPE GQA transformer and metadata-compatible variants:
//   - Llama 3.x: "llama3" RoPE frequency scaling (rope_freqs.weight).
//   - MiniCPM:   embedding_scale / residual_scale / logit_scale, including legacy defaults.
//   - Mistral-3: YaRN RoPE scaling + Llama-4-style attention temperature tuning.
//   - SmolLM3:   NoPE - RoPE is skipped on every 4th layer (noRopeLayerStep); otherwise plain
//     Llama.
// Text-only with interleaved RoPE, an F16 KV cache, and a lazy last-layer tail.
package com.qxotic.jinfer.models.llama;

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
import java.util.Optional;

/**
 * Llama-family GGUF text models, including Llama 3.x, MiniCPM, Mistral 3, and SmolLM3, dispatched
 * by the checkpoint's {@code general.architecture}.
 */
public final class Llama implements LanguageModel<Llama.Configuration, Llama.Weights, Llama.State> {

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;

    Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {
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
        return Optional.of(new LlamaCheckpointCodec(configuration));
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
        if (n <= 0) throw new IllegalArgumentException("Llama token batch must not be empty");
        if (n > s.batchCapacity()) {
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + s.batchCapacity());
        }
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
        int[] ids =
                switch (batch.input()) {
                    case Batch.Input.Tokens t -> t.ids();
                    case Batch.Input.Sequences seq ->
                            throw new UnsupportedOperationException(
                                    "Llama is generative: packed sequences (batched embedding) not"
                                            + " supported");
                    case Batch.Input.Embeddings ignored ->
                            throw new UnsupportedOperationException(
                                    "Llama is text-only: embedding input is not supported");
                };
        for (int id : ids)
            if (id < 0 || id >= configuration.vocabularySize)
                throw new IllegalArgumentException(
                        "token id " + id + " outside [0," + configuration.vocabularySize + ")");
        forward(s, ids, from, n);
        s.advance(batch);
    }

    @Override
    public MemoryView<?> logits(State s, int output) {
        MemoryView<?> result = s.exclusively(() -> projectLogits(s, output));
        Reference.reachabilityFence(this);
        return result;
    }

    private MemoryView<?> projectLogits(State s, int output) {
        if (output < 0 || output >= s.outputCount())
            throw new IllegalArgumentException("output " + output + " outside retained outputs");
        Views.checkAlive(weights.finalNorm(), "finalNorm");
        int dim = configuration.embeddingLength;
        int row = s.lastBatchSize() - s.outputCount() + output;
        {
            tailAt(s, row); // finish the deferred last-layer tail for this row -> s.th
            Norms.rmsnorm(s.normed, 0, s.th, 0, weights.finalNorm(), dim, configuration.rmsNormEps);
            MatMul.gemv(weights.wcls(), s.normed, s.logits);
            float ls = configuration.logitScale;
            if (ls != 1.0f) {
                Ops.divideInPlace(s.logits, 0, configuration.vocabularySize, ls);
            }
            return s.logits;
        }
    }

    // === Forward ===

    private void forward(State state, int[] tokens, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        // ONCE for the batch: an angle depends on the position and the schedule, never on the
        // layer, so all of them read these rows
        RoPE.fill(state.ropeCos, state.ropeSin, startPos, seqLen, config.ropeHalf(), w.rope());
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;
        float embScale = config.embeddingScale, residScale = config.residualScale;

        Views.checkAlive(w.tokenEmbeddings(), "tokenEmbeddings"); // fail-fast on freed weights
        Convert.gatherToF32(w.tokenEmbeddings(), tokens, 0, seqLen, state.residual, 0, dim);
        if (embScale != 1.0f) {
            Ops.multiplyInPlace(state.residual, 0, Math.multiplyExact(seqLen, dim), embScale);
        }

        int lastLayer = config.numberOfLayers - 1;
        for (int l = 0; l < lastLayer; l++) {
            LayerWeights lw = w.layers()[l];
            Norms.rmsnormRows(state.normed, state.residual, lw.attnNorm(), seqLen, dim, eps);
            attention(state, l, startPos, seqLen);
            Ops.addScaled(state.residual, state.normed, seqLen * dim, residScale);
            Norms.rmsnormRows(state.normed, state.residual, lw.ffnNorm(), seqLen, dim, eps);
            feedForward(state, l, seqLen);
            Ops.addScaled(state.residual, state.normed, seqLen * dim, residScale);
            if (Trace.ENABLED) {
                Trace.sum("l_out-" + l, state.residual, seqLen * dim);
            }
        }
        // Lazy last-layer split: write every row's K/V into the cache (so any row can attend
        // later), but DEFER the attention + FFN tail. state.residual is left as the last-layer
        // INPUT residual; a query finishes exactly the rows it asks for via tailAt() in head().
        // Prefill pays nothing for the tail here; the saving is the last layer's attention+FFN
        // skipped for every un-queried row.
        writeKv(state, lastLayer, startPos, seqLen);
    }

    /**
     * Commit this chunk's F32 K/V (state.batchK/batchV) into the F16 cache at {@code [startPos,
     * startPos+seqLen)}. Called INSIDE the layer: the batch buffers are single allocations reused
     * by every layer, so a deferred commit would write the LAST layer's values everywhere (the Qwen
     * 3 bug, ported as law).
     */
    private void commitKv(State state, int l, int startPos, int seqLen) {
        int kvDim = configuration.kvDim();
        int count = Math.multiplyExact(seqLen, kvDim);
        long cacheOffset = Math.multiplyExact((long) startPos, kvDim);
        Convert.f32ToF16(state.batchK, 0, state.keyCache[l], cacheOffset, count);
        Convert.f32ToF16(state.batchV, 0, state.valueCache[l], cacheOffset, count);
    }

    /**
     * Last-layer K/V half: pre-norm all rows, project + RoPE (NoPE-aware) K, and commit K/V to the
     * cache. No Q, no attention, no O, no FFN - and state.residual is left untouched (the
     * last-layer input residual).
     */
    private void writeKv(State state, int l, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        LayerWeights lw = w.layers()[l];
        int dim = config.embeddingLength, kvDim = config.kvDim(), headSize = config.headSize;
        int kvHeads = config.numberOfKeyValueHeads, ropeHalf = config.ropeHalf();
        float eps = config.rmsNormEps;
        Norms.rmsnormRows(state.normed, state.residual, lw.attnNorm(), seqLen, dim, eps);
        MatMul.gemm(lw.wk(), state.normed, state.batchK, seqLen);
        MatMul.gemm(lw.wv(), state.normed, state.batchV, seqLen);
        addBias(state.batchK, lw.bk(), seqLen, kvDim);
        addBias(state.batchV, lw.bv(), seqLen, kvDim);
        if (config.useRope(l)) {
            Parallel.forLoop(
                    seqLen,
                    s -> {
                        for (int h = 0; h < kvHeads; h++) {
                            RoPE.applyInterleaved(
                                    state.batchK,
                                    (long) s * kvDim + h * headSize,
                                    s,
                                    state.ropeCos,
                                    state.ropeSin,
                                    ropeHalf);
                        }
                    });
        }
        commitKv(state, l, startPos, seqLen); // the tail reads every row from the F16 cache
    }

    /**
     * Lazy tail: finish the last layer for retained chunk-row {@code i} into state.th, reading its
     * input from state.residual[i] and attending cache[0..pos] inclusive - a single causal query
     * aimed at row i (its own K/V is already in the F16 cache from writeKv, read like any other
     * position). state.residual is never written.
     */
    private void tailAt(State state, int i) {
        Configuration config = configuration;
        Weights w = weights;
        int L = config.numberOfLayers - 1;
        LayerWeights lw = w.layers()[L];
        int dim = config.embeddingLength, kvDim = config.kvDim(), queryDim = config.queryDim();
        int heads = config.numberOfHeads,
                headSize = config.headSize,
                kvMul = heads / config.numberOfKeyValueHeads;
        int ropeHalf = config.ropeHalf();
        float eps = config.rmsNormEps,
                residScale = config.residualScale,
                attScale = config.attentionScale();
        int startPos = state.position() - state.lastBatchSize(); // global position of chunk row 0
        int pos = startPos + i; // global position of row i

        // pre-norm reads residual[i] directly (read-only)
        Norms.rmsnorm(
                state.tailScratch, 0, state.residual, (long) i * dim, lw.attnNorm(), dim, eps);
        // Q for this row (query is free scratch outside a forward)
        MatMul.gemm(lw.wq(), state.tailScratch, state.query, 1);
        addBias(state.query, lw.bq(), 1, queryDim);
        if (config.useRope(L)) {
            // the lazy tail runs outside any ingest, so it fills row 0 for its own position
            // rather than trusting whatever range the last fill covered
            RoPE.fill(state.ropeCos, state.ropeSin, pos, 1, ropeHalf, w.rope());
            for (int h = 0; h < heads; h++) {
                RoPE.applyInterleaved(
                        state.query,
                        (long) h * headSize,
                        0,
                        state.ropeCos,
                        state.ropeSin,
                        ropeHalf);
            }
        }
        float aScale = config.attnTemp(pos);
        if (aScale != 1.0f) {
            Ops.multiplyInPlace(state.query, 0, queryDim, aScale);
        }
        // Single causal query over cache[0..pos] INCLUSIVE (batchK/batchV = null): row i's own
        // K/V is already in the F16 cache from writeKv, read like every other position.
        FlashAttention.flashDecode(
                state.query,
                state.attnOut,
                state.keyCache[L],
                state.valueCache[L],
                null,
                null,
                heads,
                pos,
                0,
                headSize,
                kvDim,
                kvMul,
                attScale,
                0,
                null,
                state.decodeScratch);
        MatMul.gemm(lw.wo(), state.attnOut, state.tailScratch, 1);
        addBias(state.tailScratch, lw.bo(), 1, dim);
        // th = residual[i] + residScale*O (born, no seed copy)
        Ops.addScaledInto(
                state.th, state.residual, (long) i * dim, state.tailScratch, dim, residScale);
        Norms.rmsnorm(state.tailScratch, 0, state.th, 0, lw.ffnNorm(), dim, eps);
        feedForwardRow(state, L, state.tailScratch); // SwiGLU, one row, in place on tailScratch
        Ops.addScaled(state.th, state.tailScratch, dim, residScale); // FFN residual -> th finished
    }

    /**
     * Standard RoPE GQA attention: Q/K/V projections, per-row interleaved RoPE (+ optional
     * attn-temp), K/V committed to the contiguous cache, then causal flash attention (or
     * single-token decode), output projection written back to {@code state.normed}.
     */
    private void attention(State state, int layer, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        LayerWeights lw = w.layers()[layer];
        int dim = config.embeddingLength;
        int headSize = config.headSize;
        int heads = config.numberOfHeads;
        int kvHeads = config.numberOfKeyValueHeads;
        int kvDim = config.kvDim();
        int queryDim = config.queryDim();
        int kvMul = heads / kvHeads;
        int ropeHalf = config.ropeHalf();

        MatMul.gemm(lw.wq(), state.normed, state.query, seqLen);
        MatMul.gemm(lw.wk(), state.normed, state.batchK, seqLen);
        MatMul.gemm(lw.wv(), state.normed, state.batchV, seqLen);
        addBias(state.query, lw.bq(), seqLen, queryDim);
        addBias(state.batchK, lw.bk(), seqLen, kvDim);
        addBias(state.batchV, lw.bv(), seqLen, kvDim);
        boolean useRope = config.useRope(layer); // SmolLM3 NoPE: some layers skip RoPE entirely
        Parallel.forLoop(
                seqLen,
                s -> {
                    if (useRope) {
                        for (int h = 0; h < heads; h++) {
                            RoPE.applyInterleaved(
                                    state.query,
                                    (long) s * queryDim + h * headSize,
                                    s,
                                    state.ropeCos,
                                    state.ropeSin,
                                    ropeHalf);
                        }
                        for (int h = 0; h < kvHeads; h++) {
                            RoPE.applyInterleaved(
                                    state.batchK,
                                    (long) s * kvDim + h * headSize,
                                    s,
                                    state.ropeCos,
                                    state.ropeSin,
                                    ropeHalf);
                        }
                    }
                    float aScale = config.attnTemp(startPos + s);
                    if (aScale != 1.0f) {
                        Ops.multiplyInPlace(state.query, (long) s * queryDim, queryDim, aScale);
                    }
                });

        MemoryView<MemorySegment> keyCache = state.keyCache[layer],
                valueCache = state.valueCache[layer];
        float attScale = config.attentionScale();
        // Full causal attention over cache[0..startPos) + this chunk's F32 K/V; window=0.
        if (seqLen == 1) {
            FlashAttention.flashDecode(
                    state.query,
                    state.attnOut,
                    keyCache,
                    valueCache,
                    state.batchK,
                    state.batchV,
                    heads,
                    startPos,
                    0,
                    headSize,
                    kvDim,
                    kvMul,
                    attScale,
                    0,
                    null,
                    state.decodeScratch);
        } else {
            FlashAttention.slidingWindowPrefill(
                    state.query,
                    state.attnOut,
                    keyCache,
                    valueCache,
                    state.batchK,
                    state.batchV,
                    heads,
                    startPos,
                    seqLen,
                    headSize,
                    kvDim,
                    queryDim,
                    kvDim,
                    kvMul,
                    attScale,
                    0,
                    0,
                    null);
        }
        commitKv(state, layer, startPos, seqLen);
        MatMul.gemm(lw.wo(), state.attnOut, state.normed, seqLen);
        addBias(state.normed, lw.bo(), seqLen, dim);
    }

    /** Dense SwiGLU FFN over the pre-normed rows in {@code state.normed}, written back in place. */
    private void feedForward(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        LayerWeights lw = weights.layers()[l];
        MatMul.gemm(lw.w1(), state.normed, state.hidden, seqLen);
        MatMul.gemm(lw.w3(), state.normed, state.hidden2, seqLen);
        addBias(state.hidden, lw.b1(), seqLen, hiddenDim);
        addBias(state.hidden2, lw.b3(), seqLen, hiddenDim);
        Activations.siluMultiply(
                state.hidden, 0, state.hidden2, 0, Math.multiplyExact(seqLen, hiddenDim));
        MatMul.gemm(lw.w2(), state.hidden, state.normed, seqLen);
        addBias(state.normed, lw.b2(), seqLen, dim);
    }

    /** The one-row FFN of the lazy tail, over {@code io} in place (gate into hidden/hidden2). */
    private void feedForwardRow(State state, int l, MemoryView<MemorySegment> io) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        LayerWeights lw = weights.layers()[l];
        MatMul.gemm(lw.w1(), io, state.hidden, 1);
        MatMul.gemm(lw.w3(), io, state.hidden2, 1);
        addBias(state.hidden, lw.b1(), 1, hiddenDim);
        addBias(state.hidden2, lw.b3(), 1, hiddenDim);
        Activations.siluMultiply(state.hidden, 0, state.hidden2, 0, hiddenDim);
        MatMul.gemm(lw.w2(), state.hidden, io, 1);
        addBias(io, lw.b2(), 1, dim);
    }

    private static void addBias(
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> bias,
            int rows,
            int columns) {
        if (bias != null) Ops.addRowBiasInPlace(output, 0, bias, 0, rows, columns);
    }

    // === Configuration ===

    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int headSize,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            float ropeTheta,
            int ropeDimensionCount,
            int hiddenDim,
            float embeddingScale,
            float residualScale,
            float logitScale,
            float attnTempScale,
            int attnTempFloorScale,
            float attentionScaleValue,
            int noRopeLayerStep)
            implements ContextConfiguration {
        public int queryDim() {
            return numberOfHeads * headSize;
        }

        public int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        public int ropeHalf() {
            return Math.min(ropeDimensionCount, headSize) / 2;
        }

        public float attentionScale() {
            return attentionScaleValue != 0f
                    ? attentionScaleValue
                    : 1.0f / (float) Math.sqrt(headSize);
        }

        /**
         * Llama-4 / Mistral-3 attention temperature tuning; 1.0 (no-op) below the floor, 0 =
         * disabled.
         */
        public float attnTemp(int position) {
            if (attnTempScale == 0f || attnTempFloorScale <= 0) return 1.0f;
            return (float)
                    (Math.log(Math.floor((double) position / attnTempFloorScale) + 1.0)
                                    * attnTempScale
                            + 1.0);
        }

        /**
         * SmolLM3 NoPE: RoPE is skipped on every {@code noRopeLayerStep}-th layer (1-indexed); 0 =
         * always RoPE.
         */
        public boolean useRope(int layer) {
            return noRopeLayerStep <= 0 || (layer + 1) % noRopeLayerStep != 0;
        }
    }

    // === Weights ===

    public record LayerWeights(
            MemoryView<MemorySegment> attnNorm,
            MemoryView<MemorySegment> wq,
            MemoryView<MemorySegment> bq,
            MemoryView<MemorySegment> wk,
            MemoryView<MemorySegment> bk,
            MemoryView<MemorySegment> wv,
            MemoryView<MemorySegment> bv,
            MemoryView<MemorySegment> wo,
            MemoryView<MemorySegment> bo,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> w1,
            MemoryView<MemorySegment> b1,
            MemoryView<MemorySegment> w2,
            MemoryView<MemorySegment> b2,
            MemoryView<MemorySegment> w3,
            MemoryView<MemorySegment> b3) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbeddings,
            LayerWeights[] layers,
            MemoryView<MemorySegment> finalNorm,
            RoPE.Schedule rope,
            MemoryView<MemorySegment> wcls) {}

    // === State ===

    public static final class State extends ContextState {

        /** The residual stream every block adds back into. */
        final MemoryView<MemorySegment> residual;

        /** Pre-norm output - the input of EVERY projection; second life as the FFN down dest. */
        final MemoryView<MemorySegment> normed;

        /** Q projection (roped in place per row). */
        final MemoryView<MemorySegment> query;

        /** Flash-attention result, all heads concatenated, pre-o_proj. */
        final MemoryView<MemorySegment> attnOut;

        /** FFN gate projection; post silu-multiply the gated hidden. */
        final MemoryView<MemorySegment> hidden;

        /** FFN up projection (the multiplicand consumed by siluMultiply). */
        final MemoryView<MemorySegment> hidden2;

        /** The LM head's output buffer. */
        final MemoryView<MemorySegment> logits;

        /**
         * Lazy last-layer tail: single-row finished hidden + scratch, kept DISTINCT from the batch
         * buffers so residual/batchK/batchV stay read-only across queries (any retained row can be
         * finished, in any order, repeatedly).
         */
        final MemoryView<MemorySegment> th, tailScratch;

        final MemoryView<MemorySegment> ropeCos, ropeSin;
        final FlashAttention.DecodeScratch decodeScratch;
        final MemoryView<MemorySegment> batchK, batchV; // this chunk's K/V (uniform kvDim)
        final MemoryView<MemorySegment>[] keyCache, valueCache; // per layer, F16

        /** Recycles this allocation: cursor to 0; stale KV rows beyond it are attention-masked. */
        @Override
        protected void clearHistory() {}

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }

        @SuppressWarnings("unchecked")
        State(
                Configuration config,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity <= 0 || contextCapacity > config.contextLength()) {
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " outside [1,"
                                + config.contextLength()
                                + "]");
            }
            if (batchCapacity <= 0)
                throw new IllegalArgumentException("batchCapacity " + batchCapacity);
            int dim = config.embeddingLength;
            int queryDim = config.queryDim();
            int kvDim = config.kvDim();
            int hidden = config.hiddenDim;
            this.residual = Views.allocateF32(memoryArena(), batchCapacity, dim);
            this.normed = Views.allocateF32(memoryArena(), batchCapacity, dim);
            this.batchK = Views.allocateF32(memoryArena(), batchCapacity, kvDim);
            this.batchV = Views.allocateF32(memoryArena(), batchCapacity, kvDim);
            this.query = Views.allocateF32(memoryArena(), batchCapacity, queryDim);
            this.attnOut = Views.allocateF32(memoryArena(), batchCapacity, queryDim);
            this.hidden = Views.allocateF32(memoryArena(), batchCapacity, hidden);
            this.hidden2 = Views.allocateF32(memoryArena(), batchCapacity, hidden);
            this.logits = Views.allocateF32(memoryArena(), 1, config.vocabularySize);
            this.th = Views.allocateF32(memoryArena(), 1, dim);
            this.tailScratch = Views.allocateF32(memoryArena(), 1, dim);
            // rotary values for the batch about to be ingested: sized by BATCH, never context
            this.ropeCos = Views.allocateF32(memoryArena(), batchCapacity, config.ropeHalf());
            this.ropeSin = Views.allocateF32(memoryArena(), batchCapacity, config.ropeHalf());
            this.decodeScratch = new FlashAttention.DecodeScratch(memoryArena());
            this.keyCache = new MemoryView[config.numberOfLayers];
            this.valueCache = new MemoryView[config.numberOfLayers];
            for (int l = 0; l < config.numberOfLayers; l++) {
                keyCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
                valueCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
            }
        }
    }

    // === Loading ===

    public static Llama loadModel(Path ggufPath, Arena arena) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, arena);
        }
    }

    public static Llama loadModel(FileChannel fileChannel, GGUF gguf, Arena arena)
            throws IOException {
        return loadModel(fileChannel, gguf, arena, null);
    }

    /**
     * As above with a caller-supplied tokenizer; null = the GGUF's own (the differential gate feeds
     * ONE tokenizer instance to both trees).
     */
    public static Llama loadModel(
            FileChannel fileChannel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        if (tokenizer == null) {
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        }
        String arch = gguf.getString("general.architecture");
        Configuration config = readConfiguration(gguf, arch, tokenizer.vocabulary().size());

        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(fileChannel, gguf, arena);
        RoPE.Schedule rope = buildRope(gguf, arch, config, tensors);
        return new Llama(config, tokenizer, loadWeights(tensors, config, rope));
    }

    static Configuration readConfiguration(GGUF gguf, String arch, int vocabularySize) {
        require(
                arch.equals("llama")
                        || arch.equals("minicpm")
                        || arch.equals("mistral3")
                        || arch.equals("smollm3"),
                "unsupported architecture '" + arch + "'");

        int contextLength = gguf.getValue(int.class, arch + ".context_length");
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        require(
                contextLength > 0
                        && embeddingLength > 0
                        && numberOfLayers > 0
                        && numberOfHeads > 0
                        && vocabularySize > 0,
                "invalid core dimensions");
        int numberOfKeyValueHeads =
                gguf.getValueOrDefault(int.class, arch + ".attention.head_count_kv", numberOfHeads);
        String keyLengthKey = arch + ".attention.key_length";
        int headSize =
                gguf.getValueOrDefault(int.class, keyLengthKey, embeddingLength / numberOfHeads);
        int valueHeadSize =
                gguf.getValueOrDefault(int.class, arch + ".attention.value_length", headSize);
        int hiddenDim = gguf.getValue(int.class, arch + ".feed_forward_length");
        float rmsNormEps =
                gguf.getValueOrDefault(
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10000f);
        int ropeDimensionCount =
                gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);

        boolean isMiniCpm = arch.equals("minicpm");
        float embeddingScale =
                gguf.getValueOrDefault(
                        float.class, arch + ".embedding_scale", isMiniCpm ? 12.0f : 1.0f);
        float residualScale =
                gguf.getValueOrDefault(
                        float.class,
                        arch + ".residual_scale",
                        isMiniCpm ? (float) (1.4 / Math.sqrt(numberOfLayers)) : 1.0f);
        float logitScale =
                gguf.getValueOrDefault(
                        float.class,
                        arch + ".logit_scale",
                        isMiniCpm ? embeddingLength / 256.0f : 1.0f); // dim_model_base = 256

        float attnTempScale =
                gguf.getValueOrDefault(float.class, arch + ".attention.temperature_scale", 0f);
        int attnTempFloorScale =
                gguf.getValueOrDefault(
                        int.class, arch + ".rope.scaling.original_context_length", 0);
        float attentionScale = gguf.getValueOrDefault(float.class, arch + ".attention.scale", 0f);
        int noRopeLayerStep =
                arch.equals("smollm3")
                        ? 4
                        : 0; // SmolLM3 NoPE: skip RoPE on every 4th layer (llama.cpp hardcodes 4)

        Configuration config =
                new Configuration(
                        embeddingLength,
                        numberOfLayers,
                        numberOfHeads,
                        numberOfKeyValueHeads,
                        headSize,
                        vocabularySize,
                        contextLength,
                        rmsNormEps,
                        ropeTheta,
                        ropeDimensionCount,
                        hiddenDim,
                        embeddingScale,
                        residualScale,
                        logitScale,
                        attnTempScale,
                        attnTempFloorScale,
                        attentionScale,
                        noRopeLayerStep);
        require(
                numberOfKeyValueHeads > 0
                        && numberOfHeads % numberOfKeyValueHeads == 0
                        && (gguf.containsKey(keyLengthKey) || embeddingLength % numberOfHeads == 0)
                        && headSize > 0
                        && valueHeadSize == headSize
                        && hiddenDim > 0,
                "invalid or unsupported attention/FFN dimensions");
        require(
                ropeDimensionCount > 0
                        && (ropeDimensionCount & 1) == 0
                        && ropeDimensionCount <= headSize,
                "invalid RoPE dimensions");
        require(
                rmsNormEps > 0f
                        && Float.isFinite(rmsNormEps)
                        && ropeTheta > 0f
                        && Float.isFinite(ropeTheta),
                "invalid normalization or RoPE metadata");
        require(
                embeddingScale > 0f
                        && Float.isFinite(embeddingScale)
                        && residualScale > 0f
                        && Float.isFinite(residualScale)
                        && logitScale > 0f
                        && Float.isFinite(logitScale),
                "invalid model scaling metadata");
        require(
                attnTempScale >= 0f
                        && Float.isFinite(attnTempScale)
                        && (attnTempScale == 0f || attnTempFloorScale > 0)
                        && (attentionScale == 0f
                                || attentionScale > 0f && Float.isFinite(attentionScale)),
                "invalid attention scaling metadata");
        require(
                gguf.getValueOrDefault(int.class, arch + ".vocab_size", vocabularySize)
                        == vocabularySize,
                "tokenizer vocabulary does not match the model");
        require(
                gguf.getValueOrDefault(int.class, arch + ".expert_count", 0) == 0,
                "MoE checkpoints are not supported");
        require(
                (long) numberOfHeads * headSize <= Integer.MAX_VALUE
                        && (long) numberOfKeyValueHeads * headSize <= Integer.MAX_VALUE,
                "attention dimensions overflow");
        return config;
    }

    /**
     * RoPE flavor from GGUF metadata: YaRN (mistral3), "llama3" per-frequency scaling
     * (rope_freqs.weight), or plain RoPE (Llama/MiniCPM). Returns the interleaved cos/sin schedule
     * applyInterleaved consumes.
     */
    static RoPE.Schedule buildRope(
            GGUF gguf,
            String arch,
            Configuration config,
            Map<String, MemoryView<MemorySegment>> tensors) {
        int ropeDim = config.ropeDimensionCount;
        String scalingType = gguf.getValueOrDefault(String.class, arch + ".rope.scaling.type", "");
        Optional<float[]> factors = ModelLoader.ropeFreqFactors(tensors);
        if (scalingType.equals("yarn")) {
            require(factors.isEmpty(), "rope_freqs.weight cannot be combined with YaRN");
            float factor = gguf.getValue(float.class, arch + ".rope.scaling.factor");
            int origCtx = gguf.getValue(int.class, arch + ".rope.scaling.original_context_length");
            float betaFast =
                    gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_fast", 32f);
            float betaSlow =
                    gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_slow", 1f);
            float logMul =
                    gguf.getValueOrDefault(
                            float.class, arch + ".rope.scaling.yarn_log_multiplier", 0f);
            require(
                    factor > 0f
                            && Float.isFinite(factor)
                            && origCtx > 0
                            && betaFast > 0f
                            && Float.isFinite(betaFast)
                            && betaSlow > 0f
                            && Float.isFinite(betaSlow)
                            && Float.isFinite(logMul),
                    "invalid YaRN metadata");
            // llama.cpp net amplitude: get_mscale(f,1)/get_mscale(f,logMul), with
            // get_mscale(f,m) = f<=1 ? 1 : 1+0.1·m·ln f  (logMul=0 → denominator 1). RoPE.yarn
            // multiplies attnFactor by ggml's internal (1+0.1 ln f), so divide that back out.
            float lnF = (float) Math.log(factor);
            float mscale1 = factor <= 1f ? 1f : 1f + 0.1f * lnF;
            float mscaleAll = factor <= 1f ? 1f : 1f + 0.1f * logMul * lnF;
            float attnFactor = mscale1 / mscaleAll / (1f + 0.1f * lnF);
            require(attnFactor > 0f && Float.isFinite(attnFactor), "invalid YaRN amplitude");
            return RoPE.yarn(
                    ropeDim, config.ropeTheta, factor, origCtx, betaFast, betaSlow, 1f, attnFactor);
        }
        require(
                scalingType.isEmpty()
                        || scalingType.equals("none")
                        || scalingType.equals("linear")
                        || scalingType.equals("llama3"),
                "unsupported rope.scaling.type '" + scalingType + "'");
        require(
                !scalingType.equals("llama3") || factors.isPresent(),
                "llama3 RoPE requires rope_freqs.weight");

        float factor = gguf.getValueOrDefault(float.class, arch + ".rope.scaling.factor", 0f);
        factor = scalingType.equals("none") || factor == 0f ? 1f : factor;
        require(factor > 0f && Float.isFinite(factor), "invalid RoPE scaling factor");

        if (factors.isEmpty() && factor == 1f) return RoPE.plain(ropeDim, config.ropeTheta);
        float[] values = factors.orElseGet(() -> new float[ropeDim / 2]);
        require(values.length == ropeDim / 2, "rope_freqs.weight has the wrong length");
        for (int i = 0; i < values.length; i++) {
            if (factors.isEmpty()) values[i] = factor;
            else values[i] *= factor;
            require(
                    values[i] > 0f && Float.isFinite(values[i]),
                    "rope_freqs.weight contains invalid data");
        }
        return RoPE.withFreqFactors(ropeDim, config.ropeTheta, values);
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors,
            Configuration config,
            RoPE.Schedule rope) {
        int n = config.numberOfLayers;
        MemoryView<MemorySegment> tokenEmbeddings =
                ModelLoader.require(tensors, "token_embd.weight");
        MemoryView<MemorySegment> wcls =
                ModelLoader.find(tensors, "output.weight").orElse(tokenEmbeddings);
        MemoryView<MemorySegment> finalNorm = ModelLoader.requireF32(tensors, "output_norm.weight");

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            require(
                    !tensors.containsKey(p + "attn_qkv.weight"),
                    "fused QKV checkpoints are not supported");
            layers[i] =
                    new LayerWeights(
                            ModelLoader.requireF32(tensors, p + "attn_norm.weight"),
                            ModelLoader.require(tensors, p + "attn_q.weight"),
                            optionalBias(tensors, p + "attn_q.bias", config.queryDim()),
                            ModelLoader.require(tensors, p + "attn_k.weight"),
                            optionalBias(tensors, p + "attn_k.bias", config.kvDim()),
                            ModelLoader.require(tensors, p + "attn_v.weight"),
                            optionalBias(tensors, p + "attn_v.bias", config.kvDim()),
                            ModelLoader.require(tensors, p + "attn_output.weight"),
                            optionalBias(tensors, p + "attn_output.bias", config.embeddingLength),
                            ModelLoader.requireF32(tensors, p + "ffn_norm.weight"),
                            ModelLoader.require(tensors, p + "ffn_gate.weight"),
                            optionalBias(tensors, p + "ffn_gate.bias", config.hiddenDim),
                            ModelLoader.require(tensors, p + "ffn_down.weight"),
                            optionalBias(tensors, p + "ffn_down.bias", config.embeddingLength),
                            ModelLoader.require(tensors, p + "ffn_up.weight"),
                            optionalBias(tensors, p + "ffn_up.bias", config.hiddenDim));
        }
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls);
    }

    private static MemoryView<MemorySegment> optionalBias(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int width) {
        MemoryView<MemorySegment> bias = ModelLoader.findF32(tensors, name).orElse(null);
        if (bias != null) require(bias.shape().size() == width, name + " has the wrong length");
        return bias;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Llama: " + message);
    }
}
