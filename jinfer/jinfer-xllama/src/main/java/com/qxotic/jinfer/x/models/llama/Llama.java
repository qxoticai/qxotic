// The standard Llama transformer (RoPE GQA attention + SwiGLU FFN + RMSNorm) against the x
// boundary: a port of jinfer-llama's Llama (cycle 3 of the FloatTensor migration), covering the
// "llama" GGUF architecture and its same-graph relatives (all distinguished by GGUF metadata, no
// extra classes):
//   - Llama 3.x: "llama3" RoPE frequency scaling (rope_freqs.weight).
//   - MiniCPM:   embedding_scale / residual_scale / logit_scale (default 1.0 -> plain Llama).
//   - Mistral-3: YaRN RoPE scaling + Llama-4-style attention temperature tuning.
//   - SmolLM3:   NoPE - RoPE is skipped on every 4th layer (noRopeLayerStep); otherwise plain
//     Llama.
//   - Granite (dense): the MiniCPM scalars plus a custom QK attention scale (see Granite.java).
// Interleaved RoPE (the GGUF "llama" pair convention), KV written to the cache before a
// causalPrefill / flashDecode read, and a scalar residual scale on each sublayer output.
// Text-only, dense FFN. Chat templates, stop tokens, and the cache codec are OUT of slice (the
// chat/cache cycles) - this is the headless generative backbone.
package com.qxotic.jinfer.x.models.llama;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.Convert;
import com.qxotic.jinfer.x.Parallel;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.BaseState;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Config;
import com.qxotic.jinfer.x.boundary.LanguageModel;
import com.qxotic.jinfer.x.kernels.Activations;
import com.qxotic.jinfer.x.kernels.FlashAttention;
import com.qxotic.jinfer.x.kernels.MatMul;
import com.qxotic.jinfer.x.kernels.ModelLoader;
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
    public State newState(int contextCapacity, int batchCapacity, Arena arena) {
        return new State(configuration, contextCapacity, batchCapacity, arena);
    }

    @Override
    public void forward(State s, Batch batch) {
        int n = batch.count();
        if (n > s.batchCapacity) {
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + s.batchCapacity);
        }
        int from = s.position();
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
            case Batch.Input.Tokens t -> {
                int[] ids = t.ids();
                if (n == 1) {
                    Parallel.onDecodePool(
                            () -> {
                                forward(s, ids, 0, from, n);
                                return null;
                            });
                } else {
                    forward(s, ids, 0, from, n);
                }
            }
            case Batch.Input.Sequences seq ->
                    throw new UnsupportedOperationException(
                            "Llama is generative: packed sequences (batched embedding) not"
                                    + " supported");
        }
        s.advance(n, batch.outputs());
    }

    @Override
    public MemoryView<?> head(State s, int output) {
        int dim = configuration.embeddingLength;
        int row = s.lastChunkLen - s.outputCount + output;
        return Parallel.onDecodePool(
                () -> {
                    tailAt(s, row); // finish the deferred last-layer tail for this row -> s.th
                    Norms.rmsnorm(
                            s.normed,
                            0,
                            s.th,
                            0,
                            weights.finalNorm(),
                            dim,
                            configuration.rmsNormEps);
                    MatMul.gemv(
                            weights.wcls(), s.normed, s.logits, configuration.vocabularySize, dim);
                    float ls = configuration.logitScale;
                    if (ls != 1.0f) {
                        Ops.mapInPlace(s.logits, 0, configuration.vocabularySize, v -> v / ls);
                    }
                    return s.logits;
                });
    }

    // === Forward ===

    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        // ONCE for the batch: an angle depends on the position and the schedule, never on the
        // layer, so all of them read these rows
        RoPE.fill(state.ropeCos, state.ropeSin, startPos, seqLen, config.ropeHalf(), w.rope());
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;
        float embScale = config.embeddingScale, residScale = config.residualScale;

        Views.checkAlive(w.tokenEmbeddings(), "tokenEmbeddings"); // fail-fast on freed weights
        for (int s = 0; s < seqLen; s++) {
            Convert.copyToF32(
                    w.tokenEmbeddings(),
                    (long) tokens[tokenOffset + s] * dim,
                    state.residual,
                    (long) s * dim,
                    dim);
        }
        if (embScale != 1.0f) {
            Ops.mapInPlace(state.residual, 0, seqLen * dim, v -> v * embScale);
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
     * by every layer, so a deferred commit would write the LAST layer's values everywhere (the
     * xqwen3 bug, ported as law).
     */
    private void commitKv(State state, int l, int startPos, int seqLen) {
        int kvDim = configuration.kvDim();
        for (int s = 0; s < seqLen; s++) {
            long kvPos = startPos + s;
            Convert.f32ToF16(
                    state.batchK, (long) s * kvDim, state.keyCache[l], kvPos * kvDim, kvDim);
            Convert.f32ToF16(
                    state.batchV, (long) s * kvDim, state.valueCache[l], kvPos * kvDim, kvDim);
        }
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
        MatMul.gemm(lw.wk(), state.normed, dim, state.batchK, kvDim, kvDim, seqLen, dim);
        MatMul.gemm(lw.wv(), state.normed, dim, state.batchV, kvDim, kvDim, seqLen, dim);
        if (config.useRope(l)) {
            Parallel.forRows(
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
        int startPos = state.position - state.lastChunkLen; // global position of chunk row 0
        int pos = startPos + i; // global position of row i

        // pre-norm reads residual[i] directly (read-only)
        Norms.rmsnorm(
                state.tailScratch, 0, state.residual, (long) i * dim, lw.attnNorm(), dim, eps);
        // Q for this row (query is free scratch outside a forward)
        MatMul.gemm(lw.wq(), state.tailScratch, dim, state.query, queryDim, queryDim, 1, dim);
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
            Ops.mapInPlace(state.query, 0, queryDim, v -> v * aScale);
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
        MatMul.gemm(lw.wo(), state.attnOut, queryDim, state.tailScratch, dim, dim, 1, queryDim);
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

        MatMul.gemm(lw.wq(), state.normed, dim, state.query, queryDim, queryDim, seqLen, dim);
        MatMul.gemm(lw.wk(), state.normed, dim, state.batchK, kvDim, kvDim, seqLen, dim);
        MatMul.gemm(lw.wv(), state.normed, dim, state.batchV, kvDim, kvDim, seqLen, dim);
        boolean useRope = config.useRope(layer); // SmolLM3 NoPE: some layers skip RoPE entirely
        Parallel.forRows(
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
                        Ops.mapInPlace(state.query, (long) s * queryDim, queryDim, v -> v * aScale);
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
        MatMul.gemm(lw.wo(), state.attnOut, queryDim, state.normed, dim, dim, seqLen, queryDim);
    }

    /** Dense SwiGLU FFN over the pre-normed rows in {@code state.normed}, written back in place. */
    private void feedForward(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        LayerWeights lw = weights.layers()[l];
        MatMul.gemm(lw.w1(), state.normed, dim, state.hidden, hiddenDim, hiddenDim, seqLen, dim);
        MatMul.gemm(lw.w3(), state.normed, dim, state.hidden2, hiddenDim, hiddenDim, seqLen, dim);
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.siluMultiply(
                                state.hidden,
                                s * hiddenDim,
                                state.hidden2,
                                s * hiddenDim,
                                hiddenDim));
        MatMul.gemm(lw.w2(), state.hidden, hiddenDim, state.normed, dim, dim, seqLen, hiddenDim);
    }

    /** The one-row FFN of the lazy tail, over {@code io} in place (gate into hidden/hidden2). */
    private void feedForwardRow(State state, int l, MemoryView<MemorySegment> io) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        LayerWeights lw = weights.layers()[l];
        MatMul.gemm(lw.w1(), io, dim, state.hidden, hiddenDim, hiddenDim, 1, dim);
        MatMul.gemm(lw.w3(), io, dim, state.hidden2, hiddenDim, hiddenDim, 1, dim);
        Activations.siluMultiply(state.hidden, 0, state.hidden2, 0, hiddenDim);
        MatMul.gemm(lw.w2(), state.hidden, hiddenDim, io, dim, dim, 1, hiddenDim);
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
            implements Config {
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
            MemoryView<MemorySegment> wk,
            MemoryView<MemorySegment> wv,
            MemoryView<MemorySegment> wo,
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

    // === State ===

    public static final class State extends BaseState {
        final int contextCapacity, batchCapacity;

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
        public void reset() {
            resumeAt(0);
        }

        @SuppressWarnings("unchecked")
        State(Configuration config, int contextCapacity, int batchCapacity, Arena arena) {
            super(arena);
            if (contextCapacity > config.contextLength()) {
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " exceeds model contextLength "
                                + config.contextLength());
            }
            this.contextCapacity = contextCapacity;
            int c = Math.max(1, batchCapacity);
            this.batchCapacity = c;
            int dim = config.embeddingLength;
            int queryDim = config.queryDim();
            int kvDim = config.kvDim();
            int hidden = config.hiddenDim;
            this.residual = Views.allocateF32(memoryArena(), c * dim);
            this.normed = Views.allocateF32(memoryArena(), c * dim);
            this.batchK = Views.allocateF32(memoryArena(), c * kvDim);
            this.batchV = Views.allocateF32(memoryArena(), c * kvDim);
            this.query = Views.allocateF32(memoryArena(), c * queryDim);
            this.attnOut = Views.allocateF32(memoryArena(), c * queryDim);
            this.hidden = Views.allocateF32(memoryArena(), c * hidden);
            this.hidden2 = Views.allocateF32(memoryArena(), c * hidden);
            this.logits = Views.allocateF32(memoryArena(), config.vocabularySize);
            this.th = Views.allocateF32(memoryArena(), dim);
            this.tailScratch = Views.allocateF32(memoryArena(), dim);
            // rotary values for the batch about to be ingested: sized by BATCH, never context
            this.ropeCos = Views.allocateF32(memoryArena(), c * config.ropeHalf());
            this.ropeSin = Views.allocateF32(memoryArena(), c * config.ropeHalf());
            this.decodeScratch = new FlashAttention.DecodeScratch(memoryArena());
            this.keyCache = new MemoryView[config.numberOfLayers];
            this.valueCache = new MemoryView[config.numberOfLayers];
            for (int l = 0; l < config.numberOfLayers; l++) {
                keyCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
                valueCache[l] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
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

        int contextLength = gguf.getValue(int.class, arch + ".context_length");
        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfLayers = gguf.getValue(int.class, arch + ".block_count");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int numberOfKeyValueHeads =
                gguf.getValueOrDefault(int.class, arch + ".attention.head_count_kv", numberOfHeads);
        int headSize =
                gguf.getValueOrDefault(
                        int.class, arch + ".attention.key_length", embeddingLength / numberOfHeads);
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
                        isMiniCpm ? embeddingLength / 256.0f : 1.0f);

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
                        tokenizer.vocabulary().size(),
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

        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(fileChannel, gguf, arena);
        RoPE.Schedule rope = buildRope(gguf, arch, config, tensors);
        return new Llama(config, tokenizer, loadWeights(tensors, config, rope));
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
        int ropeDim = Math.min(config.ropeDimensionCount, config.headSize);
        String scalingType = gguf.getValueOrDefault(String.class, arch + ".rope.scaling.type", "");
        if (scalingType.equals("yarn")) {
            float factor = gguf.getValue(float.class, arch + ".rope.scaling.factor");
            int origCtx = gguf.getValue(int.class, arch + ".rope.scaling.original_context_length");
            float betaFast =
                    gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_fast", 32f);
            float betaSlow =
                    gguf.getValueOrDefault(float.class, arch + ".rope.scaling.yarn_beta_slow", 1f);
            float logMul =
                    gguf.getValueOrDefault(
                            float.class, arch + ".rope.scaling.yarn_log_multiplier", 0f);
            float kMscale = factor <= 1f ? 1f : (float) (1.0 + 0.1 * Math.log(factor));
            float attnFactor = logMul != 0f ? 1.0f / kMscale : 1.0f;
            return RoPE.yarn(
                    ropeDim, config.ropeTheta, factor, origCtx, betaFast, betaSlow, 1f, attnFactor);
        }
        float[] ropeFreqs = ModelLoader.ropeFreqFactors(tensors);
        return ropeFreqs != null
                ? RoPE.withFreqFactors(ropeDim, config.ropeTheta, ropeFreqs)
                : RoPE.plain(ropeDim, config.ropeTheta);
    }

    // ---- loadWeights helpers: the old ModelLoader.toF32Tensor/loadQuantized fail-fast contract --

    private static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        return Objects.requireNonNull(tensors.get(name), name);
    }

    /** F32 view by name (dtype checked AT LOAD, the old toF32Tensor fail-fast), or throw. */
    private static MemoryView<MemorySegment> requireF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> v = require(tensors, name);
        Views.requireDatatype(v, DataType.FP32, name);
        return v;
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors,
            Configuration config,
            RoPE.Schedule rope) {
        int n = config.numberOfLayers;
        MemoryView<MemorySegment> tokenEmbeddings = require(tensors, "token_embd.weight");
        MemoryView<MemorySegment> wcls =
                tensors.containsKey("output.weight")
                        ? require(tensors, "output.weight")
                        : tokenEmbeddings; // tied embeddings
        MemoryView<MemorySegment> finalNorm = requireF32(tensors, "output_norm.weight");

        LayerWeights[] layers = new LayerWeights[n];
        for (int i = 0; i < n; i++) {
            String p = "blk." + i + ".";
            layers[i] =
                    new LayerWeights(
                            requireF32(tensors, p + "attn_norm.weight"),
                            require(tensors, p + "attn_q.weight"),
                            require(tensors, p + "attn_k.weight"),
                            require(tensors, p + "attn_v.weight"),
                            require(tensors, p + "attn_output.weight"),
                            requireF32(tensors, p + "ffn_norm.weight"),
                            require(tensors, p + "ffn_gate.weight"),
                            require(tensors, p + "ffn_down.weight"),
                            require(tensors, p + "ffn_up.weight"));
        }
        return new Weights(tokenEmbeddings, layers, finalNorm, rope, wcls);
    }
}
