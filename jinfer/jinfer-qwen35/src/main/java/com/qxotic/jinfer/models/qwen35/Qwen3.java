// Qwen3 dense as an EMBEDDING model (Qwen3-Embedding) against the com.qxotic.jinfer API. Qwen3
// dense is
// the standard Llama transformer (GQA + SwiGLU + RMSNorm pre-norm, no attention bias) with two
// twists:
//   - QK-norm: a per-head RMSNorm on Q and K (weight dim = head_size), applied BEFORE RoPE.
//   - NeoX (rotate-half) RoPE, not Llama's interleaved pair convention.
// The embedding is the LAST token's hidden state after the final RMSNorm, L2-normalized
// (pooling_type=LAST,
// matching llama.cpp --pooling last). No LM head / vocab projection. Single batched causal prefill,
// no decode.
package com.qxotic.jinfer.models.qwen35;

import static com.qxotic.jinfer.Norms.rmsnorm;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.*;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

public final class Qwen3
        implements EmbeddingModel<Qwen3.Configuration, Qwen3.Weights, Qwen3.State> {

    private final Configuration configuration;
    private final GgufTokenizer tokenizer;
    private final Weights weights;

    Qwen3(Configuration configuration, GgufTokenizer tokenizer, Weights weights) {
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

    public GgufTokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public State newState(int contextCapacity, int batchCapacity) {
        return new State(configuration, contextCapacity, batchCapacity);
    }

    @Override
    public void ingest(State s, com.qxotic.jinfer.Batch batch) {
        int n = batch.count();
        if (n > s.batchCapacity)
            throw new IllegalArgumentException(
                    "batch " + n + " exceeds batchCapacity " + s.batchCapacity);
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
            case com.qxotic.jinfer.Batch.Input.Tokens t -> forward(s, t.ids(), 0, from, n);
            case com.qxotic.jinfer.Batch.Input.Sequences seq ->
                    forwardSegmented(s, seq.tokens().ids(), seq.seqLen(), from, n);
            case com.qxotic.jinfer.Batch.Input.Embeddings e ->
                    throw new UnsupportedOperationException(
                            "Qwen3 embedding is text-only: embedding input is not supported");
        }
        s.advance(n, batch.outputs());
    }

    /**
     * The sentence embedding: pool the {@code index}-th retained row (last-token pooling per
     * sequence), L2-normalized. {@code index} addresses the retained rows exactly as {@code
     * logits}' output does.
     */
    @Override
    public FloatTensor embedding(State s, int index) {
        int dim = configuration.embeddingLength;
        int row =
                s.lastChunkLen
                        - s.outputCount
                        + index; // retained-output index -> chunk row (mirrors logits)
        FloatTensor out = FloatTensor.allocateF32(dim);
        rmsnorm(out, 0, s.x, (long) row * dim, weights.outputNorm, dim, configuration.rmsNormEps);
        double ss = 0;
        for (int i = 0; i < dim; i++) {
            float v = out.getFloat(i);
            ss += (double) v * v;
        }
        float inv = ss > 0 ? (float) (1.0 / Math.sqrt(ss)) : 0f;
        out.mapInPlace(0, dim, v -> v * inv);
        return out;
    }

    // === Forward ===

    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;

        for (int s = 0; s < seqLen; s++) {
            w.tokenEmbeddingTable.copyTo(
                    (long) tokens[tokenOffset + s] * dim, state.x, (long) s * dim, dim);
        }
        for (int l = 0; l < config.numberOfLayers; l++) {
            int fDim = dim;
            F32FloatTensor attNormW = w.attnNorm[l], ffnNormW = w.ffnNorm[l];
            Parallel.forRows(
                    seqLen,
                    s ->
                            rmsnorm(
                                    state.xb,
                                    (long) s * fDim,
                                    state.x,
                                    (long) s * fDim,
                                    attNormW,
                                    fDim,
                                    eps));
            attention(state, l, startPos, seqLen);
            state.x.addInPlace(0, state.xb, 0, seqLen * dim);
            Parallel.forRows(
                    seqLen,
                    s ->
                            rmsnorm(
                                    state.xb,
                                    (long) s * fDim,
                                    state.x,
                                    (long) s * fDim,
                                    ffnNormW,
                                    fDim,
                                    eps));
            feedForward(state, l, seqLen);
            state.x.addInPlace(0, state.xb, 0, seqLen * dim);
        }
    }

    /**
     * GQA attention: Q/K/V projections, per-head QK RMS-norm, NeoX RoPE, causal flash attention,
     * output projection back into {@code state.xb}.
     */
    private void attention(State state, int layer, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int headSize = config.headSize;
        int heads = config.numberOfHeads;
        int kvHeads = config.numberOfKeyValueHeads;
        int kvDim = config.kvDim();
        int queryDim = config.queryDim();
        int kvMul = heads / kvHeads;
        int ropeHalf = config.ropeHalf();
        float eps = config.rmsNormEps;

        w.wq[layer].gemm(state.xb, dim, state.attnQ, queryDim, seqLen, queryDim, dim);
        w.wk[layer].gemm(state.xb, dim, state.k, kvDim, seqLen, kvDim, dim);
        w.wv[layer].gemm(state.xb, dim, state.v, kvDim, seqLen, kvDim, dim);

        int fHeadSz = headSize,
                fHeads = heads,
                fKvHeads = kvHeads,
                fQDim = queryDim,
                fKvDim = kvDim,
                fStart = startPos,
                fHalf = ropeHalf;
        F32FloatTensor qNorm = w.qNorm[layer], kNorm = w.kNorm[layer];
        Parallel.forRows(
                seqLen,
                s -> {
                    for (int h = 0; h < fHeads; h++) {
                        long off = (long) s * fQDim + (long) h * fHeadSz;
                        rmsnorm(
                                state.attnQ,
                                off,
                                state.attnQ,
                                off,
                                qNorm,
                                fHeadSz,
                                eps); // per-head QK-norm before RoPE
                        RoPE.applyNeox(state.attnQ, off, fStart + s, w.ropeCr, w.ropeCi, fHalf);
                    }
                    for (int h = 0; h < fKvHeads; h++) {
                        long off = (long) s * fKvDim + (long) h * fHeadSz;
                        rmsnorm(state.k, off, state.k, off, kNorm, fHeadSz, eps);
                        RoPE.applyNeox(state.k, off, fStart + s, w.ropeCr, w.ropeCi, fHalf);
                    }
                });

        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        float attScale = 1.0f / (float) Math.sqrt(headSize);
        FlashAttention.slidingWindowPrefill(
                state.attnQ,
                state.attnOut,
                keyCache,
                valueCache,
                state.k,
                state.v,
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
        for (int s = 0; s < seqLen; s++) {
            state.k.copyTo((long) s * kvDim, keyCache, (long) (startPos + s) * kvDim, kvDim);
            state.v.copyTo((long) s * kvDim, valueCache, (long) (startPos + s) * kvDim, kvDim);
        }
        w.wo[layer].gemm(state.attnOut, queryDim, state.xb, dim, seqLen, dim, queryDim);
    }

    /**
     * Dense SwiGLU FFN over the pre-normed rows in {@code state.xb}, written back to {@code
     * state.xb}.
     */
    private void feedForward(State state, int l, int seqLen) {
        int dim = configuration.embeddingLength, hiddenDim = configuration.hiddenDim;
        Weights w = weights;
        w.w1[l].gemm(state.xb, dim, state.hb, hiddenDim, seqLen, hiddenDim, dim); // gate
        w.w3[l].gemm(state.xb, dim, state.hb2, hiddenDim, seqLen, hiddenDim, dim); // up
        Parallel.forRows(
                seqLen,
                s ->
                        Activations.siluMultiply(
                                state.hb, s * hiddenDim, state.hb2, s * hiddenDim, hiddenDim));
        w.w2[l].gemm(state.hb, hiddenDim, state.xb, dim, seqLen, dim, hiddenDim); // down
    }

    // === Segmented (packed batched-embedding) forward ===
    // Same GEMM-batched backbone as forward(), but RoPE positions restart per sequence and each
    // token
    // attends only to its own sequence's KV slice (causally). The chunk's tokens sit at global
    // positions
    // [cs, cs+seqLen); seqLen[] is the FULL per-sequence lengths. Projection/FFN GEMMs stay batched
    // over the
    // whole chunk (the throughput win); only positions + the attention are segmented.

    void forwardSegmented(State state, int[] tokens, int[] fullSeqLen, int cs, int seqLen) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        float eps = config.rmsNormEps;

        // Per chunk-row: within-sequence RoPE position (restarts per sequence). Then group
        // consecutive rows of
        // the same sequence into segments {row0, len, kvStart (global cache row), prior (tokens
        // before this chunk)}.
        int[] posOf = new int[seqLen];
        int[] segRow0 = new int[seqLen],
                segLen = new int[seqLen],
                segKv = new int[seqLen],
                segPrior = new int[seqLen];
        int nSeg = 0, gStart = 0, j = 0;
        while (j < fullSeqLen.length && gStart + fullSeqLen[j] <= cs) {
            gStart += fullSeqLen[j];
            j++;
        }
        for (int s = 0; s < seqLen; ) {
            while (j < fullSeqLen.length && cs + s >= gStart + fullSeqLen[j]) {
                gStart += fullSeqLen[j];
                j++;
            }
            int segStart = s,
                    prior = (cs + s) - gStart,
                    kvStart = gStart,
                    segEnd = gStart + fullSeqLen[j];
            for (; s < seqLen && cs + s < segEnd; s++) posOf[s] = (cs + s) - gStart;
            segRow0[nSeg] = segStart;
            segLen[nSeg] = s - segStart;
            segKv[nSeg] = kvStart;
            segPrior[nSeg] = prior;
            nSeg++;
        }

        for (int s = 0; s < seqLen; s++) {
            w.tokenEmbeddingTable.copyTo((long) tokens[s] * dim, state.x, (long) s * dim, dim);
        }
        for (int l = 0; l < config.numberOfLayers; l++) {
            int fDim = dim;
            F32FloatTensor attNormW = w.attnNorm[l], ffnNormW = w.ffnNorm[l];
            Parallel.forRows(
                    seqLen,
                    s ->
                            rmsnorm(
                                    state.xb,
                                    (long) s * fDim,
                                    state.x,
                                    (long) s * fDim,
                                    attNormW,
                                    fDim,
                                    eps));
            attentionSegmented(state, l, cs, seqLen, posOf, segRow0, segLen, segKv, segPrior, nSeg);
            state.x.addInPlace(0, state.xb, 0, seqLen * dim);
            Parallel.forRows(
                    seqLen,
                    s ->
                            rmsnorm(
                                    state.xb,
                                    (long) s * fDim,
                                    state.x,
                                    (long) s * fDim,
                                    ffnNormW,
                                    fDim,
                                    eps));
            feedForward(state, l, seqLen);
            state.x.addInPlace(0, state.xb, 0, seqLen * dim);
        }
    }

    /**
     * GQA attention with per-sequence isolation: Q/K/V projections batched over the chunk, per-head
     * QK-norm + NeoX RoPE at each token's within-sequence position, K/V written to the cache at
     * global positions, then each sequence runs the SAME flash {@link FlashAttention#causalPrefill}
     * the single-sequence path uses - its Q + full KV slice (prior chunks from the cache, this
     * chunk from the F32 batch) gathered to scratch.
     */
    private void attentionSegmented(
            State state,
            int layer,
            int cs,
            int seqLen,
            int[] posOf,
            int[] segRow0,
            int[] segLen,
            int[] segKv,
            int[] segPrior,
            int nSeg) {
        Configuration config = configuration;
        Weights w = weights;
        int dim = config.embeddingLength;
        int headSize = config.headSize,
                heads = config.numberOfHeads,
                kvHeads = config.numberOfKeyValueHeads;
        int kvDim = config.kvDim(),
                queryDim = config.queryDim(),
                kvMul = heads / kvHeads,
                ropeHalf = config.ropeHalf();
        float eps = config.rmsNormEps, attScale = 1.0f / (float) Math.sqrt(headSize);

        w.wq[layer].gemm(state.xb, dim, state.attnQ, queryDim, seqLen, queryDim, dim);
        w.wk[layer].gemm(state.xb, dim, state.k, kvDim, seqLen, kvDim, dim);
        w.wv[layer].gemm(state.xb, dim, state.v, kvDim, seqLen, kvDim, dim);

        int fHeadSz = headSize,
                fHeads = heads,
                fKvHeads = kvHeads,
                fQDim = queryDim,
                fKvDim = kvDim,
                fHalf = ropeHalf;
        F32FloatTensor qNorm = w.qNorm[layer], kNorm = w.kNorm[layer];
        Parallel.forRows(
                seqLen,
                s -> {
                    int pos = posOf[s]; // within-sequence position for RoPE (restarts per sequence)
                    for (int h = 0; h < fHeads; h++) {
                        long off = (long) s * fQDim + (long) h * fHeadSz;
                        rmsnorm(state.attnQ, off, state.attnQ, off, qNorm, fHeadSz, eps);
                        RoPE.applyNeox(state.attnQ, off, pos, w.ropeCr, w.ropeCi, fHalf);
                    }
                    for (int h = 0; h < fKvHeads; h++) {
                        long off = (long) s * fKvDim + (long) h * fHeadSz;
                        rmsnorm(state.k, off, state.k, off, kNorm, fHeadSz, eps);
                        RoPE.applyNeox(state.k, off, pos, w.ropeCr, w.ropeCi, fHalf);
                    }
                });

        FloatTensor keyCache = state.keyCache[layer], valueCache = state.valueCache[layer];
        for (int s = 0;
                s < seqLen;
                s++) { // K/V into the cache at global positions (for later chunks)
            state.k.copyTo((long) s * kvDim, keyCache, (long) (cs + s) * kvDim, kvDim);
            state.v.copyTo((long) s * kvDim, valueCache, (long) (cs + s) * kvDim, kvDim);
        }
        for (int g = 0; g < nSeg; g++) { // one flash prefill per sequence over its own KV slice
            int r0 = segRow0[g], sl = segLen[g], kvg = segKv[g], prior = segPrior[g];
            state.attnQ.copyTo((long) r0 * queryDim, state.segQ, 0, sl * queryDim);
            if (prior > 0) { // sequence's earlier tokens live in the F16 cache (prior chunks)
                keyCache.copyTo((long) kvg * kvDim, state.segK, 0, prior * kvDim);
                valueCache.copyTo((long) kvg * kvDim, state.segV, 0, prior * kvDim);
            }
            state.k.copyTo(
                    (long) r0 * kvDim,
                    state.segK,
                    (long) prior * kvDim,
                    sl * kvDim); // this chunk (F32)
            state.v.copyTo((long) r0 * kvDim, state.segV, (long) prior * kvDim, sl * kvDim);
            if (prior == 0) // sequence wholly in this chunk: the exact single-sequence kernel
                // (bit-for-bit)
                FlashAttention.slidingWindowPrefill(
                        state.segQ,
                        state.segOut,
                        keyCache,
                        valueCache,
                        state.segK,
                        state.segV,
                        heads,
                        0,
                        sl,
                        headSize,
                        kvDim,
                        queryDim,
                        kvDim,
                        kvMul,
                        attScale,
                        0,
                        0,
                        null);
            else // spanning: prior tokens gathered from the cache, so drive from one F32 KV slice
            FlashAttention.causalPrefill(
                        (F32FloatTensor) state.segQ,
                        (F32FloatTensor) state.segOut,
                        state.segK,
                        state.segV,
                        heads,
                        prior,
                        sl,
                        headSize,
                        kvDim,
                        queryDim,
                        kvMul);
            state.segOut.copyTo(0, state.attnOut, (long) r0 * queryDim, sl * queryDim);
        }
        w.wo[layer].gemm(state.attnOut, queryDim, state.xb, dim, seqLen, dim, queryDim);
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
            int hiddenDim)
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
    }

    // === Weights ===

    public record Weights(
            FloatTensor tokenEmbeddingTable,
            F32FloatTensor outputNorm,
            F32FloatTensor[] attnNorm,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            F32FloatTensor[] qNorm,
            F32FloatTensor[] kNorm,
            F32FloatTensor[] ffnNorm,
            FloatTensor[] w1,
            FloatTensor[] w3,
            FloatTensor[] w2,
            float[] ropeCr,
            float[] ropeCi) {}

    // === State ===

    public static final class State extends com.qxotic.jinfer.BaseState {
        final int contextCapacity, batchCapacity;
        final FloatTensor x, xb, k, v, attnQ, attnOut, hb, hb2;
        final FloatTensor segQ, segOut, segK, segV; // per-sequence attention scratch (packed path)
        final FloatTensor[] keyCache, valueCache;

        State(Configuration config, int contextCapacity, int batchCapacity) {
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
            int dim = config.embeddingLength,
                    queryDim = config.queryDim(),
                    kvDim = config.kvDim(),
                    hidden = config.hiddenDim;
            this.x = FloatTensor.allocateF32(c * dim);
            this.xb = FloatTensor.allocateF32(c * dim);
            this.k = FloatTensor.allocateF32(c * kvDim);
            this.v = FloatTensor.allocateF32(c * kvDim);
            this.attnQ = FloatTensor.allocateF32(c * queryDim);
            this.attnOut = FloatTensor.allocateF32(c * queryDim);
            this.hb = FloatTensor.allocateF32(c * hidden);
            this.hb2 = FloatTensor.allocateF32(c * hidden);
            // per-sequence attention scratch: q/out over a chunk's rows, K/V over a sequence's full
            // extent
            this.segQ = FloatTensor.allocateF32(c * queryDim);
            this.segOut = FloatTensor.allocateF32(c * queryDim);
            this.segK = FloatTensor.allocateF32(contextCapacity * kvDim);
            this.segV = FloatTensor.allocateF32(contextCapacity * kvDim);
            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            for (int l = 0; l < config.numberOfLayers; l++) {
                keyCache[l] = FloatTensor.allocateF16(contextCapacity, kvDim);
                valueCache[l] = FloatTensor.allocateF16(contextCapacity, kvDim);
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
            position = 0;
        }
    }

    // === Loading ===

    public static Qwen3 loadModel(Path ggufPath, int contextLength) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, contextLength, true);
        }
    }

    static Qwen3 loadModel(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag)
            throws IOException {
        GgufTokenizer tokenizer = new GgufTokenizer(gguf, JinjaRenderer::template);
        String arch = gguf.getString("general.architecture");

        int modelContextLength = gguf.getValue(int.class, arch + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength)
            contextLength = modelContextLength;

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
                        float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 1000000f);
        int ropeDimensionCount =
                gguf.getValueOrDefault(int.class, arch + ".rope.dimension_count", headSize);

        Configuration config =
                new Configuration(
                        embeddingLength,
                        numberOfLayers,
                        numberOfHeads,
                        numberOfKeyValueHeads,
                        headSize,
                        tokenizer.vocabularySize(),
                        contextLength,
                        rmsNormEps,
                        ropeTheta,
                        ropeDimensionCount,
                        hiddenDim);

        if (!loadWeightsFlag) return new Qwen3(config, tokenizer, null);
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        int ropeDim = Math.min(config.ropeDimensionCount, config.headSize);
        Pair<float[], float[]> rope =
                RoPE.precomputeFreqsCis(config.contextLength, ropeDim, config.ropeTheta);
        return new Qwen3(config, tokenizer, loadWeights(tensors, config, rope));
    }

    static Weights loadWeights(
            Map<String, GGMLTensorEntry> tensors,
            Configuration config,
            Pair<float[], float[]> rope) {
        int n = config.numberOfLayers;
        return new Weights(
                ModelLoader.loadQuantized(tensors.get("token_embd.weight")),
                ModelLoader.toF32Tensor(tensors.get("output_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_q.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_k.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_v.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_output.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_q_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_k_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down.weight")),
                rope.first(),
                rope.second());
    }
}
