// Gemma 4 model support, kept fully separate from the LFM2.5 implementation (Llama): its own
// configuration, weights, state and forward pass. Shared infrastructure — tensors, kernels,
// samplers, the GGUF tokenizer (toknroll), RoPE and the chat layer — lives outside this file.
//
// Architecture (ported from the reference gemma4.java): per-layer sliding-window vs full
// attention with distinct RoPE thetas, shared KV across the tail layers, per-head Q/K RMS norms
// plus a bare V norm, embedding scaling by sqrt(dim), GELU MLP, pre/post attention and FFN norms,
// per-layer output scaling, final-logit soft-capping, optional per-layer embeddings (PLE) and an
// optional shared-MLP + top-k expert MoE FFN (the A4B variant).
package com.llama4j;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;

import static com.llama4j.Norms.rmsnorm;
import static com.llama4j.Norms.rmsnormNoWeight;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class Gemma4 implements Model {

    private final Configuration configuration;
    private final LFMTokenizer tokenizer;
    private final Weights weights;

    Gemma4(Configuration configuration, LFMTokenizer tokenizer, Weights weights) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
    }

    Configuration configuration() {
        return configuration;
    }

    // --- Model seam ---

    @Override
    public LFMTokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public int contextLength() {
        return configuration.contextLength;
    }

    @Override
    public int vocabularySize() {
        return configuration.vocabularySize;
    }

    /** Force the single-token reference forward even for multi-token chunks (parity debugging). */
    private static final boolean SINGLE_TOKEN_PREFILL = System.getProperty("gemma.singleTokenPrefill") != null;

    // -Dgemma.trace: dump per-layer residual-stream checkpoints (sum over all positions) matching
    // llama.cpp's eval-callback node names (inp_scaled, l_out-{l}), for cross-engine validation.
    static final boolean TRACE = System.getProperty("gemma.trace") != null;
    static void traceSum(String name, FloatTensor t, int n) {
        if (!TRACE) return;
        double s = 0;
        for (int i = 0; i < n; i++) s += t.getFloat(i);
        // sum + first 3 values of position 0 (cancellation-free, matches eval-callback's row-0 sample)
        System.err.printf("[trace] %s sum=%.6f v0=%.4f,%.4f,%.4f%n", name, s, t.getFloat(0), t.getFloat(1), t.getFloat(2));
    }

    @Override
    public int batchCapacity() {
        return SINGLE_TOKEN_PREFILL ? 1 : Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
    }

    @Override
    public State createNewState() {
        State state = new State(configuration);
        Integer bos = tokenizer.getSpecialTokens().get("<bos>");
        state.latestToken = bos != null ? bos : 2;
        return state;
    }

    @Override
    public void ingest(InferenceState state, int[] tokens, int tokenOffset, int startPosition, int sequenceLength) {
        State s = (State) state;
        if (sequenceLength > s.capacity) {
            throw new IllegalArgumentException("sequenceLength " + sequenceLength + " exceeds batch capacity " + s.capacity);
        }
        if (SINGLE_TOKEN_PREFILL) {
            for (int i = 0; i < sequenceLength; i++) forward(s, tokens, tokenOffset + i, startPosition + i, 1);
        } else {
            forward(s, tokens, tokenOffset, startPosition, sequenceLength);
        }
        s.latestToken = tokens[tokenOffset + sequenceLength - 1];
        s.logitsValid = false;
    }

    /**
     * Logits for the last ingested token. Both forward paths leave the post-final-layer residual
     * in {@code state.x} (at {@code lastRowOffset}); this finalizes the final norm + output head
     * for that single row. Idempotent between ingests.
     */
    @Override
    public FloatTensor computeLogits(InferenceState state) {
        State s = (State) state;
        if (s.logitsValid) {
            return s.logits;
        }
        int dim = configuration.embeddingLength;
        rmsnorm(s.xb, 0, s.x, s.lastRowOffset, weights.rmsFinalWeight, dim, configuration.rmsNormEps);
        weights.wcls.matmul(s.xb, s.logits, configuration.vocabularySize, dim);
        if (configuration.logitSoftcapping > 0) {
            float cap = configuration.logitSoftcapping;
            s.logits.mapInPlace(v -> cap * (float) Math.tanh(v / cap));
        }
        s.logitsValid = true;
        return s.logits;
    }

    @Override
    public ChatFormat chatFormat() {
        return new GemmaChatFormat(tokenizer);
    }

    @Override
    public Set<Integer> stopTokens() {
        // Gemma's turn delimiter is <turn|> (end of turn) and <eos>; this conversion also exposes
        // <end_of_turn> on some builds. Stop on whichever are present.
        Set<Integer> stops = new HashSet<>();
        for (String name : new String[]{"<turn|>", "<end_of_turn>", "<eos>", "<|endoftext|>"}) {
            Integer id = tokenizer.getSpecialTokens().get(name);
            if (id != null) stops.add(id);
        }
        return stops;
    }

    // === Configuration ===

    static final class Configuration {
        final int embeddingLength;
        final int[] feedForwardLength;             // per-layer (shared MLP)
        final int numberOfLayers;
        final int numberOfHeads;
        final int[] numberOfKeyValueHeadsPerLayer; // per-layer KV head count
        final int vocabularySize;
        final int contextLength;
        final float rmsNormEps;
        final float ropeTheta;                     // full-attention RoPE theta
        final float ropeThetaSWA;                  // SWA RoPE theta
        final int headSizeFull;                    // head size for full-attention layers
        final int headSizeSWA;                     // head size for sliding-window layers
        final int slidingWindow;                   // power of 2
        final float logitSoftcapping;
        final boolean[] isSWA;                     // per-layer: true = sliding window, false = full
        final int nLayerKvFromStart;               // first N layers own their KV; the rest reuse it
        final int embeddingLengthPerLayer;         // per-layer embedding dim (0 = disabled)
        final int expertCount;                     // 0 = dense
        final int expertUsedCount;                 // top-k experts per token
        final int expertFeedForwardLength;         // expert FFN hidden dim

        Configuration(int embeddingLength, int[] feedForwardLength, int numberOfLayers, int numberOfHeads,
                      int[] numberOfKeyValueHeadsPerLayer, int vocabularySize, int contextLength, float rmsNormEps,
                      float ropeTheta, float ropeThetaSWA, int headSizeFull, int headSizeSWA, int slidingWindow,
                      float logitSoftcapping, boolean[] isSWA, int nLayerKvFromStart, int embeddingLengthPerLayer,
                      int expertCount, int expertUsedCount, int expertFeedForwardLength) {
            if (slidingWindow <= 0 || Integer.bitCount(slidingWindow) != 1) {
                throw new IllegalArgumentException("slidingWindow must be a power of 2, got " + slidingWindow);
            }
            this.embeddingLength = embeddingLength;
            this.feedForwardLength = feedForwardLength;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeadsPerLayer = numberOfKeyValueHeadsPerLayer;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.ropeThetaSWA = ropeThetaSWA;
            this.headSizeFull = headSizeFull;
            this.headSizeSWA = headSizeSWA;
            this.slidingWindow = slidingWindow;
            this.logitSoftcapping = logitSoftcapping;
            this.isSWA = isSWA;
            this.nLayerKvFromStart = nLayerKvFromStart;
            this.embeddingLengthPerLayer = embeddingLengthPerLayer;
            this.expertCount = expertCount;
            this.expertUsedCount = expertUsedCount;
            this.expertFeedForwardLength = expertFeedForwardLength;
        }

        boolean isMoE() {
            return expertCount > 0;
        }

        boolean hasKv(int layer) {
            return layer < nLayerKvFromStart;
        }

        /** For a layer without its own KV, the earlier layer whose cache it reuses (last own-KV
         *  layer of the same attention type: SWA -> nLayerKvFromStart-2, full -> -1). */
        int kvSourceLayer(int layer) {
            if (layer < nLayerKvFromStart) return layer;
            return nLayerKvFromStart - (isSWA[layer] ? 2 : 1);
        }

        int headSize(int layer) {
            return isSWA[layer] ? headSizeSWA : headSizeFull;
        }

        int kvCachePositions(int layer) {
            return isSWA[layer] ? Math.min(contextLength, slidingWindow) : contextLength;
        }

        int kvCacheIndex(int layer, int position) {
            return isSWA[layer] ? (position & (slidingWindow - 1)) : position;
        }

        int kvDim(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] * headSize(layer);
        }

        int queryDim(int layer) {
            return numberOfHeads * headSize(layer);
        }

        int maxHiddenDim() {
            return Arrays.stream(feedForwardLength).max().orElseThrow();
        }
    }

    // === Weights ===

    static final class Weights {
        final FloatTensor tokenEmbeddingTable;
        final F32FloatTensor[] rmsAttWeight;
        final FloatTensor[] wq, wk, wv, wo;        // wv[l] null => V = K
        final F32FloatTensor[] attnQNorm, attnKNorm, postAttentionNorm;
        final F32FloatTensor[] rmsFfnWeight;
        final FloatTensor[] w1, w2, w3;            // gate, down, up
        final F32FloatTensor[] postFfwNorm;
        final F32FloatTensor rmsFinalWeight;
        final float[] layerOutputScale;
        final F32FloatTensor freqCisRealFull, freqCisImagFull, freqCisRealSwa, freqCisImagSwa;
        final FloatTensor wcls;
        // Per-layer embeddings (null when absent)
        final FloatTensor perLayerTokenEmbd, perLayerModelProj;
        final F32FloatTensor perLayerProjNorm;
        final FloatTensor[] perLayerInpGate, perLayerProj;
        final F32FloatTensor[] perLayerPostNorm;
        // MoE (null for dense models)
        final FloatTensor[] ffnGateInp, ffnGateUpExps, ffnDownExps;
        final F32FloatTensor[] ffnGateInpScale, ffnDownExpsScale, ffnPostNorm1, preFfwNorm2, ffnPostNorm2;

        Weights(FloatTensor tokenEmbeddingTable, F32FloatTensor[] rmsAttWeight, FloatTensor[] wq, FloatTensor[] wk,
                FloatTensor[] wv, FloatTensor[] wo, F32FloatTensor[] attnQNorm, F32FloatTensor[] attnKNorm,
                F32FloatTensor[] postAttentionNorm, F32FloatTensor[] rmsFfnWeight, FloatTensor[] w1, FloatTensor[] w2,
                FloatTensor[] w3, F32FloatTensor[] postFfwNorm, F32FloatTensor rmsFinalWeight, float[] layerOutputScale,
                F32FloatTensor freqCisRealFull, F32FloatTensor freqCisImagFull, F32FloatTensor freqCisRealSwa,
                F32FloatTensor freqCisImagSwa, FloatTensor wcls, FloatTensor perLayerTokenEmbd,
                FloatTensor perLayerModelProj, F32FloatTensor perLayerProjNorm, FloatTensor[] perLayerInpGate,
                FloatTensor[] perLayerProj, F32FloatTensor[] perLayerPostNorm, FloatTensor[] ffnGateInp,
                F32FloatTensor[] ffnGateInpScale, FloatTensor[] ffnGateUpExps, FloatTensor[] ffnDownExps,
                F32FloatTensor[] ffnDownExpsScale, F32FloatTensor[] ffnPostNorm1, F32FloatTensor[] preFfwNorm2,
                F32FloatTensor[] ffnPostNorm2) {
            this.tokenEmbeddingTable = tokenEmbeddingTable;
            this.rmsAttWeight = rmsAttWeight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
            this.attnQNorm = attnQNorm;
            this.attnKNorm = attnKNorm;
            this.postAttentionNorm = postAttentionNorm;
            this.rmsFfnWeight = rmsFfnWeight;
            this.w1 = w1;
            this.w2 = w2;
            this.w3 = w3;
            this.postFfwNorm = postFfwNorm;
            this.rmsFinalWeight = rmsFinalWeight;
            this.layerOutputScale = layerOutputScale;
            this.freqCisRealFull = freqCisRealFull;
            this.freqCisImagFull = freqCisImagFull;
            this.freqCisRealSwa = freqCisRealSwa;
            this.freqCisImagSwa = freqCisImagSwa;
            this.wcls = wcls;
            this.perLayerTokenEmbd = perLayerTokenEmbd;
            this.perLayerModelProj = perLayerModelProj;
            this.perLayerProjNorm = perLayerProjNorm;
            this.perLayerInpGate = perLayerInpGate;
            this.perLayerProj = perLayerProj;
            this.perLayerPostNorm = perLayerPostNorm;
            this.ffnGateInp = ffnGateInp;
            this.ffnGateInpScale = ffnGateInpScale;
            this.ffnGateUpExps = ffnGateUpExps;
            this.ffnDownExps = ffnDownExps;
            this.ffnDownExpsScale = ffnDownExpsScale;
            this.ffnPostNorm1 = ffnPostNorm1;
            this.preFfwNorm2 = preFfwNorm2;
            this.ffnPostNorm2 = ffnPostNorm2;
        }
    }

    // === State ===

    static final class State implements InferenceState {
        // Batched scratch (capacity rows): the residual stream and projections hold one row per
        // token in the current chunk; att / plGate / plProj / the MoE buffers stay single-row and
        // are reused across rows. KV cache is the cross-row source of truth.
        final int capacity;
        final FloatTensor x, xb, xbK, xb2, hb, hb2, q, k, v, att, logits;
        final FloatTensor[] keyCache, valueCache;   // only the own-KV layers are allocated
        // Linear (non-ring) K/V for the current chunk, per own-KV layer: attention reads in-chunk
        // keys here and prior keys from the ring cache, so a multi-token chunk never overwrites a
        // cache slot another row in the same chunk still needs. Written to the cache at chunk end.
        final FloatTensor[] batchK, batchV;
        final FloatTensor perLayerInputs, plGate, plProj;
        // Batched MoE scratch (chunk-wide), allocated only for MoE models. The expert FFN groups the
        // chunk's tokens by routed expert (CSR) so each expert's weights are read once per chunk via
        // GEMM instead of once per token via GEMV; the shared MLP is likewise batched across rows.
        final FloatTensor moeShared, moeInputB, moeRouterScaled, moeRouterB, moeOutB, moeGather, moeDownB;
        final int[] moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeRowTopE;
        final float[] moeProbByExpert, moeRowTopP;
        final Moe.Routing moeRouting;
        int latestToken;
        boolean logitsValid;   // computeLogits memo for the current chunk
        int lastRowOffset;     // offset into x of the row whose logits computeLogits finalizes

        State(Configuration config) {
            int c = Math.max(1, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
            this.capacity = c;
            int dim = config.embeddingLength;
            int maxQueryDim = config.numberOfHeads * config.headSizeFull;
            int maxKVDim = java.util.stream.IntStream.range(0, config.numberOfLayers).map(config::kvDim).max().orElse(0);
            int maxHiddenDim = config.maxHiddenDim();
            this.x = F32FloatTensor.allocate(c * dim);
            this.xb = F32FloatTensor.allocate(c * dim);
            this.xbK = F32FloatTensor.allocate(c * maxQueryDim);
            this.xb2 = F32FloatTensor.allocate(c * dim);
            this.hb = F32FloatTensor.allocate(c * maxHiddenDim);
            this.hb2 = F32FloatTensor.allocate(c * maxHiddenDim);
            this.q = F32FloatTensor.allocate(c * maxQueryDim);
            // k/v are single-row scratch used only by the reference single-token forward path;
            // forwardBatch uses the per-layer batchK/batchV buffers instead.
            this.k = SINGLE_TOKEN_PREFILL ? F32FloatTensor.allocate(c * maxKVDim) : null;
            this.v = SINGLE_TOKEN_PREFILL ? F32FloatTensor.allocate(c * maxKVDim) : null;
            this.att = F32FloatTensor.allocate(config.numberOfHeads * config.contextLength);
            this.logits = F32FloatTensor.allocate(config.vocabularySize);
            int plDim = config.embeddingLengthPerLayer;
            this.perLayerInputs = plDim > 0 ? F32FloatTensor.allocate(c * plDim * config.numberOfLayers) : null;
            this.plGate = plDim > 0 ? F32FloatTensor.allocate(c * plDim) : null;
            this.plProj = plDim > 0 ? F32FloatTensor.allocate(c * dim) : null;
            if (config.isMoE()) {
                int e = config.expertCount, tk = config.expertUsedCount;
                this.moeShared = F32FloatTensor.allocate(c * dim);
                this.moeInputB = F32FloatTensor.allocate(c * dim);
                this.moeRouterScaled = F32FloatTensor.allocate(c * dim);
                this.moeRouterB = F32FloatTensor.allocate(c * e);
                this.moeOutB = F32FloatTensor.allocate(c * dim);
                this.moeGather = F32FloatTensor.allocate(c * dim);
                this.moeDownB = F32FloatTensor.allocate(c * dim);
                this.moeExpertCounts = new int[e];
                this.moeExpertOffsets = new int[e + 1];
                this.moeCursor = new int[e];
                this.moeRowByExpert = new int[c * tk];
                this.moeRowTopE = new int[c * tk];
                this.moeProbByExpert = new float[c * tk];
                this.moeRowTopP = new float[c * tk];
                this.moeRouting = new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts, moeExpertOffsets, moeCursor, moeRowByExpert, moeProbByExpert);
            } else {
                this.moeShared = this.moeInputB = this.moeRouterScaled = this.moeRouterB = null;
                this.moeOutB = this.moeGather = this.moeDownB = null;
                this.moeExpertCounts = this.moeExpertOffsets = this.moeCursor = this.moeRowByExpert = this.moeRowTopE = null;
                this.moeProbByExpert = this.moeRowTopP = null;
                this.moeRouting = null;
            }
            this.keyCache = new FloatTensor[config.nLayerKvFromStart];
            this.valueCache = new FloatTensor[config.nLayerKvFromStart];
            this.batchK = new FloatTensor[config.nLayerKvFromStart];
            this.batchV = new FloatTensor[config.nLayerKvFromStart];
            for (int l = 0; l < config.nLayerKvFromStart; l++) {
                int kvDim = config.kvDim(l);
                int kvPositions = config.kvCachePositions(l);
                keyCache[l] = F16FloatTensor.allocate(kvPositions, kvDim);
                valueCache[l] = F16FloatTensor.allocate(kvPositions, kvDim);
                batchK[l] = F32FloatTensor.allocate(c * kvDim);
                batchV[l] = F32FloatTensor.allocate(c * kvDim);
            }
        }

        @Override public int latestToken() { return latestToken; }

        @Override public void latestToken(int token) { this.latestToken = token; }
    }

    // === Math helpers (Gemma RMS norm uses no +1 offset; the GGUF weights are pre-incremented) ===

    private static final float GELU_C = (float) Math.sqrt(2.0 / Math.PI);

    static float gelu(float x) {
        // tanh approximation; x*x*x (not Math.pow, which is ~10x slower)
        float inner = GELU_C * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + (float) Math.tanh(inner));
    }

    // SIMD GELU (VectorOperators.TANH): a big win ONLY where the JIT intrinsifies the SIMD
    // transcendental to SVML (HotSpot C2: ~20x; GraalVM JIT: ~2x SLOWER — no intrinsic). Off by
    // default (the engine's recommended runtime is GraalVM); enable on C2 with -Dgemma.vectorGelu.
    private static final boolean VECTOR_GELU = Boolean.getBoolean("gemma.vectorGelu");

    /**
     * Fused {@code gate[gateOff+i] = gelu(gate[gateOff+i]) * up[upOff+i]} over {@code n} elements.
     * Used for all gated activations (FFN, MoE, PLE) so every path stays bit-consistent (parity);
     * callers parallelize across rows.
     */
    static void geluMultiply(FloatTensor gate, int gateOff, FloatTensor up, int upOff, int n) {
        if (VECTOR_GELU && gate instanceof F32FloatTensor g && up instanceof F32FloatTensor u) {
            var sp = FloatTensor.F_SPECIES;
            int bound = sp.loopBound(n);
            for (int i = 0; i < bound; i += sp.length()) {
                long gb = (long) (gateOff + i) * Float.BYTES;
                long ub = (long) (upOff + i) * Float.BYTES;
                FloatVector x = FloatVector.fromMemorySegment(sp, g.vseg, g.vbase + gb, ByteOrder.LITTLE_ENDIAN);
                FloatVector uv = FloatVector.fromMemorySegment(sp, u.vseg, u.vbase + ub, ByteOrder.LITTLE_ENDIAN);
                FloatVector inner = x.mul(x).mul(x).mul(0.044715f).add(x).mul(GELU_C);
                FloatVector t = inner.lanewise(VectorOperators.TANH);
                x.mul(0.5f).mul(t.add(1.0f)).mul(uv)
                        .intoMemorySegment(g.vseg, g.vbase + gb, ByteOrder.LITTLE_ENDIAN);
            }
            for (int i = bound; i < n; i++) {
                gate.setFloat(gateOff + i, gelu(gate.getFloat(gateOff + i)) * up.getFloat(upOff + i));
            }
            return;
        }
        for (int i = 0; i < n; i++) {
            gate.setFloat(gateOff + i, gelu(gate.getFloat(gateOff + i)) * up.getFloat(upOff + i));
        }
    }

    // === Forward (single token at the given position) — reference path for parity testing ===

    /**
     * Single forward pass over {@code seqLen} tokens at {@code [startPos, startPos+seqLen)}: per-token
     * Q/K/V/O, shared MLP and PLE projections become GEMMs (GEMM short-circuits to GEMV at seqLen=1,
     * so one decode token flows through the same path), K/V written to the cache before attention,
     * which dispatches scalar single-token vs SWA flash per chunk. Leaves the post-final-layer residual
     * in {@code x}; {@link #computeLogits} finalizes the last row.
     */
    void forward(State state, int[] tokens, int tokenOffset, int startPos, int seqLen) {
        Configuration config = configuration;
        Weights weights = this.weights;
        int dim = config.embeddingLength;
        float sqrtDim = (float) Math.sqrt(dim);

        // Embed all rows + scale by sqrt(dim)
        for (int s = 0; s < seqLen; s++) {
            int token = tokens[tokenOffset + s];
            weights.tokenEmbeddingTable.copyTo(token * dim, state.x, s * dim, dim);
        }
        state.x.mapInPlace(0, seqLen * dim, v -> v * sqrtDim);

        // Per-layer inputs (PLE) for all rows
        int plDim = config.embeddingLengthPerLayer;
        int plTotal = plDim * config.numberOfLayers;
        if (plDim > 0 && weights.perLayerTokenEmbd != null) {
            float sqrtPlDim = (float) Math.sqrt(plDim);
            float projScale = (float) (1.0 / Math.sqrt(dim));
            float inputScale = (float) (1.0 / Math.sqrt(2.0));
            weights.perLayerModelProj.gemm(state.x, dim, state.perLayerInputs, plTotal, seqLen, plTotal, dim);
            int fPlDim = plDim, fPlTotal = plTotal, fNL = config.numberOfLayers;
            Parallel.forRows(seqLen, s -> {   // per-layer RMS-norm + token-embedding add, parallel over rows
                int base = s * fPlTotal;
                for (int i = 0; i < fPlTotal; i++) state.perLayerInputs.setFloat(base + i, state.perLayerInputs.getFloat(base + i) * projScale);
                for (int l = 0; l < fNL; l++) {
                    rmsnorm(state.perLayerInputs, base + l * fPlDim, state.perLayerInputs, base + l * fPlDim,
                            weights.perLayerProjNorm, fPlDim, config.rmsNormEps);
                }
                long tokEmbOffset = (long) tokens[tokenOffset + s] * fPlTotal;
                for (int i = 0; i < fPlTotal; i++) {
                    float tokEmb = weights.perLayerTokenEmbd.getFloat(tokEmbOffset + i) * sqrtPlDim;
                    state.perLayerInputs.setFloat(base + i, (state.perLayerInputs.getFloat(base + i) + tokEmb) * inputScale);
                }
            });
        }
        traceSum("inp_scaled", state.x, seqLen * dim);

        for (int l = 0; l < config.numberOfLayers; l++) {
            boolean layerIsSWA = config.isSWA[l];
            int headSize = config.headSize(l);
            int halfHead = headSize / 2;
            int queryDim = config.queryDim(l);
            int kvDim = config.kvDim(l);
            int hiddenDim = config.feedForwardLength[l];
            int nKvHeads = config.numberOfKeyValueHeadsPerLayer[l];
            int kvMul = config.numberOfHeads / nKvHeads;
            int kvLayer = config.kvSourceLayer(l);
            F32FloatTensor freqsReal = layerIsSWA ? weights.freqCisRealSwa : weights.freqCisRealFull;
            F32FloatTensor freqsImag = layerIsSWA ? weights.freqCisImagSwa : weights.freqCisImagFull;
            // final aliases so the per-row lambdas don't capture the loop counter / mutable locals
            int fDim = dim, fHeadSz = headSize, fHalf = halfHead, fQDim = queryDim, fNKv = nKvHeads, fStart = startPos;
            F32FloatTensor attNormW = weights.rmsAttWeight[l], qNormW = weights.attnQNorm[l], kNormW = weights.attnKNorm[l];
            F32FloatTensor postAttW = weights.postAttentionNorm[l], ffnNormW = weights.rmsFfnWeight[l], postFfwW = weights.postFfwNorm[l];

            // attention norm (rows are independent -> parallel)
            Parallel.forRows(seqLen, s ->
                    rmsnorm(state.xb, s * fDim, state.x, s * fDim, attNormW, fDim, config.rmsNormEps));

            // Q = wq @ xb (GEMM), per-head Q-norm + RoPE (rows parallel)
            weights.wq[l].gemm(state.xb, dim, state.q, queryDim, seqLen, queryDim, dim);
            Parallel.forRows(seqLen, s -> {
                for (int h = 0; h < config.numberOfHeads; h++) {
                    rmsnorm(state.q, s * fQDim + h * fHeadSz, state.q, s * fQDim + h * fHeadSz, qNormW, fHeadSz, config.rmsNormEps);
                }
                RoPE.applyNeox(state.q, s * fQDim, config.numberOfHeads, fHeadSz, fHalf, fStart + s, freqsReal, freqsImag);
            });

            // K/V projection into the per-layer LINEAR batch buffer (own-KV layers): norm + RoPE.
            // NOT written to the ring cache yet — committed at chunk end so prior reads stay intact.
            if (config.hasKv(l)) {
                FloatTensor bKl = state.batchK[l], bVl = state.batchV[l];
                weights.wk[l].gemm(state.xb, dim, bKl, kvDim, seqLen, kvDim, dim);
                if (weights.wv[l] != null) {
                    weights.wv[l].gemm(state.xb, dim, bVl, kvDim, seqLen, kvDim, dim);
                } else {
                    bKl.copyTo(0, bVl, 0, seqLen * kvDim);
                }
                int kd = kvDim;
                Parallel.forRows(seqLen, s -> {
                    for (int h = 0; h < fNKv; h++) {
                        rmsnorm(bKl, s * kd + h * fHeadSz, bKl, s * kd + h * fHeadSz, kNormW, fHeadSz, config.rmsNormEps);
                        rmsnormNoWeight(bVl, s * kd + h * fHeadSz, bVl, s * kd + h * fHeadSz, fHeadSz, config.rmsNormEps);
                    }
                    RoPE.applyNeox(bKl, s * kd, fNKv, fHeadSz, fHalf, fStart + s, freqsReal, freqsImag);
                });
            }

            // Batched causal attention (scale = 1.0): flash/tiled for prefill, simple 2-pass for
            // decode (a single query doesn't amortize the flash tiling/rescales).
            if (seqLen > 1) {
                flashAttention(state, l, startPos, seqLen, headSize, kvDim, queryDim, kvLayer, kvMul, layerIsSWA);
            } else {
                decodeAttention(state, l, startPos, headSize, kvDim, queryDim, kvLayer, kvMul, layerIsSWA);
            }

            // O = wo @ xbK (GEMM), post-attention norm + residual (rows parallel)
            weights.wo[l].gemm(state.xbK, queryDim, state.xb2, dim, seqLen, dim, queryDim);
            Parallel.forRows(seqLen, s ->
                    rmsnorm(state.xb2, s * fDim, state.xb2, s * fDim, postAttW, fDim, config.rmsNormEps));
            state.x.addInPlace(0, state.xb2, 0, seqLen * dim);

            // FFN
            if (config.isMoE() && weights.ffnGateInp[l] != null) {
                moeFfnBatch(state, l, dim, hiddenDim, seqLen);
            } else {
                Parallel.forRows(seqLen, s ->
                        rmsnorm(state.xb, s * fDim, state.x, s * fDim, ffnNormW, fDim, config.rmsNormEps));
                weights.w1[l].gemm(state.xb, dim, state.hb, hiddenDim, seqLen, hiddenDim, dim);
                weights.w3[l].gemm(state.xb, dim, state.hb2, hiddenDim, seqLen, hiddenDim, dim);
                int fHidden = hiddenDim;
                Parallel.forRows(seqLen, s -> geluMultiply(state.hb, s * fHidden, state.hb2, s * fHidden, fHidden));
                weights.w2[l].gemm(state.hb, hiddenDim, state.xb, dim, seqLen, dim, hiddenDim);
                Parallel.forRows(seqLen, s ->
                        rmsnorm(state.xb, s * fDim, state.xb, s * fDim, postFfwW, fDim, config.rmsNormEps));
                state.x.addInPlace(0, state.xb, 0, seqLen * dim);
            }

            // Per-layer embedding: GELU-gated projection (batched GEMMs, rows parallel)
            if (plDim > 0 && weights.perLayerInpGate != null) {
                int li = l, fPlDim = plDim, fPlTotal = plTotal;
                F32FloatTensor plPostW = weights.perLayerPostNorm[l];
                weights.perLayerInpGate[l].gemm(state.x, dim, state.plGate, plDim, seqLen, plDim, dim);
                Parallel.forRows(seqLen, s -> geluMultiply(state.plGate, s * fPlDim, state.perLayerInputs, s * fPlTotal + li * fPlDim, fPlDim));
                weights.perLayerProj[l].gemm(state.plGate, plDim, state.plProj, dim, seqLen, dim, plDim);
                Parallel.forRows(seqLen, s ->
                        rmsnorm(state.plProj, s * fDim, state.plProj, s * fDim, plPostW, fDim, config.rmsNormEps));
                state.x.addInPlace(0, state.plProj, 0, seqLen * dim);
            }

            float scale = weights.layerOutputScale[l];
            if (scale != 1.0f) {
                state.x.mapInPlace(0, seqLen * dim, v -> v * scale);
            }
            if (TRACE) traceSum("l_out-" + l, state.x, seqLen * dim);
        }

        // Commit the chunk's K/V to the ring caches for future chunks/decode (position order;
        // later positions overwrite the oldest slots, leaving the last `window` positions in place).
        for (int l = 0; l < config.nLayerKvFromStart; l++) {
            int kvDim = config.kvDim(l);
            for (int s = 0; s < seqLen; s++) {
                int kvPos = config.kvCacheIndex(l, startPos + s);
                state.batchK[l].copyTo(s * kvDim, state.keyCache[l], kvPos * kvDim, kvDim);
                state.batchV[l].copyTo(s * kvDim, state.valueCache[l], kvPos * kvDim, kvDim);
            }
        }

        state.lastRowOffset = (seqLen - 1) * dim;
        state.logitsValid = false;
    }

    /**
     * Gemma4 adapter: ring-SWA (or full) flash attention via the shared
     * {@link FlashAttention#slidingWindowPrefill} block. {@code scale = 1.0} (Gemma folds the QK scale
     * into the per-head Q-norm), no attention sinks. SWA layers ring their KV cache (slot =
     * {@code pos & (slidingWindow-1)}, power-of-two enforced in {@link Configuration}); full layers
     * store linearly. The batch K/V buffer is stride {@code kvDim}.
     */
    private void flashAttention(State state, int layer, int startPos, int seqLen,
                               int headSize, int kvDim, int queryDim, int kvLayer, int kvMul, boolean isSWA) {
        int window = isSWA ? configuration.slidingWindow : 0;
        int ringMask = isSWA ? configuration.slidingWindow - 1 : 0;
        FlashAttention.slidingWindowPrefill(state.q, state.xbK,
                state.keyCache[kvLayer], state.valueCache[kvLayer], state.batchK[kvLayer], state.batchV[kvLayer],
                configuration.numberOfHeads, startPos, seqLen, headSize, kvDim, queryDim, kvDim, kvMul,
                1.0f, window, ringMask, null);
    }

    /** Simple 2-pass causal attention for one query (decode), scale = 1.0: parallel over heads,
     *  reading the single in-chunk key from the batch buffer and prior keys from the ring cache. */
    private void decodeAttention(State state, int layer, int position,
                                int headSize, int kvDim, int queryDim, int kvLayer, int kvMul, boolean isSWA) {
        Configuration config = configuration;
        int window = config.slidingWindow;
        FloatTensor bK = state.batchK[kvLayer], bV = state.batchV[kvLayer];
        FloatTensor cK = state.keyCache[kvLayer], cV = state.valueCache[kvLayer];
        int attStart = isSWA ? Math.max(0, position - window + 1) : 0;
        Parallel.parallelFor(0, config.numberOfHeads, h -> {
            int attOff = h * config.contextLength;
            int kvHeadOffset = (h / kvMul) * headSize;
            int qOff = h * headSize;
            for (int t = attStart; t <= position; t++) {
                int off = (t >= position ? 0 : config.kvCacheIndex(layer, t) * kvDim) + kvHeadOffset;
                state.att.setFloat(attOff + t, state.q.dot(qOff, t >= position ? bK : cK, off, headSize));
            }
            state.att.softmaxInPlace(attOff + attStart, position - attStart + 1);
            state.xbK.fillInPlace(qOff, headSize, 0f);
            for (int t = attStart; t <= position; t++) {
                int off = (t >= position ? 0 : config.kvCacheIndex(layer, t) * kvDim) + kvHeadOffset;
                state.xbK.saxpyInPlace(qOff, t >= position ? bV : cV, off, headSize, state.att.getFloat(attOff + t));
            }
        });
    }

    /** Shared dense MLP + top-k expert MoE FFN (the A4B variant), batched across the whole chunk
     *  ({@code seqLen} rows packed at stride {@code dim} in {@code state.x}). The shared MLP runs as
     *  batched GEMMs; the expert FFN groups rows by routed expert (CSR) so each expert's weights are
     *  read once per chunk. Degenerates to per-row GEMV when {@code seqLen == 1} (decode). Output is
     *  {@code post_ffw_norm(post_norm_1(shared) + post_norm_2(experts))}, added to the residual. */
    private void moeFfnBatch(State state, int l, int dim, int hiddenDim, int seqLen) {
        Configuration config = configuration;
        Weights weights = this.weights;
        float eps = config.rmsNormEps;
        int nExperts = config.expertCount;
        int topK = config.expertUsedCount;
        int expertFF = config.expertFeedForwardLength;
        int gateUpDim = 2 * expertFF;
        F32FloatTensor gateInpScale = weights.ffnGateInpScale[l];
        float invSqrtDim = 1.0f / (float) Math.sqrt(dim);

        // --- Shared MLP (batched): ffn_norm -> gate/up/down -> post_norm_1 -> moeShared ---
        F32FloatTensor ffnNormW = weights.rmsFfnWeight[l], postNorm1 = weights.ffnPostNorm1[l];
        Parallel.forRows(seqLen, s -> rmsnorm(state.xb, s * dim, state.x, s * dim, ffnNormW, dim, eps));
        weights.w1[l].gemm(state.xb, dim, state.hb, hiddenDim, seqLen, hiddenDim, dim);
        weights.w3[l].gemm(state.xb, dim, state.hb2, hiddenDim, seqLen, hiddenDim, dim);
        Parallel.forRows(seqLen, s -> geluMultiply(state.hb, s * hiddenDim, state.hb2, s * hiddenDim, hiddenDim));
        weights.w2[l].gemm(state.hb, hiddenDim, state.moeShared, dim, seqLen, dim, hiddenDim);
        Parallel.forRows(seqLen, s -> rmsnorm(state.moeShared, s * dim, state.moeShared, s * dim, postNorm1, dim, eps));

        // --- Expert routing (batched): pre_ffw_norm2 -> expert input; rms-scaled x -> router input ---
        F32FloatTensor preNorm2 = weights.preFfwNorm2[l];
        Parallel.forRows(seqLen, s -> {
            rmsnorm(state.moeInputB, s * dim, state.x, s * dim, preNorm2, dim, eps);
            float ss = state.x.reduce(s * dim, dim, 0f, (acc, xi) -> acc + xi * xi);
            float rmsScale = (float) (1.0 / Math.sqrt(ss / dim + eps)) * invSqrtDim;
            for (int i = 0; i < dim; i++) {
                state.moeRouterScaled.setFloat(s * dim + i, state.x.getFloat(s * dim + i) * rmsScale * gateInpScale.getFloat(i));
            }
        });
        weights.ffnGateInp[l].gemm(state.moeRouterScaled, dim, state.moeRouterB, nExperts, seqLen, nExperts, dim);

        // Per-row softmax + top-k selection; bucket (row, prob) pairs by expert into CSR layout.
        int[] counts = state.moeExpertCounts;
        Arrays.fill(counts, 0);
        for (int s = 0; s < seqLen; s++) {
            state.moeRouterB.softmaxInPlace(s * nExperts, nExperts);
            for (int ki = 0; ki < topK; ki++) {
                int bestIdx = 0;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int ei = 0; ei < nExperts; ei++) {
                    float val = state.moeRouterB.getFloat(s * nExperts + ei);
                    if (val > bestVal) { bestVal = val; bestIdx = ei; }
                }
                state.moeRowTopE[s * topK + ki] = bestIdx;
                state.moeRowTopP[s * topK + ki] = bestVal;
                state.moeRouterB.setFloat(s * nExperts + bestIdx, Float.NEGATIVE_INFINITY);
                counts[bestIdx]++;
            }
        }
        // CSR grouping + gather + scatter is shared; the per-expert math (gated GELU, combined gate+up)
        // is here. The per-expert down scale folds into the combine weight via expertScale.
        Moe.Routing r = state.moeRouting;
        r.seqLen = seqLen; r.topK = topK; r.numExperts = nExperts;
        Moe.dispatch(r, dim, state.moeInputB, state.moeGather, state.moeDownB, state.moeOutB, weights.ffnDownExpsScale[l],
                (e, n, gather, out) -> {
                    // gate/up into state.hb (free after the shared MLP; maxHiddenDim >= gateUpDim), then GELU.
                    weights.ffnGateUpExps[l].gemm(gather, dim, state.hb, gateUpDim, n, gateUpDim, dim, e * gateUpDim * dim);
                    Parallel.forRows(n, j -> geluMultiply(state.hb, j * gateUpDim, state.hb, j * gateUpDim + expertFF, expertFF));
                    // down reads the first expertFF of each gateUpDim-strided row.
                    weights.ffnDownExps[l].gemm(state.hb, gateUpDim, out, dim, n, dim, expertFF, e * dim * expertFF);
                });

        // post_norm_2(experts) + shared, then post_ffw_norm, added to the residual (per row).
        F32FloatTensor postNorm2 = weights.ffnPostNorm2[l], postFfw = weights.postFfwNorm[l];
        Parallel.forRows(seqLen, s -> {
            rmsnorm(state.moeOutB, s * dim, state.moeOutB, s * dim, postNorm2, dim, eps);
            state.moeShared.addInPlace(s * dim, state.moeOutB, s * dim, dim);
            rmsnorm(state.moeShared, s * dim, state.moeShared, s * dim, postFfw, dim, eps);
            state.x.addInPlace(s * dim, state.moeShared, s * dim, dim);
        });
    }

    // === Loading ===

    static Gemma4 loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load Gemma4 model")) {
            try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                GGUF gguf = ModelLoader.readGguf(fileChannel, ggufPath.toString());
                return loadModel(fileChannel, gguf, contextLength, true);
            }
        }
    }

    static Gemma4 loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        LFMTokenizer tokenizer = new LFMTokenizer(gguf);

        int modelContextLength = gguf.getValue(int.class, "gemma4.context_length");
        if (contextLength < 0 || modelContextLength < contextLength) {
            contextLength = modelContextLength;
        }
        int embeddingLength = gguf.getValue(int.class, "gemma4.embedding_length");
        int numberOfHeads = gguf.getValue(int.class, "gemma4.attention.head_count");
        int numberOfLayers = gguf.getValue(int.class, "gemma4.block_count");
        int headSizeFull = gguf.getValue(int.class, "gemma4.attention.key_length");
        int headSizeSWA = gguf.getValue(int.class, "gemma4.attention.key_length_swa");
        int slidingWindow = gguf.getValue(int.class, "gemma4.attention.sliding_window");
        float logitSoftcapping = gguf.getValueOrDefault(float.class, "gemma4.final_logit_softcapping", 0f);
        float rmsNormEps = gguf.getValueOrDefault(float.class, "gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = gguf.getValueOrDefault(float.class, "gemma4.rope.freq_base", 1000000f);
        float ropeThetaSWA = gguf.getValueOrDefault(float.class, "gemma4.rope.freq_base_swa", 10000f);
        int expertCount = gguf.getValueOrDefault(int.class, "gemma4.expert_count", 0);
        int expertUsedCount = gguf.getValueOrDefault(int.class, "gemma4.expert_used_count", 0);
        int expertFeedForwardLength = gguf.getValueOrDefault(int.class, "gemma4.expert_feed_forward_length", 0);
        int embeddingLengthPerLayer = gguf.getValueOrDefault(int.class, "gemma4.embedding_length_per_layer_input", 0);
        int sharedKvLayers = gguf.getValueOrDefault(int.class, "gemma4.attention.shared_kv_layers", 0);
        int nLayerKvFromStart = numberOfLayers - sharedKvLayers;

        int[] feedForwardLength;
        Object ffnRaw = gguf.getValue(Object.class, "gemma4.feed_forward_length");
        if (ffnRaw instanceof int[] arr) {
            feedForwardLength = arr;
        } else {
            feedForwardLength = new int[numberOfLayers];
            Arrays.fill(feedForwardLength, (int) ffnRaw);
        }

        // Per-layer sliding-window vs full attention: prefer the explicit pattern, else derive
        // from the Q-norm size (== headSizeSWA for sliding-window layers).
        boolean[] isSWA = new boolean[numberOfLayers];
        Object swaRaw = gguf.getValueOrDefault(Object.class, "gemma4.attention.sliding_window_pattern", null);
        if (swaRaw instanceof boolean[] arr && arr.length == numberOfLayers) {
            isSWA = arr;
        } else {
            for (int i = 0; i < numberOfLayers; i++) {
                TensorEntry qNorm = gguf.getTensor("blk." + i + ".attn_q_norm.weight");
                isSWA[i] = qNorm != null
                        ? FloatTensor.numberOfElementsLong(Arrays.stream(qNorm.shape()).mapToInt(Math::toIntExact).toArray()) == headSizeSWA
                        : (i % 5 != 4);
            }
        }

        int[] numberOfKeyValueHeadsPerLayer = new int[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            TensorEntry kWeight = gguf.getTensor("blk." + i + ".attn_k.weight");
            int headSize = isSWA[i] ? headSizeSWA : headSizeFull;
            numberOfKeyValueHeadsPerLayer[i] = kWeight != null ? Math.toIntExact(kWeight.shape()[1]) / headSize : numberOfHeads;
        }

        Configuration config = new Configuration(embeddingLength, feedForwardLength, numberOfLayers, numberOfHeads,
                numberOfKeyValueHeadsPerLayer, tokenizer.vocabularySize(), contextLength, rmsNormEps, ropeTheta,
                ropeThetaSWA, headSizeFull, headSizeSWA, slidingWindow, logitSoftcapping, isSWA, nLayerKvFromStart,
                embeddingLengthPerLayer, expertCount, expertUsedCount, expertFeedForwardLength);

        // Shared-KV tail layers index their source layer's cache, so the KV shapes must match
        // (fail loudly at load rather than mis-index a future model with a different layout).
        for (int l = nLayerKvFromStart; l < numberOfLayers; l++) {
            int src = config.kvSourceLayer(l);
            if (src < 0 || src >= nLayerKvFromStart || config.kvDim(l) != config.kvDim(src)) {
                throw new IllegalStateException("layer " + l + " reuses KV from layer " + src
                        + " but their KV shapes differ (kvDim " + config.kvDim(l) + " vs " + config.kvDim(src) + ")");
            }
        }

        if (!loadWeightsFlag) {
            return new Gemma4(config, tokenizer, null);
        }
        Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fileChannel, gguf);
        return new Gemma4(config, tokenizer, loadWeights(tensors, config));
    }

    static Weights loadWeights(Map<String, GGMLTensorEntry> tensors, Configuration config) {
        int n = config.numberOfLayers;
        Pair<float[], float[]> ropeSwa = RoPE.precomputeFreqsCis(config.contextLength, config.headSizeSWA, config.ropeThetaSWA);
        float[] freqs = ModelLoader.ropeFreqFactors(tensors);
        Pair<float[], float[]> ropeFull = freqs != null
                ? RoPE.precomputeFreqsCisFromFreqs(config.contextLength, config.headSizeFull, config.ropeTheta, freqs)
                : RoPE.precomputeFreqsCis(config.contextLength, config.headSizeFull, config.ropeTheta);

        FloatTensor tokenEmbeddingTable = ModelLoader.loadQuantized(tensors.get("token_embd.weight"));

        float[] layerOutputScale = new float[n];
        for (int i = 0; i < n; i++) {
            GGMLTensorEntry scale = tensors.get("blk." + i + ".layer_output_scale.weight");
            layerOutputScale[i] = scale != null ? ModelLoader.toF32Tensor(scale).getFloat(0) : 1.0f;
        }

        FloatTensor[] wv = new FloatTensor[n];
        for (int i = 0; i < n; i++) {
            wv[i] = ModelLoader.quantOrNull(tensors, "blk." + i + ".attn_v.weight");
        }

        // Per-layer embeddings (PLE)
        FloatTensor perLayerTokenEmbd = null, perLayerModelProj = null;
        F32FloatTensor perLayerProjNorm = null;
        FloatTensor[] perLayerInpGate = null, perLayerProj = null;
        F32FloatTensor[] perLayerPostNorm = null;
        if (config.embeddingLengthPerLayer > 0 && tensors.containsKey("per_layer_token_embd.weight")) {
            perLayerTokenEmbd = ModelLoader.loadQuantized(tensors.get("per_layer_token_embd.weight"));
            perLayerModelProj = ModelLoader.loadQuantized(tensors.get("per_layer_model_proj.weight"));
            perLayerProjNorm = ModelLoader.toF32Tensor(tensors.get("per_layer_proj_norm.weight"));
            perLayerInpGate = ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".inp_gate.weight"));
            perLayerProj = ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".proj.weight"));
            perLayerPostNorm = ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".post_norm.weight"));
        }

        // MoE (A4B)
        FloatTensor[] ffnGateInp = null, ffnGateUpExps = null, ffnDownExps = null;
        F32FloatTensor[] ffnGateInpScale = null, ffnDownExpsScale = null, ffnPostNorm1 = null, preFfwNorm2 = null, ffnPostNorm2 = null;
        if (config.isMoE()) {
            ffnGateInp = ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate_inp.weight"));
            ffnGateInpScale = ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_gate_inp.scale"));
            ffnGateUpExps = ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate_up_exps.weight"));
            ffnDownExps = ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down_exps.weight"));
            ffnDownExpsScale = ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_down_exps.scale"));
            ffnPostNorm1 = ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".post_ffw_norm_1.weight"));
            preFfwNorm2 = ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".pre_ffw_norm_2.weight"));
            ffnPostNorm2 = ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".post_ffw_norm_2.weight"));
        }

        return new Weights(
                tokenEmbeddingTable,
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_q.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_k.weight")),
                wv,
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".attn_output.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_q_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".attn_k_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".post_attention_norm.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".ffn_norm.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_gate.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_down.weight")),
                ModelLoader.quantArray(n, i -> tensors.get("blk." + i + ".ffn_up.weight")),
                ModelLoader.f32Array(n, i -> tensors.get("blk." + i + ".post_ffw_norm.weight")),
                ModelLoader.toF32Tensor(tensors.get("output_norm.weight")),
                layerOutputScale,
                F32FloatTensor.of(ropeFull.first()), F32FloatTensor.of(ropeFull.second()),
                F32FloatTensor.of(ropeSwa.first()), F32FloatTensor.of(ropeSwa.second()),
                tensors.containsKey("output.weight") ? ModelLoader.loadQuantized(tensors.get("output.weight")) : tokenEmbeddingTable,
                perLayerTokenEmbd, perLayerModelProj, perLayerProjNorm, perLayerInpGate, perLayerProj, perLayerPostNorm,
                ffnGateInp, ffnGateInpScale, ffnGateUpExps, ffnDownExps, ffnDownExpsScale, ffnPostNorm1, preFfwNorm2, ffnPostNorm2);
    }

}

/**
 * Gemma's chat format as a small hand-written Java implementation rather than its (large, macro-
 * heavy) Jinja template — the template's whitespace-control trims aren't reproduced faithfully by
 * the in-tree renderer, which shifts the BOS boundary. This builds the exact prompt string and
 * tokenizes it with special-token mapping, matching llama.cpp token-for-token:
 *   {@code <bos>[<|turn>system\n[<|think|>][system]<turn|>\n]<|turn>{role}\n{content}<turn|>\n...<|turn>model\n}
 * Turn markers are {@code <|turn>} / {@code <turn|>}; thinking is enabled by injecting
 * {@code <|think|>} into a leading system turn (mirrors the template's {@code enable_thinking}).
 */
final class GemmaChatFormat implements ChatFormat {
    private final LFMTokenizer tokenizer;

    GemmaChatFormat(LFMTokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public List<Integer> encode(ChatContext ctx) {
        List<Object> messages = ctx.messages();
        StringBuilder sb = new StringBuilder("<bos>");

        boolean hasSystem = false;
        String systemText = null;
        if (!messages.isEmpty() && messages.getFirst() instanceof Map<?, ?> first) {
            String role = Values.stringValue(first.get("role"), "");
            if ("system".equals(role) || "developer".equals(role)) {
                hasSystem = true;
                systemText = Values.messageContent(first.get("content")).trim();
            }
        }

        if (ctx.enableThinking() || hasSystem) {
            sb.append("<|turn>system\n");
            if (ctx.enableThinking()) sb.append("<|think|>");
            if (hasSystem) sb.append(systemText);
            sb.append("<turn|>\n");
        }

        for (int i = hasSystem ? 1 : 0; i < messages.size(); i++) {
            if (!(messages.get(i) instanceof Map<?, ?> m)) continue;
            String role = Values.stringValue(m.get("role"), "user");
            if ("assistant".equals(role)) role = "model";
            sb.append("<|turn>").append(role).append('\n');
            sb.append(Values.messageContent(m.get("content")).trim());
            sb.append("<turn|>\n");
        }

        if (ctx.addGenerationPrompt()) {
            sb.append("<|turn>model\n");
        }
        return new ArrayList<>(tokenizer.encodeWithSpecialTokens(sb.toString()));
    }
}
