// LFM2.5 (Liquid Foundation Model 2.5): a Llama-architecture variant with short-convolution mixer
// layers interleaved with attention, dense or MoE FFN. The {@code Llama} record is this model's
// {@link Model} implementation (the name reflects the llama-family base; the plain standard-Llama
// transformer — Llama 3.x / MiniCPM / Mistral-3 / Granite — lives in {@link Llama3}). Also hosts the
// llama-family Configuration/Weights records {@link ModelLoader} fills and {@link PromptCache}
// understands. Single-token forward + batched prefill.
package com.qxotic.jinfer;

import static com.qxotic.jinfer.Norms.rmsnorm;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Optional;
import java.util.Set;

record Llama(Configuration configuration, LFMTokenizer tokenizer, Weights weights) implements Model {

    private static void rollingAttentionAccumulate(FloatTensor out, int outOffset, FloatTensor valueCache, int valueOffset,
                                                   int headSize, float oldScale, float scoreScale) {
        if (out instanceof F32FloatTensor outF32 && valueCache instanceof F16FloatTensor f16Value && FloatTensor.USE_VECTOR_API) {
            FloatVector oldScaleVector = FloatVector.broadcast(FloatTensor.F_SPECIES, oldScale);
            FloatVector scoreScaleVector = FloatVector.broadcast(FloatTensor.F_SPECIES, scoreScale);
            int upperBound = FloatTensor.F_SPECIES.loopBound(headSize);
            for (int i = 0; i < upperBound; i += FloatTensor.F_SPECIES.length()) {
                long outByteOffset = (long) (outOffset + i) * Float.BYTES;
                FloatVector acc = FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, outF32.vseg, outF32.vbase + outByteOffset, ByteOrder.LITTLE_ENDIAN).mul(oldScaleVector);
                // f16ToF32Vector inlined by hand for C2
                var bits32 = ShortVector.fromMemorySegment(FloatTensor.S_SPECIES_HALF, f16Value.vseg, f16Value.vbase + (long) (valueOffset + i) * Float16.BYTES, ByteOrder.LITTLE_ENDIAN)
                        .castShape(FloatTensor.I_SPECIES, 0).reinterpretAsInts();
                var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
                FloatVector value = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                        .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask))
                        .reinterpretAsFloats();
                value.fma(scoreScaleVector, acc).intoMemorySegment(outF32.vseg, outF32.vbase + outByteOffset, ByteOrder.LITTLE_ENDIAN);
            }
            for (int i = upperBound; i < headSize; i++) {
                outF32.setFloat(outOffset + i, outF32.getFloat(outOffset + i) * oldScale + valueCache.getFloat(valueOffset + i) * scoreScale);
            }
            return;
        }
        if (out instanceof F32FloatTensor outF32 && valueCache instanceof F32FloatTensor valueF32 && FloatTensor.USE_VECTOR_API) {
            FloatVector oldScaleVector = FloatVector.broadcast(FloatTensor.F_SPECIES, oldScale);
            FloatVector scoreScaleVector = FloatVector.broadcast(FloatTensor.F_SPECIES, scoreScale);
            int upperBound = FloatTensor.F_SPECIES.loopBound(headSize);
            for (int i = 0; i < upperBound; i += FloatTensor.F_SPECIES.length()) {
                long outByteOffset = (long) (outOffset + i) * Float.BYTES;
                FloatVector acc = FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, outF32.vseg, outF32.vbase + outByteOffset, ByteOrder.LITTLE_ENDIAN).mul(oldScaleVector);
                FloatVector value = FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, valueF32.vseg, valueF32.vbase + (long) (valueOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                value.fma(scoreScaleVector, acc).intoMemorySegment(outF32.vseg, outF32.vbase + outByteOffset, ByteOrder.LITTLE_ENDIAN);
            }
            for (int i = upperBound; i < headSize; i++) {
                outF32.setFloat(outOffset + i, outF32.getFloat(outOffset + i) * oldScale + valueF32.getFloat(valueOffset + i) * scoreScale);
            }
            return;
        }
        for (int i = 0; i < headSize; i++) {
            out.setFloat(outOffset + i, out.getFloat(outOffset + i) * oldScale + valueCache.getFloat(valueOffset + i) * scoreScale);
        }
    }

    private static int attentionStart(Configuration config, boolean isSWA, int position) {
        if (isSWA) {
            return Math.max(0, position - config.slidingWindow + 1);
        }
        if (RuntimeFlags.FULL_ATTENTION_WINDOW > 0) {
            return Math.max(0, position - RuntimeFlags.FULL_ATTENTION_WINDOW + 1);
        }
        return 0;
    }

    public State createNewState() {
        State state = new State(configuration());
        Integer bos = tokenizer.getSpecialTokens().get("<bos>");
        if (bos == null) bos = tokenizer.getSpecialTokens().get("<|startoftext|>");
        if (bos == null) bos = 1;
        state.latestToken = bos;
        return state;
    }

    // --- Model seam (delegates to this architecture's static forward pass) ---


    public int contextLength() {
        return configuration.contextLength;
    }


    public int vocabularySize() {
        return configuration.vocabularySize;
    }


    public int batchCapacity() {
        return RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH;
    }


    public void ingest(InferenceState state, int[] tokens, int tokenOffset, int startPosition, int sequenceLength) {
        ingestTokens(this, (State) state, tokens, tokenOffset, startPosition, sequenceLength);
    }


    public FloatTensor computeLogits(InferenceState state) {
        return computeLogits(this, (State) state);
    }


    public Set<Integer> stopTokens() {
        return new LFMChatFormat(tokenizer).getStopTokens();
    }


    public ChatFormat chatFormat() {
        return ChatFormats.forModel(tokenizer);
    }

    /** LFM2.5 supports the incremental prompt cache (its tree/bx tiers know this KV+conv layout). */

    public Optional<PromptCacheSupport> promptCacheSupport() {
        return Optional.of(new PromptCacheSupport() {

            public long kvBytesPerToken() {
                long bytes = 0;
                for (int l = 0; l < configuration.nLayerKvFromStart; l++) {
                    int kvDim = configuration.kvDim(l);
                    if (kvDim > 0) bytes += 2L * kvDim * Float16.BYTES;
                }
                return bytes;
            }


            public PromptCache create(CacheStore store) {
                return new PromptCache(configuration, store);
            }
        });
    }

    public static final class Configuration {
        public final int embeddingLength;
        public final int[] feedForwardLength; // per-layer (shared MLP)
        public final int numberOfLayers;
        public final int numberOfHeads;
        public final int[] numberOfKeyValueHeadsPerLayer; // per-layer KV head count
        public final int vocabularySize;
        public final int contextLength;
        public final float rmsNormEps;
        public final float ropeTheta;       // full attention RoPE theta
        public final float ropeThetaSWA;    // SWA RoPE theta
        public final int headSizeFull;      // head size for full attention layers
        public final int headSizeSWA;       // head size for SWA layers
        public final int slidingWindow;
        public final float logitSoftcapping;
        public final boolean[] isSWA;       // per-layer: true = SWA, false = full attention
        public final int nLayerKvFromStart; // first N layers have own KV cache, rest reuse
        // MoE fields
        public final int expertCount;           // 0 = dense model (no MoE)
        public final int expertUsedCount;       // top-k experts per token
        public final int expertFeedForwardLength; // expert FFN hidden dim
        public final int shortConvLCache;
        public final int leadingDenseBlockCount; // first N layers use dense FFN (rest use MoE)
        public final int expertGatingFunc;      // 1=softmax, 2=sigmoid

        public Configuration(int embeddingLength, int[] feedForwardLength, int numberOfLayers,
                             int numberOfHeads, int[] numberOfKeyValueHeadsPerLayer, int vocabularySize,
                             int contextLength, float rmsNormEps, float ropeTheta, float ropeThetaSWA,
                             int headSizeFull, int headSizeSWA, int slidingWindow,
                             float logitSoftcapping, boolean[] isSWA, int nLayerKvFromStart,
                             int expertCount, int expertUsedCount, int expertFeedForwardLength,
                             int shortConvLCache,
                             int leadingDenseBlockCount,
                             int expertGatingFunc) {
            if (slidingWindow <= 0 || Integer.bitCount(slidingWindow) != 1) {
                throw new IllegalArgumentException("slidingWindow must be a power of 2, got " + slidingWindow);
            }
            if (feedForwardLength.length != numberOfLayers
                    || numberOfKeyValueHeadsPerLayer.length != numberOfLayers
                    || isSWA.length != numberOfLayers) {
                throw new IllegalArgumentException("per-layer configuration arrays must have numberOfLayers entries");
            }
            this.embeddingLength = embeddingLength;
            this.feedForwardLength = Arrays.copyOf(feedForwardLength, feedForwardLength.length);
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeadsPerLayer = Arrays.copyOf(numberOfKeyValueHeadsPerLayer, numberOfKeyValueHeadsPerLayer.length);
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.ropeThetaSWA = ropeThetaSWA;
            this.headSizeFull = headSizeFull;
            this.headSizeSWA = headSizeSWA;
            this.slidingWindow = slidingWindow;
            this.logitSoftcapping = logitSoftcapping;
            this.isSWA = Arrays.copyOf(isSWA, isSWA.length);
            this.nLayerKvFromStart = nLayerKvFromStart;
            this.expertCount = expertCount;
            this.expertUsedCount = expertUsedCount;
            this.expertFeedForwardLength = expertFeedForwardLength;
            this.shortConvLCache = shortConvLCache;
            this.leadingDenseBlockCount = leadingDenseBlockCount;
            this.expertGatingFunc = expertGatingFunc;
        }

        public boolean isMoE() { return expertCount > 0; }
        public boolean isMoELayer(int layer) { return expertCount > 0 && layer >= leadingDenseBlockCount; }

        // For layers without own KV, return the layer whose cache to reuse
        public int kvSourceLayer(int layer) {
            if (layer < nLayerKvFromStart) return layer; // has own KV
            // Reuse the last KV layer of the same attention type
            return nLayerKvFromStart - (isSWA[layer] ? 2 : 1);
        }

        public int headSize(int layer) {
            return isSWA[layer] ? headSizeSWA : headSizeFull;
        }

        public int numberOfKeyValueHeads(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer];
        }

        public int kvCachePositions(int layer) {
            return isSWA[layer] ? Math.min(contextLength, slidingWindow) : contextLength;
        }

        public int kvCacheIndex(int layer, int position) {
            return isSWA[layer] ? (position & (slidingWindow - 1)) : position;
        }

        public int kvDim(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] * headSize(layer);
        }

        /** Widest per-layer kv row; the batch q/k/v buffers are strided by this constant. */
        public int maxKvDim() {
            int max = 0;
            for (int l = 0; l < numberOfLayers; l++) max = Math.max(max, kvDim(l));
            return max;
        }

        public int queryDim(int layer) {
            return numberOfHeads * headSize(layer);
        }

        public boolean isRecurrentLayer(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] == 0;
        }

        public int maxHiddenDim() {
            return Arrays.stream(feedForwardLength).max().orElseThrow();
        }

        public Configuration withContextLength(int newContextLength) {
            return new Configuration(embeddingLength, feedForwardLength, numberOfLayers,
                    numberOfHeads, numberOfKeyValueHeadsPerLayer, vocabularySize,
                    newContextLength, rmsNormEps, ropeTheta, ropeThetaSWA,
                    headSizeFull, headSizeSWA, slidingWindow,
                    logitSoftcapping, isSWA, nLayerKvFromStart,
                        expertCount, expertUsedCount, expertFeedForwardLength,
                    shortConvLCache,
                    leadingDenseBlockCount,
                    expertGatingFunc);
        }
    }

    /** Attention block weights (null on recurrent layers); wv null = the model shares K as V. */
    public record AttentionWeights(FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
                                   F32FloatTensor qNorm, F32FloatTensor kNorm) {}

    /** Short-convolution block weights (null on attention layers). {@code kernel} is the GGUF layout
     *  (per channel: dConv taps); {@code kernelTaps} is the same data tap-major (per tap: dim
     *  channels) so the scan's vector loop loads unit-stride. */
    public record ShortConvWeights(F32FloatTensor kernel, F32FloatTensor kernelTaps, FloatTensor inProj, FloatTensor outProj) {
        public static ShortConvWeights of(F32FloatTensor kernel, FloatTensor inProj, FloatTensor outProj, int dim) {
            int dConv = Math.toIntExact(kernel.size() / dim);
            F32FloatTensor taps = F32FloatTensor.allocate(dConv * dim);
            for (int k = 0; k < dConv; k++) {
                for (int c = 0; c < dim; c++) {
                    taps.setFloat(k * dim + c, kernel.getFloat((long) c * dConv + k));
                }
            }
            return new ShortConvWeights(kernel, taps, inProj, outProj);
        }
    }

    /** Dense MLP weights (null on MoE layers). */
    public record DenseFfnWeights(FloatTensor gate, FloatTensor down, FloatTensor up) {}

    /** Mixture-of-experts weights (null on dense layers); expProbsBias is optional. */
    public record MoeFfnWeights(FloatTensor router, FloatTensor gateExps, FloatTensor upExps,
                                FloatTensor downExps, F32FloatTensor expProbsBias) {}

    /** One layer's weights: exactly one of (attention | shortConv) and one of (dense | moe) is set. */
    public record LayerWeights(F32FloatTensor attnNorm, F32FloatTensor postAttnNorm,
                               F32FloatTensor ffnNorm, F32FloatTensor postFfnNorm,
                               float outputScale,
                               AttentionWeights attention, ShortConvWeights shortConv,
                               DenseFfnWeights dense, MoeFfnWeights moe) {}

    public record Weights(FloatTensor tokenEmbeddingTable,
                          LayerWeights[] layers,
                          F32FloatTensor finalNorm,
                          F32FloatTensor freqCisRealFull, F32FloatTensor freqCisImagFull,
                          F32FloatTensor freqCisRealSwa, F32FloatTensor freqCisImagSwa,
                          FloatTensor wcls) {}

    /** Persistent generation state: everything that survives across chunks — the kv caches,
     *  the rolling short-conv states, the last sampled token and the logits — plus the
     *  per-chunk activation scratch ({@link BatchState}). One State = one sequence. */
    public static final class State implements InferenceState {
        public final FloatTensor logits; // output logits (valid after computeLogits)
        // kv cache - variable sizes per layer
        public final FloatTensor[] keyCache;   // (n_layer, seq_len, kvDim_per_layer)
        public final FloatTensor[] valueCache; // (n_layer, seq_len, kvDim_per_layer)
        public final FloatTensor[] shortConvState; // recurrent layer -> (d_conv - 1, dim)
        public final BatchState batch;

        public int latestToken;
        /** Optional per-layer conv-input observer for {@link #ingestTokens} chunks (prompt
         *  cache bx harvesting); invoked after each recurrent layer's scan while
         *  {@link BatchState#shortConvTmp} still holds that layer's (b, c_gate, x) rows. */
        public ConvHarvest convHarvest;

        F32FloatTensor decodeBlockO; // (partitions * nHeads * headSize)
        float[] decodeBlockM;  // (partitions * nHeads)
        double[] decodeBlockL; // (partitions * nHeads)

        State(Configuration config) {
            this.logits = F32FloatTensor.allocate(config.vocabularySize);
            // Only allocate KV caches for layers that have their own KV (not shared)
            this.keyCache = new FloatTensor[config.nLayerKvFromStart];
            this.valueCache = new FloatTensor[config.nLayerKvFromStart];
            for (int l = 0; l < config.nLayerKvFromStart; l++) {
                int kvDim = config.kvDim(l);
                int kvPositions = config.kvCachePositions(l);
                keyCache[l] = F16FloatTensor.allocate(kvPositions, kvDim);
                valueCache[l] = F16FloatTensor.allocate(kvPositions, kvDim);
            }
            this.shortConvState = new FloatTensor[config.numberOfLayers];
            int dConv = Math.max(config.shortConvLCache - 1, 0);
            if (dConv > 0) {
                for (int l = 0; l < config.numberOfLayers; l++) {
                    if (config.isRecurrentLayer(l)) {
                        shortConvState[l] = F32FloatTensor.allocate(dConv * config.embeddingLength);
                    }
                }
            }
            this.batch = new BatchState(config, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH);
        }

         public int latestToken() { return latestToken; }

         public void latestToken(int token) { this.latestToken = token; }
    }

    /** Observer for {@link State#convHarvest}. */
    public interface ConvHarvest {
        void layer(int layer, BatchState seq);
    }

    /** Per-chunk activation scratch, overwritten by every {@code ingestTokens} call; carries no
     *  state between chunks beyond the pending-chunk bookkeeping (pendingPosition/pendingCount). */
    public static final class BatchState {
        public final int capacity;
        public final int dim;
        public int sequenceLength;
        public int pendingPosition;   // position of the most recently ingested chunk
        public int pendingCount;      // its length; 0 = nothing ever ingested
        public boolean logitsValid;   // computeLogits memo for the current pending chunk
        public final FloatTensor x;
        public final FloatTensor xb;
        public final FloatTensor xb_k;
        public final FloatTensor xb2;
        public final FloatTensor hb;
        public final FloatTensor hb2;
        public final FloatTensor q;
        public final FloatTensor k;
        public final FloatTensor v;
        public final FloatTensor att;   // only allocated if !RuntimeFlags.ROLLING_ATTENTION
        public final FloatTensor shortConvTmp;
        public final FloatTensor shortConvOut;
        public final FloatTensor routerLogits;
        public final FloatTensor moeInput;
        public final FloatTensor moeGate;
        public final FloatTensor moeUp;
        public final FloatTensor moeDown;
        public final int[] topExperts;
        public final float[] topProbs;
        final int moeTopK;
        final int[] moeRouteTokens;
        BatchState(Configuration config, int capacity) {
            this.capacity = capacity;
            this.dim = config.embeddingLength;
            int maxQueryDim = config.numberOfHeads * config.headSizeFull;
            int maxKVDim = config.maxKvDim();
            int maxHiddenDim = config.maxHiddenDim();
            this.x = F32FloatTensor.allocate(capacity * dim);
            this.xb = F32FloatTensor.allocate(capacity * dim);
            this.xb_k = F32FloatTensor.allocate(capacity * maxQueryDim);
            this.xb2 = F32FloatTensor.allocate(capacity * dim);
            this.hb = F32FloatTensor.allocate(capacity * maxHiddenDim);
            this.hb2 = F32FloatTensor.allocate(capacity * maxHiddenDim);
            this.q = F32FloatTensor.allocate(capacity * maxQueryDim);
            this.k = F32FloatTensor.allocate(capacity * maxKVDim);
            this.v = F32FloatTensor.allocate(capacity * maxKVDim);
            this.att = RuntimeFlags.ROLLING_ATTENTION ? null
                : F32FloatTensor.allocate(capacity * config.numberOfHeads * config.contextLength);
            this.shortConvTmp = F32FloatTensor.allocate(capacity * 3 * dim);
            this.shortConvOut = F32FloatTensor.allocate(capacity * dim);
            if (config.isMoE()) {
                int maxRoutes = capacity * config.expertUsedCount;
                this.moeTopK = config.expertUsedCount;
                this.routerLogits = F32FloatTensor.allocate(capacity * config.expertCount);
                this.moeInput = F32FloatTensor.allocate(maxRoutes * dim);
                this.moeGate = F32FloatTensor.allocate(maxRoutes * config.expertFeedForwardLength);
                this.moeUp = F32FloatTensor.allocate(maxRoutes * config.expertFeedForwardLength);
                this.moeDown = F32FloatTensor.allocate(maxRoutes * dim);
                this.topExperts = new int[maxRoutes];
                this.topProbs = new float[maxRoutes];
                this.moeRouteTokens = new int[maxRoutes];
            } else {
                this.moeTopK = 0;
                this.routerLogits = null;
                this.moeInput = null;
                this.moeGate = null;
                this.moeUp = null;
                this.moeDown = null;
                this.topExperts = null;
                this.topProbs = null;
                this.moeRouteTokens = null;
            }
        }
    }

    /** Apply Rotary Positional Embeddings (RoPE) to a tensor. */
    /** RMS-norm every pending position of x into xb with the given weight (llama.cpp build_norm). */
    static void rmsNormRows(Llama.Configuration config, Llama.BatchState batch, F32FloatTensor weight, int dim) {
        for (int s = 0; s < batch.sequenceLength; s++) {
            rmsnorm(batch.xb, s * dim, batch.x, s * dim, weight, dim, config.rmsNormEps);
        }
    }

    /** Short-conv layer: norm -> in-proj -> causal conv scan [-> out-proj + residual]. */
    static void forwardShortConvLayer(Llama.Configuration config, Llama.Weights weights,
                                              Llama.State state, Llama.BatchState seq, int layer, int dim, boolean needOutput) {
        rmsNormRows(config, seq, weights.layers()[layer].attnNorm(), dim);
        weights.layers()[layer].shortConv().inProj().gemm(seq.xb, dim, seq.shortConvTmp, 3 * dim, seq.sequenceLength, 3 * dim, dim);
        shortConvScan(config, weights, state, seq, layer, dim);
        if (!needOutput) return;
        shortConvOutput(weights, seq, layer, dim);
    }

    /**
     * The causal short-convolution over the chunk as a dConv-tap FIR over bx rows:
     * {@code out[s] = cg ∘ Σ_k taps[k] ∘ bx[s - hist + k]}, rows before the chunk coming from
     * shortConvState. bx = B∘x is materialized IN PLACE over the B rows of shortConvTmp —
     * positions then carry no loop dependency, and the rows double as the prompt-cache bx
     * harvest source (post-scan contract). The last hist bx rows roll into shortConvState.
     * The vector path accumulates taps in the scalar path's order (mul+add, ascending taps),
     * so both produce bit-identical outputs.
     */
    static void shortConvScan(Llama.Configuration config, Llama.Weights weights,
                              Llama.State state, Llama.BatchState seq, int layer, int dim) {
        FloatTensor convState = state.shortConvState[layer];
        int dConv = config.shortConvLCache;
        int hist = dConv - 1;
        if (FloatTensor.USE_VECTOR_API && FloatTensor.GLOBAL_SEGMENT != null && hist > 0
                && seq.shortConvTmp instanceof F32FloatTensor tmp
                && convState instanceof F32FloatTensor convF32
                && seq.xb2 instanceof F32FloatTensor out
                && dim % FloatTensor.F_SPECIES.length() == 0) {
            shortConvScanVector(tmp, convF32, out, weights.layers()[layer].shortConv().kernelTaps(),
                    seq.sequenceLength, dim, dConv);
            return;
        }
        F32FloatTensor kernel = weights.layers()[layer].shortConv().kernel();
        for (int s = 0; s < seq.sequenceLength; s++) {
            int tmpOffset = s * 3 * dim;
            int outOffset = s * dim;
            for (int c = 0; c < dim; c++) {
                float b = seq.shortConvTmp.getFloat(tmpOffset + c);
                float cg = seq.shortConvTmp.getFloat(tmpOffset + dim + c);
                float xv = seq.shortConvTmp.getFloat(tmpOffset + 2 * dim + c);
                float bx = b * xv;
                seq.shortConvTmp.setFloat(tmpOffset + c, bx); // post-scan contract: B row = bx
                int kBase = c * dConv;
                float sum = 0f;
                for (int k = 0; k < hist; k++) sum += convState.getFloat(k * dim + c) * kernel.getFloat(kBase + k);
                sum += bx * kernel.getFloat(kBase + dConv - 1);
                seq.xb2.setFloat(outOffset + c, cg * sum);
                for (int k = 0; k < hist - 1; k++) convState.setFloat(k * dim + c, convState.getFloat((k + 1) * dim + c));
                if (hist > 0) convState.setFloat((hist - 1) * dim + c, bx);
            }
        }
    }

    /** Vector FIR over materialized bx rows; row pointers resolve tap positions either into
     *  earlier (already materialized) bx rows of the chunk or into the pre-chunk conv state. */
    private static void shortConvScanVector(F32FloatTensor tmp, F32FloatTensor convState, F32FloatTensor out,
                                            F32FloatTensor taps, int seqLen, int dim, int dConv) {
        VectorSpecies<Float> species = FloatTensor.F_SPECIES;
        int lanes = species.length();
        int hist = dConv - 1;
        long rowBytes = (long) dim * Float.BYTES;
        for (int s = 0; s < seqLen; s++) {
            long bBase = tmp.vbase + s * 3 * rowBytes;
            long cgBase = bBase + rowBytes;
            long xBase = bBase + 2 * rowBytes;
            long outBase = out.vbase + s * rowBytes;
            for (int c = 0; c < dim; c += lanes) {
                long cb = (long) c * Float.BYTES;
                FloatVector bx = FloatVector.fromMemorySegment(species, tmp.vseg, bBase + cb, ByteOrder.LITTLE_ENDIAN)
                        .mul(FloatVector.fromMemorySegment(species, tmp.vseg, xBase + cb, ByteOrder.LITTLE_ENDIAN));
                bx.intoMemorySegment(tmp.vseg, bBase + cb, ByteOrder.LITTLE_ENDIAN);
                FloatVector sum = FloatVector.zero(species);
                for (int k = 0; k < hist; k++) {
                    FloatVector tap = FloatVector.fromMemorySegment(species, taps.vseg,
                            taps.vbase + k * rowBytes + cb, ByteOrder.LITTLE_ENDIAN);
                    // ONE statically-typed load site: the vector path requires GLOBAL_SEGMENT,
                    // where every tensor shares vseg and selection is pure address arithmetic
                    // (a segment phi/array would compile to SVM's generic path and segfault)
                    int pos = s - hist + k;
                    long rowBase = pos < 0
                            ? convState.vbase + (pos + hist) * rowBytes
                            : tmp.vbase + pos * 3 * rowBytes;
                    FloatVector row = FloatVector.fromMemorySegment(species, tmp.vseg, rowBase + cb, ByteOrder.LITTLE_ENDIAN);
                    sum = sum.add(row.mul(tap));
                }
                FloatVector lastTap = FloatVector.fromMemorySegment(species, taps.vseg,
                        taps.vbase + hist * rowBytes + cb, ByteOrder.LITTLE_ENDIAN);
                sum = sum.add(bx.mul(lastTap));
                FloatVector cg = FloatVector.fromMemorySegment(species, tmp.vseg, cgBase + cb, ByteOrder.LITTLE_ENDIAN);
                cg.mul(sum).intoMemorySegment(out.vseg, outBase + cb, ByteOrder.LITTLE_ENDIAN);
            }
        }
        // roll the conv state: row k = bx[seqLen - hist + k]; rows from before the chunk shift up
        for (int k = 0; k < hist; k++) {
            int pos = seqLen - hist + k;
            if (pos < 0) {
                MemorySegment.copy(convState.memorySegment, (long) (pos + hist) * rowBytes,
                        convState.memorySegment, (long) k * rowBytes, rowBytes);
            } else {
                MemorySegment.copy(tmp.memorySegment, (long) pos * 3 * rowBytes,
                        convState.memorySegment, (long) k * rowBytes, rowBytes);
            }
        }
    }

    /** Short-conv output: out-projection + residual add (deferred for the last layer). */
    static void shortConvOutput(Llama.Weights weights, Llama.BatchState seq, int layer, int dim) {
        weights.layers()[layer].shortConv().outProj().gemm(seq.xb2, dim, seq.shortConvOut, dim, seq.sequenceLength, dim, dim);
        for (int s = 0; s < seq.sequenceLength; s++) seq.x.addInPlace(s * dim, seq.shortConvOut, s * dim, dim);
    }

    /** Attention layer: norm -> qkv -> rope -> kv-store [-> attention -> output + residual]. */
    static void forwardAttentionLayer(Llama.Configuration config, Llama.Weights weights,
                                             Llama.State state, Llama.BatchState seq, int layer, int startPosition, int dim, boolean needOutput) {
        rmsNormRows(config, seq, weights.layers()[layer].attnNorm(), dim);
        projectQkv(config, weights, seq, layer, dim);
        qkNormRope(config, weights, seq, layer, startPosition);
        writeKvCache(config, state, seq, layer, startPosition);
        if (!needOutput) return;
        finishAttentionLayer(config, weights, state, seq, layer, startPosition, dim);
    }

    /** Q/K/V projections for the chunk (V falls back to K when the model shares them). */
    static void projectQkv(Llama.Configuration config, Llama.Weights weights, Llama.BatchState seq, int layer, int dim) {
        int queryDim = config.queryDim(layer);
        int kvDim = config.kvDim(layer);
        int maxQueryDim = config.numberOfHeads * config.headSizeFull;
        int maxKVDim = config.maxKvDim();
        int seqLen = seq.sequenceLength;
        Llama.AttentionWeights attn = weights.layers()[layer].attention();
        attn.wq().gemm(seq.xb, dim, seq.q, maxQueryDim, seqLen, queryDim, dim);
        attn.wk().gemm(seq.xb, dim, seq.k, maxKVDim, seqLen, kvDim, dim);
        if (attn.wv() != null) {
            attn.wv().gemm(seq.xb, dim, seq.v, maxKVDim, seqLen, kvDim, dim);
        } else {
            for (int s = 0; s < seqLen; s++) seq.k.copyTo(s * maxKVDim, seq.v, s * maxKVDim, kvDim);
        }
    }

    /** Per-head Q/K RMS-norms and rotary embedding at each pending position. */
    static void qkNormRope(Llama.Configuration config, Llama.Weights weights, Llama.BatchState seq, int layer, int startPosition) {
        boolean layerIsSWA = config.isSWA[layer];
        int headSize = config.headSize(layer);
        int halfHead = headSize / 2;
        int maxQueryDim = config.numberOfHeads * config.headSizeFull;
        int maxKVDim = config.maxKvDim();
        int nKvHeads = config.numberOfKeyValueHeads(layer);
        F32FloatTensor freqsReal = layerIsSWA ? weights.freqCisRealSwa() : weights.freqCisRealFull();
        F32FloatTensor freqsImag = layerIsSWA ? weights.freqCisImagSwa() : weights.freqCisImagFull();
        for (int s = 0; s < seq.sequenceLength; s++) {
            int position = startPosition + s;
            Llama.AttentionWeights attn = weights.layers()[layer].attention();
            for (int h = 0; h < config.numberOfHeads; h++) {
                rmsnorm(seq.q, s * maxQueryDim + h * headSize, seq.q, s * maxQueryDim + h * headSize, attn.qNorm(), headSize, config.rmsNormEps);
            }
            for (int h = 0; h < nKvHeads; h++) {
                rmsnorm(seq.k, s * maxKVDim + h * headSize, seq.k, s * maxKVDim + h * headSize, attn.kNorm(), headSize, config.rmsNormEps);
            }
            int ropePos = Math.max(0, Math.min(config.contextLength - 1, position));
            RoPE.applyNeox(seq.q, s * maxQueryDim, config.numberOfHeads, headSize, halfHead, ropePos, freqsReal, freqsImag);
            RoPE.applyNeox(seq.k, s * maxKVDim, nKvHeads, headSize, halfHead, ropePos, freqsReal, freqsImag);
        }
    }

    /** Append the chunk's K/V rows to the kv cache (SWA layers write their ring positions). */
    static void writeKvCache(Llama.Configuration config, Llama.State state, Llama.BatchState seq, int layer, int startPosition) {
        int kvDim = config.kvDim(layer);
        int maxKVDim = config.maxKvDim();
        int kvLayer = config.kvSourceLayer(layer);
        for (int s = 0; s < seq.sequenceLength; s++) {
            int kvPos = config.kvCacheIndex(layer, startPosition + s);
            seq.k.copyTo(s * maxKVDim, state.keyCache[kvLayer], kvPos * kvDim, kvDim);
            seq.v.copyTo(s * maxKVDim, state.valueCache[kvLayer], kvPos * kvDim, kvDim);
        }
    }

    /** LFM2.5 adapter: resolve this layer's config/state into the shared {@link FlashAttention#slidingWindowPrefill}
     *  block. SWA layers ring their KV cache (slot = {@code pos & (slidingWindow-1)}, hence the
     *  power-of-two requirement enforced in {@link Configuration}); full layers store linearly. */
    static void flashAttentionLayerSequence(Llama.Configuration config, Llama.Weights weights,
                                            Llama.State state, Llama.BatchState seq, int layer,
                                            int startPosition, int dim) {
        int headSize = config.headSize(layer);
        int nHeads = config.numberOfHeads;
        boolean isSWA = config.isSWA[layer];
        int kvLayer = config.kvSourceLayer(layer);
        int window = isSWA ? config.slidingWindow
                : (RuntimeFlags.FULL_ATTENTION_WINDOW > 0 ? RuntimeFlags.FULL_ATTENTION_WINDOW : 0);
        int ringMask = isSWA ? config.slidingWindow - 1 : 0;
        FlashAttention.slidingWindowPrefill(seq.q, seq.xb_k,
                state.keyCache[kvLayer], state.valueCache[kvLayer], seq.k, seq.v,
                nHeads, startPosition, seq.sequenceLength, headSize, config.kvDim(layer),
                nHeads * config.headSizeFull, config.maxKvDim(),
                nHeads / config.numberOfKeyValueHeads(layer),
                1.0f / (float) Math.sqrt(headSize), window, ringMask, null);
    }

    static boolean decodeAttentionBlockParallel(Llama.Configuration config, Llama.State state,
                                             Llama.BatchState seq, int layer, int startPosition) {
        int seqLen = seq.sequenceLength;
        if (seqLen != 1) return false;
        int position = startPosition;
        boolean layerIsSWA = config.isSWA[layer];
        int attStart = attentionStart(config, layerIsSWA, position);
        int attendedRange = position - attStart + 1;
        if (attendedRange < RuntimeFlags.DECODE_BLOCK_PARALLEL_MIN_RANGE) return false;

        int headSize = config.headSize(layer);
        int nHeads = config.numberOfHeads;
        int kvDim = config.kvDim(layer);
        int nKvHeads = config.numberOfKeyValueHeads(layer);
        int kvLayer = config.kvSourceLayer(layer);
        int kvMul = nHeads / nKvHeads;
        float attnScale = 1.0f / (float) Math.sqrt(headSize);

        if (!(seq.q instanceof F32FloatTensor qF32) || !(seq.xb_k instanceof F32FloatTensor outF32)
                || !(state.keyCache[kvLayer] instanceof F16FloatTensor keyCache)
                || !(state.valueCache[kvLayer] instanceof F16FloatTensor valueCache)) {
            return false;
        }

        int lanes = FloatTensor.F_SPECIES.length();

        int maxPartitions = Runtime.getRuntime().availableProcessors();
        int nPartitions = Math.max(1, Math.min(maxPartitions, attendedRange / RuntimeFlags.DECODE_BLOCK_SIZE + 1));

        int blockSize = attendedRange / nPartitions;
        int totalPartials = nPartitions * nHeads;
        int oFloats = totalPartials * headSize;

        if (state.decodeBlockO == null || state.decodeBlockO.size() < oFloats) {
            state.decodeBlockO = F32FloatTensor.allocate(oFloats);
        }
        if (state.decodeBlockM == null || state.decodeBlockM.length < totalPartials) {
            state.decodeBlockM = new float[totalPartials];
        }
        if (state.decodeBlockL == null || state.decodeBlockL.length < totalPartials) {
            state.decodeBlockL = new double[totalPartials];
        }

        F32FloatTensor blockO = state.decodeBlockO;
        float[] blockM = state.decodeBlockM;
        double[] blockL = state.decodeBlockL;

        // Partition bounds are pure functions of p, computed inline in each task (no per-call arrays).
        // The headSize-64 register-tiled kernel (rollingAttnCached64) decodes 8 f16 KV vectors per key
        // in one method; only a vector-intrinsifying compiler (Graal JIT / native AOT) keeps that
        // unboxed. On HotSpot C2 it boxes and collapses decode at long context, so C2 falls through to
        // the generic per-key path below (keyCache.dot + hand-inlined V decode), which never boxes.
        if (headSize == 64 && FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512
                && RuntimeFlags.GRAAL_VECTOR_JIT) {
            Parallel.parallelFor(0, nPartitions * nHeads, task -> {
                int p = task / nHeads;
                int h = task - p * nHeads;
                int pStart = attStart + p * blockSize;
                int pEnd = (p + 1 == nPartitions) ? position + 1 : attStart + (p + 1) * blockSize;
                decodeAttentionPartition64(config, layer, pStart, pEnd, attnScale,
                        qF32, h * headSize, keyCache, valueCache, kvDim, (h / kvMul) * headSize,
                        blockO, task * headSize, blockM, blockL, task);
            });
            mergeDecodePartitions(outF32, blockO, blockM, blockL, nPartitions, nHeads, headSize);
            return true;
        }
        Parallel.parallelFor(0, nPartitions, p -> {
            int pStart = attStart + p * blockSize;
            int pEnd = (p + 1 == nPartitions) ? position + 1 : attStart + (p + 1) * blockSize;
            for (int h = 0; h < nHeads; h++) {
                int kvHeadOffset = (h / kvMul) * headSize;
                int qOffset = h * headSize;
                int taskIdx = p * nHeads + h;
                int oOff = taskIdx * headSize;

                blockO.fillInPlace(oOff, headSize, 0f);
                float m = Float.NEGATIVE_INFINITY;
                double l = 0.0;

                for (int t = pStart; t < pEnd; t++) {
                    int cachePos = config.kvCacheIndex(layer, t);
                    int kOff = cachePos * kvDim + kvHeadOffset;
                    float score = attnScale * keyCache.dot(kOff, qF32, qOffset, headSize);
                    int vOff = cachePos * kvDim + kvHeadOffset;

                    float newMax = Math.max(m, score);
                    float oldScale = (float) Math.exp(m - newMax);
                    float scoreScale = (float) Math.exp(score - newMax);

                    int upperBound = FloatTensor.F_SPECIES.loopBound(headSize);
                    FloatVector oldVec = FloatVector.broadcast(FloatTensor.F_SPECIES, oldScale);
                    FloatVector scoreVec = FloatVector.broadcast(FloatTensor.F_SPECIES, scoreScale);
                    for (int d = 0; d < upperBound; d += lanes) {
                        long accByteOffset = blockO.vbase + (long) (oOff + d) * Float.BYTES;
                        FloatVector acc = FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, blockO.vseg, accByteOffset, ByteOrder.LITTLE_ENDIAN).mul(oldVec);
                        // f16ToF32Vector inlined by hand for C2 (see rollingAttentionAccumulate)
                        var bits32 = ShortVector.fromMemorySegment(FloatTensor.S_SPECIES_HALF, valueCache.vseg, valueCache.vbase + (long) (vOff + d) * Float16.BYTES, ByteOrder.LITTLE_ENDIAN)
                                .castShape(FloatTensor.I_SPECIES, 0).reinterpretAsInts();
                        var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
                        FloatVector v = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                                .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask))
                                .reinterpretAsFloats();
                        v.fma(scoreVec, acc).intoMemorySegment(blockO.vseg, accByteOffset, ByteOrder.LITTLE_ENDIAN);
                    }
                    for (int d = upperBound; d < headSize; d++) {
                        blockO.setFloat(oOff + d, blockO.getFloat(oOff + d) * oldScale + valueCache.getFloat(vOff + d) * scoreScale);
                    }
                    l = l * oldScale + scoreScale;
                    m = newMax;
                }

                blockM[taskIdx] = m;
                blockL[taskIdx] = l;
            }
        });

        mergeDecodePartitions(outF32, blockO, blockM, blockL, nPartitions, nHeads, headSize);
        return true;
    }

    /** Merge per-partition online-softmax partials (blockO rows + blockM/blockL stats) into out. */
    private static void mergeDecodePartitions(F32FloatTensor out, F32FloatTensor blockO,
                                              float[] blockM, double[] blockL,
                                              int nPartitions, int nHeads, int headSize) {
        Parallel.parallelFor(0, nHeads, h -> {
            int xbOffset = h * headSize;
            float mGlobal = Float.NEGATIVE_INFINITY;
            double lGlobal = 0.0;
            out.fillInPlace(xbOffset, headSize, 0f);

            for (int p = 0; p < nPartitions; p++) {
                int taskIdx = p * nHeads + h;
                float mBlock = blockM[taskIdx];
                if (mBlock == Float.NEGATIVE_INFINITY) continue;
                double lBlock = blockL[taskIdx];
                int oOff = taskIdx * headSize;

                float newMax = Math.max(mGlobal, mBlock);
                float globalScale = (float) Math.exp(mGlobal - newMax);
                float blockScale = (float) Math.exp(mBlock - newMax);

                int upperBound = FloatTensor.F_SPECIES.loopBound(headSize);
                FloatVector gVec = FloatVector.broadcast(FloatTensor.F_SPECIES, globalScale);
                FloatVector bVec = FloatVector.broadcast(FloatTensor.F_SPECIES, blockScale);
                for (int d = 0; d < upperBound; d += FloatTensor.F_SPECIES.length()) {
                    long byteOffset = (long) (xbOffset + d) * Float.BYTES;
                    FloatVector prev = FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, out.vseg, out.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN).mul(gVec);
                    FloatVector bv = FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, blockO.vseg, blockO.vbase + (long) (oOff + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN).mul(bVec);
                    prev.add(bv).intoMemorySegment(out.vseg, out.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN);
                }
                for (int d = upperBound; d < headSize; d++) {
                    out.setFloat(xbOffset + d, out.getFloat(xbOffset + d) * globalScale + blockO.getFloat(oOff + d) * blockScale);
                }
                lGlobal = lGlobal * globalScale + lBlock * blockScale;
                mGlobal = newMax;
            }

            if (lGlobal > 0.0) {
                FlashAttention.normalize(out, xbOffset, headSize, (float) (1.0 / lGlobal));
            }
        });
    }

    /**
     * Rolling (online-softmax) attention for one (token, head) with headSize 64 on 512-bit
     * vectors: query and output accumulator live in 8 zmm registers inside each phase, with
     * lazy max rescaling (common case per position: one dot, one exp, four fma — no memory
     * rescale). Split into two phase methods because a single method with ~100 vector ops
     * exceeds the Graal intrinsification budget and compiles to boxed vector calls; the
     * accumulator crosses the phase boundary through the output slot and (m, l) through a
     * packed long, so nothing boxes.
     */
    private static void rollingAttentionHead64(Llama.Configuration config, int layer,
                                               int position, int attStart, int startPosition, float attnScale,
                                               F32FloatTensor q, int qOffset,
                                               F16FloatTensor keyCache, F16FloatTensor valueCache, int kvDim, int kvHeadOffset,
                                               F32FloatTensor out, int xbOffset) {
        // The kv cache already holds the entire current chunk (written before the attention
        // finish), so the whole attended range reads from the f16 cache: one phase method, one
        // call. Keep this driver free of vector ops and with a single vector-heavy callee — a
        // caller whose inlined callees jointly exceed the Graal vector intrinsification budget
        // compiles with boxed vectors (~15x slower), nondeterministically (profile-dependent).
        rollingAttnCached64(config, layer, attStart, position + 1, attnScale,
                q, qOffset, keyCache, valueCache, kvDim, kvHeadOffset, out, xbOffset,
                packML(Float.NEGATIVE_INFINITY, 0f), true, true);
    }

    private static long packML(float m, float l) {
        return ((long) Float.floatToRawIntBits(m) << 32) | (Float.floatToRawIntBits(l) & 0xFFFFFFFFL);
    }

    private static float unpackM(long state) {
        return Float.intBitsToFloat((int) (state >>> 32));
    }

    private static float unpackL(long state) {
        return Float.intBitsToFloat((int) state);
    }

    private static void decodeAttentionPartition64(Llama.Configuration config, int layer, int tStart, int tEnd, float attnScale,
                                                   F32FloatTensor q, int qOffset,
                                                   F16FloatTensor keyCache, F16FloatTensor valueCache, int kvDim, int kvHeadOffset,
                                                   F32FloatTensor blockO, int oOff, float[] blockM, double[] blockL, int task) {
        long ml = rollingAttnCached64(config, layer, tStart, tEnd, attnScale,
                q, qOffset, keyCache, valueCache, kvDim, kvHeadOffset,
                blockO, oOff, packML(Float.NEGATIVE_INFINITY, 0f), true, false);
        blockM[task] = unpackM(ml);
        blockL[task] = unpackL(ml);
    }

    private static long rollingAttnCached64(Llama.Configuration config, int layer, int tStart, int tEnd, float attnScale,
                                            F32FloatTensor q, int qOffset,
                                            F16FloatTensor keyCache, F16FloatTensor valueCache, int kvDim, int kvHeadOffset,
                                            F32FloatTensor out, int xbOffset, long state, boolean initZero, boolean finalize) {
        var F = FloatTensor.F_SPECIES;
        final long qb = q.vbase + 4L * qOffset;
        final FloatVector q0 = FloatVector.fromMemorySegment(F, q.vseg, qb, ByteOrder.LITTLE_ENDIAN);
        final FloatVector q1 = FloatVector.fromMemorySegment(F, q.vseg, qb + 64, ByteOrder.LITTLE_ENDIAN);
        final FloatVector q2 = FloatVector.fromMemorySegment(F, q.vseg, qb + 128, ByteOrder.LITTLE_ENDIAN);
        final FloatVector q3 = FloatVector.fromMemorySegment(F, q.vseg, qb + 192, ByteOrder.LITTLE_ENDIAN);
        final long ob = out.vbase + 4L * xbOffset;
        FloatVector a0, a1, a2, a3;
        if (initZero) {
            a0 = FloatVector.zero(F); a1 = FloatVector.zero(F); a2 = FloatVector.zero(F); a3 = FloatVector.zero(F);
        } else {
            a0 = FloatVector.fromMemorySegment(F, out.vseg, ob, ByteOrder.LITTLE_ENDIAN);
            a1 = FloatVector.fromMemorySegment(F, out.vseg, ob + 64, ByteOrder.LITTLE_ENDIAN);
            a2 = FloatVector.fromMemorySegment(F, out.vseg, ob + 128, ByteOrder.LITTLE_ENDIAN);
            a3 = FloatVector.fromMemorySegment(F, out.vseg, ob + 192, ByteOrder.LITTLE_ENDIAN);
        }
        float m = unpackM(state);
        float l = unpackL(state);
        final MemorySegment kc = keyCache.vseg;
        final MemorySegment vc = valueCache.vseg;
        final long kcBase = keyCache.vbase;
        final long vcBase = valueCache.vbase;
        for (int t = tStart; t < tEnd; t++) {
            long off = (long) (config.kvCacheIndex(layer, t) * kvDim + kvHeadOffset) * Float16.BYTES;
            long base = kcBase + off;

            // 4x4 key tile: load all four keys, compute dot product with all four queries
            FloatVector k0 = f16DecodeVector(kc, base);
            FloatVector k1 = f16DecodeVector(kc, base + 32);
            FloatVector k2 = f16DecodeVector(kc, base + 64);
            FloatVector k3 = f16DecodeVector(kc, base + 96);
            var sv = q3.fma(k3, q2.fma(k2, q1.fma(k1, q0.mul(k0))));

            float score = sv.reduceLanes(VectorOperators.ADD) * attnScale;
            float p;
            if (score <= m) {
                p = (float) Math.exp(score - m);
            } else {
                float scale = (float) Math.exp(m - score);
                a0 = a0.mul(scale);
                a1 = a1.mul(scale);
                a2 = a2.mul(scale);
                a3 = a3.mul(scale);
                l *= scale;
                m = score;
                p = 1f;
            }
            l += p;
            var pv = FloatVector.broadcast(F, p);

            // 4x4 value tile: load all four values, accumulate into all four accumulators
            base = vcBase + off;
            FloatVector v0 = f16DecodeVector(vc, base);
            FloatVector v1 = f16DecodeVector(vc, base + 32);
            FloatVector v2 = f16DecodeVector(vc, base + 64);
            FloatVector v3 = f16DecodeVector(vc, base + 96);
            a0 = v0.fma(pv, a0);
            a1 = v1.fma(pv, a1);
            a2 = v2.fma(pv, a2);
            a3 = v3.fma(pv, a3);
        }
        if (finalize && l > 0f) {
            var iv = FloatVector.broadcast(F, 1f / l);
            a0 = a0.mul(iv); a1 = a1.mul(iv); a2 = a2.mul(iv); a3 = a3.mul(iv);
        }
        a0.intoMemorySegment(out.vseg, ob, ByteOrder.LITTLE_ENDIAN);
        a1.intoMemorySegment(out.vseg, ob + 64, ByteOrder.LITTLE_ENDIAN);
        a2.intoMemorySegment(out.vseg, ob + 128, ByteOrder.LITTLE_ENDIAN);
        a3.intoMemorySegment(out.vseg, ob + 192, ByteOrder.LITTLE_ENDIAN);
        return packML(m, l);
    }

    /**
     * Loads 16 f16 values from a memory segment, casts to ints, and decodes to a float vector.
     * Extracted so each call stays under Graal CE's Vector API expansion budget.
     */
    private static FloatVector f16DecodeVector(MemorySegment seg, long offset) {
        var bits32 = ShortVector.fromMemorySegment(FloatTensor.S_SPECIES_HALF, seg, offset, ByteOrder.LITTLE_ENDIAN)
                .castShape(FloatTensor.I_SPECIES, 0).reinterpretAsInts();
        var zem = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
        return bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zem))
                .reinterpretAsFloats();
    }

    /** The layer's deferred half: attention scores+mix, then output projection + residual. */
    static void finishAttentionLayer(Llama.Configuration config, Llama.Weights weights,
                                             Llama.State state, Llama.BatchState seq, int layer, int startPosition, int dim) {
        long t0 = Timing.ENABLED ? System.nanoTime() : 0L;
        attention(config, weights, state, seq, layer, startPosition, dim);
        if (Timing.ENABLED) Timing.addAttn(System.nanoTime() - t0);
        attentionOutput(config, weights, seq, layer, dim);
    }

    /** Causal attention over cached + in-chunk K/V into xb_k. Pure kernel dispatch: flash prefill
     *  kernel, block-parallel decode, or the rolling fallback — all compute the same thing for any
     *  chunk length. */
    static void attention(Llama.Configuration config, Llama.Weights weights,
                          Llama.State state, Llama.BatchState seq, int layer, int startPosition, int dim) {
        int maxQueryDim = config.numberOfHeads * config.headSizeFull;
        int maxKVDim = config.maxKvDim();
        int seqLen = seq.sequenceLength;
        if (RuntimeFlags.FLASH_ATTENTION && seq.sequenceLength > 1) {
            flashAttentionLayerSequence(config, weights, state, seq, layer, startPosition, dim);
        } else if (seqLen == 1 && decodeAttentionBlockParallel(config, state, seq, layer, startPosition)) {
            // block-parallel decode attention for long context
        } else {
            // fallback: per-position rolling attention for seqLen <= 1
            boolean layerIsSWA = config.isSWA[layer];
            int headSize = config.headSize(layer);
            int kvDim = config.kvDim(layer);
            int nKvHeads = config.numberOfKeyValueHeads(layer);
            int kvLayer = config.kvSourceLayer(layer);
            int kvMul = config.numberOfHeads / nKvHeads;
            float attnScale = 1.0f / (float) Math.sqrt(headSize);
            Parallel.parallelFor(0, seqLen * config.numberOfHeads, index -> {
                int s = index / config.numberOfHeads;
                int h = index - s * config.numberOfHeads;
                int position = startPosition + s;
                int attStart = attentionStart(config, layerIsSWA, position);
                int qOffset = s * maxQueryDim + h * headSize;
                int kvHeadOffset = (h / kvMul) * headSize;
                int xbOffset = s * maxQueryDim + h * headSize;
                if (!RuntimeFlags.ROLLING_ATTENTION) {
                    int attOffset = (s * config.numberOfHeads + h) * config.contextLength;
                    for (int t = attStart; t <= position; t++) {
                        float score;
                        if (t >= startPosition) {
                            int keyOffset = (t - startPosition) * maxKVDim + kvHeadOffset;
                            score = seq.q.dot(qOffset, seq.k, keyOffset, headSize) * attnScale;
                        } else {
                            int keyCacheOffset = config.kvCacheIndex(layer, t) * kvDim + kvHeadOffset;
                            score = seq.q.dot(qOffset, state.keyCache[kvLayer], keyCacheOffset, headSize) * attnScale;
                        }
                        seq.att.setFloat(attOffset + t, score);
                    }
                    seq.att.softmaxInPlace(attOffset + attStart, position - attStart + 1);
                    seq.xb_k.fillInPlace(xbOffset, headSize, 0f);
                    for (int t = attStart; t <= position; t++) {
                        int vOffset = config.kvCacheIndex(layer, t) * kvDim + kvHeadOffset;
                        float a = seq.att.getFloat(attOffset + t);
                        seq.xb_k.saxpyInPlace(xbOffset, state.valueCache[kvLayer], vOffset, headSize, a);
                    }
                    return;
                }
                if (headSize == 64 && FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512
                        && seq.q instanceof F32FloatTensor qf && seq.xb_k instanceof F32FloatTensor of
                        && state.keyCache[kvLayer] instanceof F16FloatTensor kcache
                        && state.valueCache[kvLayer] instanceof F16FloatTensor vcache) {
                    rollingAttentionHead64(config, layer, position, attStart, startPosition, attnScale,
                            qf, qOffset, kcache, vcache, kvDim, kvHeadOffset, of, xbOffset);
                    return;
                }
                float maxScore = Float.NEGATIVE_INFINITY;
                double sumExp = 0.0;
                seq.xb_k.fillInPlace(xbOffset, headSize, 0f);
                int t = attStart;
                while (t <= position) {
                    float score;
                    FloatTensor value;
                    int vOffset;
                    if (t >= startPosition) {
                        int localOffset = (t - startPosition) * maxKVDim + kvHeadOffset;
                        score = seq.q.dot(qOffset, seq.k, localOffset, headSize) * attnScale;
                        value = seq.v;
                        vOffset = localOffset;
                    } else {
                        int keyCacheOffset = config.kvCacheIndex(layer, t) * kvDim + kvHeadOffset;
                        score = seq.q.dot(qOffset, state.keyCache[kvLayer], keyCacheOffset, headSize) * attnScale;
                        value = state.valueCache[kvLayer];
                        vOffset = config.kvCacheIndex(layer, t) * kvDim + kvHeadOffset;
                    }
                    float newMax = Math.max(maxScore, score);
                    float oldScale = (float) Math.exp(maxScore - newMax);
                    float scoreScale = (float) Math.exp(score - newMax);
                    rollingAttentionAccumulate(seq.xb_k, xbOffset, value, vOffset, headSize, oldScale, scoreScale);
                    sumExp = sumExp * oldScale + scoreScale;
                    maxScore = newMax;
                    t++;
                }
                float invSumExp = (float) (1.0 / sumExp);
                FlashAttention.normalize(seq.xb_k, xbOffset, headSize, invSumExp);
            });
        }
    }

    /** Attention output projection, optional post-attention norm, residual add into x. */
    static void attentionOutput(Llama.Configuration config, Llama.Weights weights, Llama.BatchState seq, int layer, int dim) {
        int queryDim = config.queryDim(layer);
        int maxQueryDim = config.numberOfHeads * config.headSizeFull;
        int seqLen = seq.sequenceLength;
        weights.layers()[layer].attention().wo().gemm(seq.xb_k, maxQueryDim, seq.xb2, dim, seqLen, dim, queryDim);
        if (weights.layers()[layer].postAttnNorm() != null) {
            for (int s = 0; s < seqLen; s++) rmsnorm(seq.xb2, s * dim, seq.xb2, s * dim, weights.layers()[layer].postAttnNorm(), dim, config.rmsNormEps);
        }
        for (int s = 0; s < seqLen; s++) seq.x.addInPlace(s * dim, seq.xb2, s * dim, dim);
    }

    /**
     * Low-level token-stream access: extends the stream by one chunk at the given position.
     * Callable repeatedly with consecutive positions (prefill chunks and decode steps alike);
     * chunks are bounded by {@code state.batch.capacity} and by the context length (kv-cache
     * writes are unchecked, so overflow is rejected here instead of corrupting memory).
     * Promises no outputs: implementations may skip or defer any computation that is not
     * observable through {@link #computeLogits} (currently the final layer's attention finish
     * and FFN are deferred and later computed for the last row only).
     */
    static void ingestTokens(Llama model, Llama.State state, int[] tokens, int tokenOffset,
                              int startPosition, int sequenceLength) {
        if (sequenceLength > state.batch.capacity) {
            throw new IllegalArgumentException(
                "sequenceLength " + sequenceLength + " exceeds batch capacity " + state.batch.capacity);
        }
        Llama.Configuration config = model.configuration();
        if (startPosition + sequenceLength > config.contextLength) {
            throw new IllegalArgumentException(
                "tokens [" + startPosition + ", " + (startPosition + sequenceLength) + ") exceed context length " + config.contextLength);
        }
        Llama.Weights weights = model.weights();
        int dim = config.embeddingLength;
        Llama.BatchState seq = state.batch;
        seq.sequenceLength = sequenceLength;
        seq.pendingPosition = startPosition;
        seq.pendingCount = sequenceLength;

        embedTokens(weights, seq, tokens, tokenOffset, sequenceLength, dim, config.vocabularySize);

        for (int l = 0; l < config.numberOfLayers; l++) {
            boolean needOutput = l + 1 < config.numberOfLayers;
            forwardLayer(config, weights, state, seq, l, startPosition, sequenceLength, dim, needOutput);
            if (state.convHarvest != null && config.isRecurrentLayer(l)) {
                state.convHarvest.layer(l, seq); // shortConvTmp is intact until the next recurrent layer
            }
            if (!needOutput) break;
        }

        state.latestToken = tokens[tokenOffset + sequenceLength - 1];
        seq.logitsValid = false;
    }

    static void embedTokens(Llama.Weights weights, Llama.BatchState batch, int[] tokens, int tokenOffset, int sequenceLength, int dim, int vocabSize) {
        for (int s = 0; s < sequenceLength; s++) {
            int token = tokens[tokenOffset + s];
            if (token < 0 || token >= vocabSize) {
                throw new IllegalArgumentException(
                    "token id " + token + " out of range [0, " + vocabSize + ")");
            }
            weights.tokenEmbeddingTable().copyTo(token * dim, batch.x, s * dim, dim);
        }
    }

    /** One layer over the pending chunk — identical for prefill chunks and decode steps.
     *  Per-layer graph, llama.cpp style: norm -> (short-conv | qkv -> rope -> kv-store -> attention)
     *  -> output + residual -> ffn. needOutput=false defers everything after
     *  the kv-store/conv-scan for the last layer (computeLogits finishes the last row only). */
    static void forwardLayer(Llama.Configuration config, Llama.Weights weights, Llama.State state, Llama.BatchState batch,
                             int layer, int startPosition, int sequenceLength, int dim, boolean needOutput) {
        if (config.isRecurrentLayer(layer)) {
            forwardShortConvLayer(config, weights, state, batch, layer, dim, needOutput);
        } else {
            forwardAttentionLayer(config, weights, state, batch, layer, startPosition, dim, needOutput);
        }
        if (!needOutput) return;
        forwardFfn(config, weights, batch, layer, sequenceLength, dim);
        float scale = weights.layers()[layer].outputScale();
        if (scale != 1.0f) batch.x.mapInPlace(0, sequenceLength * dim, v -> v * scale);
    }

    /** FFN: norm -> dense MLP or MoE (router + experts), each adding its own residual. */
    static void forwardFfn(Llama.Configuration config, Llama.Weights weights, Llama.BatchState batch,
                           int layer, int sequenceLength, int dim) {
        rmsNormRows(config, batch, weights.layers()[layer].ffnNorm(), dim);
        if (config.isMoELayer(layer)) {
            moeFfn(config, weights, batch, layer, sequenceLength, dim);
        } else {
            denseFfn(config, weights, batch, layer, sequenceLength, dim);
        }
    }

    static void moeFfn(Llama.Configuration config, Llama.Weights weights, Llama.BatchState batch,
                              int layer, int sequenceLength, int dim) {
        int nExperts = config.expertCount;
        int topK = config.expertUsedCount;
        int expertFF = config.expertFeedForwardLength;
        int maxRoutes = sequenceLength * topK;
        Llama.MoeFfnWeights moe = weights.layers()[layer].moe();

        moe.router().gemm(batch.xb, dim, batch.routerLogits, nExperts, sequenceLength, nExperts, dim);
        for (int s = 0; s < sequenceLength; s++) {
            int routerOffset = s * nExperts;
            if (moe.expProbsBias() != null) {
                for (int i = 0; i < nExperts; i++) batch.routerLogits.setFloat(routerOffset + i, batch.routerLogits.getFloat(routerOffset + i) + moe.expProbsBias().getFloat(i));
            }
            if (config.expertGatingFunc == 2) {
                batch.routerLogits.mapInPlace(routerOffset, nExperts, Activations::sigmoid);
            } else {
                batch.routerLogits.softmaxInPlace(routerOffset, nExperts);
            }
            for (int ki = 0; ki < topK; ki++) {
                int bestIdx = 0;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int ei = 0; ei < nExperts; ei++) {
                    float val = batch.routerLogits.getFloat(routerOffset + ei);
                    if (val > bestVal) {
                        bestVal = val;
                        bestIdx = ei;
                    }
                }
                int route = s * topK + ki;
                batch.topExperts[route] = bestIdx;
                batch.topProbs[route] = bestVal;
                batch.routerLogits.setFloat(routerOffset + bestIdx, Float.NEGATIVE_INFINITY);
            }
            float weightSum = 0f;
            for (int ki = 0; ki < topK; ki++) weightSum += batch.topProbs[s * topK + ki];
            for (int ki = 0; ki < topK; ki++) batch.topProbs[s * topK + ki] /= weightSum;
        }
        batch.xb2.fillInPlace(0, sequenceLength * dim, 0f);
        if (sequenceLength == 1) {
            // decode fast path: one dispatch for all experts' gate+up rows, one for the downs,
            // reading xb directly (no per-expert gather copies, no 12 separate gemv round trips)
            final FloatTensor gateW = moe.gateExps();
            final FloatTensor upW = moe.upExps();
            final FloatTensor downW = moe.downExps();
            final int fDim = dim;
            final int fExpertFF = expertFF;
            Parallel.parallelFor(0, topK * 2 * expertFF, idx -> {
                int e = idx / (2 * fExpertFF);
                int rem = idx - e * 2 * fExpertFF;
                int expertBase = batch.topExperts[e] * fExpertFF * fDim;
                if (rem < fExpertFF) {
                    batch.moeGate.setFloat(e * fExpertFF + rem, gateW.dot(expertBase + rem * fDim, batch.xb, 0, fDim));
                } else {
                    int r = rem - fExpertFF;
                    batch.moeUp.setFloat(e * fExpertFF + r, upW.dot(expertBase + r * fDim, batch.xb, 0, fDim));
                }
            });
            batch.moeGate.siluMultiplyInPlace(0, batch.moeUp, 0, topK * expertFF);
            Parallel.parallelFor(0, topK * dim, idx -> {
                int e = idx / fDim;
                int r = idx - e * fDim;
                int expertBase = batch.topExperts[e] * fDim * fExpertFF;
                batch.moeDown.setFloat(e * fDim + r, downW.dot(expertBase + r * fExpertFF, batch.moeGate, e * fExpertFF, fExpertFF));
            });
            for (int e = 0; e < topK; e++) {
                batch.xb2.saxpyInPlace(0, batch.moeDown, e * dim, dim, batch.topProbs[e]);
            }
        } else {
        int[] routeTokens = batch.moeRouteTokens;
        for (int expertIdx = 0; expertIdx < nExperts; expertIdx++) {
            int count = 0;
            for (int route = 0; route < maxRoutes; route++) {
                if (batch.topExperts[route] == expertIdx) {
                    int tokenIndex = route / topK;
                    routeTokens[count] = route;
                    batch.xb.copyTo(tokenIndex * dim, batch.moeInput, count * dim, dim);
                    count++;
                }
            }
            if (count == 0) continue;
            long gateOffset = (long) expertIdx * expertFF * dim;
            long upOffset = (long) expertIdx * expertFF * dim;
            moe.gateExps().gemm(batch.moeInput, dim, batch.moeGate, expertFF, count, expertFF, dim, gateOffset);
            moe.upExps().gemm(batch.moeInput, dim, batch.moeUp, expertFF, count, expertFF, dim, upOffset);
            batch.moeGate.siluMultiplyInPlace(0, batch.moeUp, 0, count * expertFF);
            long downOffset = (long) expertIdx * dim * expertFF;
            moe.downExps().gemm(batch.moeGate, expertFF, batch.moeDown, dim, count, dim, expertFF, downOffset);

            for (int i = 0; i < count; i++) {
                int route = routeTokens[i];
                int tokenIndex = route / topK;
                batch.xb2.saxpyInPlace(tokenIndex * dim, batch.moeDown, i * dim, dim, batch.topProbs[route]);
            }
        }
        }
        for (int s = 0; s < sequenceLength; s++) {
            int xbOffset = s * dim;
            if (weights.layers()[layer].postFfnNorm() != null) rmsnorm(batch.xb2, xbOffset, batch.xb2, xbOffset, weights.layers()[layer].postFfnNorm(), dim, config.rmsNormEps);
            batch.x.addInPlace(xbOffset, batch.xb2, xbOffset, dim);
        }
    }

    static void denseFfn(Llama.Configuration config, Llama.Weights weights, Llama.BatchState batch,
                                int layer, int sequenceLength, int dim) {
        int hiddenDim = config.feedForwardLength[layer];
        Llama.DenseFfnWeights dense = weights.layers()[layer].dense();
        Ffn.dense(dense.gate(), dense.up(), dense.down(), batch.xb, batch.hb, batch.hb2, batch.xb,
                sequenceLength, dim, hiddenDim, Ffn.Act.SILU_GLU);
        F32FloatTensor postFfnNorm = weights.layers()[layer].postFfnNorm();
        if (postFfnNorm != null) {
            for (int s = 0; s < sequenceLength; s++) rmsnorm(batch.xb, s * dim, batch.xb, s * dim, postFfnNorm, dim, config.rmsNormEps);
        }
        for (int s = 0; s < sequenceLength; s++) batch.x.addInPlace(s * dim, batch.xb, s * dim, dim);
    }

    static int finishFinalLayerForLogits(Llama.Configuration config, Llama.Weights weights,
                                          Llama.State state, Llama.BatchState batch, int dim) {
        int layer = config.numberOfLayers - 1;
        int sequenceLength = batch.pendingCount;
        if (config.isRecurrentLayer(layer)) {
            shortConvOutput(weights, batch, layer, dim);
        } else {
            finishAttentionLayer(config, weights, state, batch, layer, batch.pendingPosition, dim);
        }
        if (RuntimeFlags.LAST_ROW_LOGITS) {
            // Only the last token's logits are observable, so run the deferred FFN for that row
            // alone (a 1024-row prefill chunk would otherwise pay the full last-layer FFN for
            // 1023 rows nobody reads). Move the row to index 0 and reuse the 1-token paths.
            int last = sequenceLength - 1;
            if (last != 0) {
                batch.x.copyTo(last * dim, batch.x, 0, dim);
            }
            sequenceLength = 1;
        }
        forwardFfn(config, weights, batch, layer, sequenceLength, dim);
        float scale = weights.layers()[layer].outputScale();
        if (scale != 1.0f) batch.x.mapInPlace(0, sequenceLength * dim, v -> v * scale);
        return (sequenceLength - 1) * dim; // offset of the last token's row in batch.x
    }

    /**
     * Logits for the LAST ingested token. Idempotent: repeat calls without an intervening
     * {@link #ingestTokens} return the same tensor with no recomputation. Completes any work
     * the implementation deferred during ingestion (an implementation detail; an eager
     * implementation would make this a plain accessor).
     */
    static FloatTensor computeLogits(Llama model, Llama.State state) {
        if (state.batch.logitsValid) {
            return state.logits;
        }
        if (state.batch.pendingCount == 0) {
            throw new IllegalStateException("no tokens ingested; call ingestTokens first");
        }
        Llama.Configuration config = model.configuration();
        Llama.Weights weights = model.weights();
        int dim = config.embeddingLength;
        int lastOffset = finishFinalLayerForLogits(config, weights, state, state.batch, dim);
        // final norm of the last row into xb (free after the FFN), then the output head
        rmsnorm(state.batch.xb, 0, state.batch.x, lastOffset, weights.finalNorm(), dim, config.rmsNormEps);
        weights.wcls().gemv(state.batch.xb, state.logits, config.vocabularySize, dim);
        if (config.logitSoftcapping > 0) {
            float cap = config.logitSoftcapping;
            state.logits.mapInPlace(v -> cap * (float) Math.tanh(v / cap));
        }
        state.batch.logitsValid = true;
        return state.logits;
    }

}
