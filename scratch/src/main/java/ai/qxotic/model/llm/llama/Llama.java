package ai.qxotic.model.llm.llama;

import ai.qxotic.span.FloatMatrixView;
import ai.qxotic.span.FloatSpan;
import ai.qxotic.span.FloatUnaryOperator;
import ai.qxotic.span.KernelOps;
import ai.qxotic.tokenizers.IntSequence;
import java.util.function.UnaryOperator;
import java.util.stream.Stream;

public class Llama extends AbstractModel<Llama.Configuration, Llama.Weights, Llama.State> {

    public Llama(
            Configuration configuration,
            KernelOps<FloatSpan, FloatMatrixView> kernelOps,
            FloatSpanFactory<? extends FloatSpan> spanFactory) {
        super(configuration, kernelOps, spanFactory);
    }

    @Override
    public Configuration configuration() {
        return configuration;
    }

    public static class Configuration {
        public final int embeddingLength; // transformer dimension
        public final int ffnLength; // for ffn layers (hiddenDim)
        public final int numberOfLayers; // number of layers
        public final int numberOfHeads; // number of query heads
        public final int
                numberOfKeyValueHeads; // number of key/value heads (can be < query heads because of
        // multiquery)
        public final int vocabularySize; // vocabulary size
        public final int contextLength; // max sequence length
        public final float rmsNormEps;
        public final float ropeTheta;
        public final int headSize;
        public final boolean ropeIsNeoxStyle;

        public final float attentionScale;

        // For Granite model.
        public final float residualScale;
        public final float logitScale;
        public final float embeddingScale;

        public final int keyHeadSize;
        public final int valueHeadSize;

        public final FloatUnaryOperator activationFunction;
        public final FloatUnaryOperator[] activationFunctionPerLayer;

        public Configuration(
                int embeddingLength,
                int ffnLength,
                int numberOfLayers,
                int numberOfHeads,
                int keyHeadSize,
                int valueHeadSize,
                int numberOfKeyValueHeads,
                int vocabularySize,
                int contextLength,
                float rmsNormEps,
                float ropeTheta,
                boolean ropeIsNeoxStyle) {
            this(
                    embeddingLength,
                    ffnLength,
                    numberOfLayers,
                    numberOfHeads,
                    keyHeadSize,
                    valueHeadSize,
                    numberOfKeyValueHeads,
                    vocabularySize,
                    contextLength,
                    rmsNormEps,
                    ropeTheta,
                    ropeIsNeoxStyle,
                    Float.NaN,
                    Float.NaN,
                    Float.NaN,
                    Float.NaN,
                    FloatUnaryOperator.SILU,
                    null);
        }

        public Configuration(
                int embeddingLength,
                int ffnLength,
                int numberOfLayers,
                int numberOfHeads,
                int keyHeadSize,
                int valueHeadSize,
                int numberOfKeyValueHeads,
                int vocabularySize,
                int contextLength,
                float rmsNormEps,
                float ropeTheta,
                boolean ropeIsNeoxStyle,
                float attentionScale,
                float residualScale,
                float logitScale,
                float embeddingScale,
                FloatUnaryOperator activationFunction,
                FloatUnaryOperator[] activationFunctionPerLayer) {
            this.embeddingLength = embeddingLength;

            this.ffnLength = ffnLength;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;

            this.keyHeadSize = keyHeadSize;
            this.valueHeadSize = valueHeadSize;

            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.ropeIsNeoxStyle = ropeIsNeoxStyle;
            this.residualScale = residualScale;
            this.logitScale = logitScale;
            this.embeddingScale = embeddingScale;
            this.activationFunction = activationFunction;
            this.activationFunctionPerLayer = activationFunctionPerLayer;

            // Derived parameters.
            this.headSize = embeddingLength / numberOfHeads;
            // Derive if not specified.
            this.attentionScale =
                    Float.isNaN(attentionScale)
                            ? (float) (1.0 / Math.sqrt(keyHeadSize))
                            : attentionScale;
        }

        public Configuration with(UnaryOperator<Builder> modifier) {
            return modifier.apply(new Builder(this)).build();
        }

        public static class Builder {
            int embeddingLength; // transformer dimension
            int ffnLength; // for ffn layers (hiddenDim)
            int numberOfLayers; // number of layers
            int numberOfHeads; // number of query heads

            int keyHeadSize; // llama.cpp's n_embd_head_k
            int valueHeadSize; // llama.cpp's n_embd_head_v

            int numberOfKeyValueHeads; // number of key/value heads (can be < query heads
            // because of multiquery)
            int vocabularySize; // vocabulary size, usually 256 (byte-level)
            int contextLength; // max sequence length
            float rmsNormEps;
            float ropeTheta;
            boolean ropeIsNeoxStyle;
            float attentionScale;

            // For Granite model.
            float residualScale;
            float logitScale;
            float embeddingScale;
            FloatUnaryOperator activationFunction;
            FloatUnaryOperator[] activationFunctionPerLayer;

            public Builder embeddingLength(int embeddingLength) {
                this.embeddingLength = embeddingLength;
                return this;
            }

            public Builder ffnLength(int ffnLength) {
                this.ffnLength = ffnLength;
                return this;
            }

            public Builder numberOfLayers(int numberOfLayers) {
                this.numberOfLayers = numberOfLayers;
                return this;
            }

            public Builder numberOfHeads(int numberOfHeads) {
                this.numberOfHeads = numberOfHeads;
                return this;
            }

            public Builder numberOfKeyValueHeads(int numberOfKeyValueHeads) {
                this.numberOfKeyValueHeads = numberOfKeyValueHeads;
                return this;
            }

            public Builder vocabularySize(int vocabularySize) {
                this.vocabularySize = vocabularySize;
                return this;
            }

            public Builder contextLength(int contextLength) {
                this.contextLength = contextLength;
                return this;
            }

            public Builder rmsNormEps(float rmsNormEps) {
                this.rmsNormEps = rmsNormEps;
                return this;
            }

            public Builder ropeTheta(float ropeTheta) {
                this.ropeTheta = ropeTheta;
                return this;
            }

            public Builder ropeIsNeoxStyle(boolean ropeIsNeoxStyle) {
                this.ropeIsNeoxStyle = ropeIsNeoxStyle;
                return this;
            }

            public Builder residualScale(float residualScale) {
                this.residualScale = residualScale;
                return this;
            }

            public Builder logitScale(float logitScale) {
                this.logitScale = logitScale;
                return this;
            }

            public Builder attentionScale(float attentionScale) {
                this.attentionScale = attentionScale;
                return this;
            }

            public Builder embeddingScale(float embeddingScale) {
                this.embeddingScale = embeddingScale;
                return this;
            }

            public Builder activationFunction(FloatUnaryOperator activationFunction) {
                this.activationFunction = activationFunction;
                return this;
            }

            public Builder activationFunctionPerLayer(
                    FloatUnaryOperator[] activationFunctionPerLayer) {
                this.activationFunctionPerLayer = activationFunctionPerLayer;
                return this;
            }

            private Builder(Configuration configuration) {
                this.embeddingLength = configuration.embeddingLength;
                this.ffnLength = configuration.ffnLength;
                this.numberOfLayers = configuration.numberOfLayers;
                this.numberOfHeads = configuration.numberOfHeads;

                this.keyHeadSize = configuration.keyHeadSize;
                this.valueHeadSize = configuration.valueHeadSize;

                this.numberOfKeyValueHeads = configuration.numberOfKeyValueHeads;
                this.vocabularySize = configuration.vocabularySize;
                this.contextLength = configuration.contextLength;
                this.rmsNormEps = configuration.rmsNormEps;
                this.ropeTheta = configuration.ropeTheta;
                this.ropeIsNeoxStyle = configuration.ropeIsNeoxStyle;
                this.attentionScale = configuration.attentionScale;

                this.residualScale = configuration.residualScale;
                this.logitScale = configuration.logitScale;
                this.embeddingScale = configuration.embeddingScale;
                this.activationFunction = configuration.activationFunction;
                this.activationFunctionPerLayer = configuration.activationFunctionPerLayer;
            }

            public Configuration build() {
                return new Configuration(
                        embeddingLength,
                        ffnLength,
                        numberOfLayers,
                        numberOfHeads,
                        keyHeadSize,
                        valueHeadSize,
                        numberOfKeyValueHeads,
                        vocabularySize,
                        contextLength,
                        rmsNormEps,
                        ropeTheta,
                        ropeIsNeoxStyle,
                        attentionScale,
                        residualScale,
                        logitScale,
                        embeddingScale,
                        activationFunction,
                        activationFunctionPerLayer);
            }
        }
    }

    public static class Weights {
        // token embedding table
        public final FloatMatrixView tokenEmbeddings; // (vocab_size, dim)
        // weights for rmsnorms
        public final FloatSpan[] rmsAttentionWeights; // (layer, dim) rmsnorm weights
        // weights for matmuls
        public final FloatMatrixView[] queryWeights; // (layer, n_heads, head_size, dim)
        public final FloatMatrixView[] keyWeights; // (layer, n_kv_heads, head_size, dim)

        public final FloatSpan[] queryNormWeights; // (layer, head_size)
        public final FloatSpan[] keyNormWeights; // (layer, head_size)

        public final FloatMatrixView[] valueWeights; // (layer, n_kv_heads, head_size, dim)

        public final FloatSpan[] queryBias; // (layer, dim)
        public final FloatSpan[] keyBias; // (layer, kv_dim)
        public final FloatSpan[] valueBias; // (layer, kv_dim)

        public final FloatMatrixView[] outputWeights; // (layer, n_heads * head_size, dim)
        public final FloatSpan[] rmsFFNWeights; // (layer, dim)
        // weights for ffn
        public final FloatMatrixView[] ffnGate; // (layer, ffn_length, dim)
        public final FloatMatrixView[] ffnDown; // (layer, dim, ffn_length)
        public final FloatMatrixView[] ffnUp; // (layer, ffn_length, dim)
        // public final rmsnorm
        public final FloatSpan rmsFinalWeights; // (dim,)
        // freq_cis for RoPE relatively positional embeddings
        public final FloatSpan ropeReal; // (seq_len, head_size/2)
        public final FloatSpan ropeImag; // (seq_len, head_size/2)
        // (optional) classifier weights for the logits, on the last layer
        public final FloatMatrixView classifierWeights; // (vocab_size, dim)

        public Weights(
                FloatMatrixView tokenEmbeddings,
                FloatSpan[] rmsAttentionWeights,
                FloatMatrixView[] queryWeights,
                FloatMatrixView[] keyWeights,
                FloatMatrixView[] valueWeights,
                FloatSpan[] queryNormWeights,
                FloatSpan[] keyNormWeights,
                FloatSpan[] queryBias,
                FloatSpan[] keyBias,
                FloatSpan[] valueBias,
                FloatMatrixView[] outputWeights,
                FloatSpan[] rmsFFNWeights,
                FloatMatrixView[] ffnGate,
                FloatMatrixView[] ffnDown,
                FloatMatrixView[] ffnUp,
                FloatSpan rmsFinalWeights,
                FloatSpan ropeReal,
                FloatSpan ropeImag,
                FloatMatrixView classifierWeights) {
            this.tokenEmbeddings = tokenEmbeddings;
            this.rmsAttentionWeights = rmsAttentionWeights;
            this.queryWeights = queryWeights;
            this.keyWeights = keyWeights;
            this.valueWeights = valueWeights;

            this.queryNormWeights = queryNormWeights;
            this.keyNormWeights = keyNormWeights;

            this.queryBias = queryBias;
            this.keyBias = keyBias;
            this.valueBias = valueBias;

            this.outputWeights = outputWeights;
            this.rmsFFNWeights = rmsFFNWeights;
            this.ffnGate = ffnGate;
            this.ffnDown = ffnDown;
            this.ffnUp = ffnUp;
            this.rmsFinalWeights = rmsFinalWeights;
            this.ropeReal = ropeReal;
            this.ropeImag = ropeImag;
            this.classifierWeights = classifierWeights;
        }
    }

    public static class State {
        public final int batchSize;
        public final FloatMatrixView x; // activation at current time stamp (batchSize, dim,)
        public final FloatMatrixView xb; // same, but inside a residual branch (batchSize, dim,)
        public final FloatMatrixView
                attention_out; // same, but inside a residual branch (batchSize, dim,)
        public final FloatMatrixView
                xb2; // an additional buffer just for convenience (batchSize, dim,)
        public final FloatMatrixView
                hb; // buffer for hidden dimension in the ffn (batchSize, ffn_length,)
        public final FloatMatrixView
                hb2; // buffer for hidden dimension in the ffn (batchSize, ffn_length,)
        public final FloatMatrixView query; // query (batchSize, dim,)
        public final FloatMatrixView key; // key (batchSize, kvDim,)
        public final FloatMatrixView value; // value (batchSize, kvDim,)
        public final FloatMatrixView
                attentionScores; // buffer for scores/attention values (batchSize, n_heads, seq_len)

        // These do not need batches.
        public final FloatSpan logits; // output logits (vocab_size,)
        // kv cache
        public final FloatSpan[] keyCache; // (n_layer, seq_len, kv_dim)
        public final FloatSpan[] valueCache; // (n_layer, kv_dim, seq_len) stored transposed

        //        public int latestToken; // mutable state
        public int latestIngestedTokenBatchIndex;
        public final IntSequence.Builder ingestedTokens;

        protected State(
                int batchSize,
                FloatSpan x,
                FloatSpan xb,
                FloatSpan attention_out,
                FloatSpan xb2,
                FloatSpan hb,
                FloatSpan hb2,
                FloatSpan query,
                FloatSpan key,
                FloatSpan value,
                FloatSpan attentionScores,
                FloatSpan logits,
                FloatSpan[] keyCache,
                FloatSpan[] valueCache) {
            this.batchSize = batchSize;
            this.x = FloatMatrixView.inBatchesCached(x, batchSize);
            this.xb = FloatMatrixView.inBatchesCached(xb, batchSize);
            this.attention_out = FloatMatrixView.inBatchesCached(attention_out, batchSize);
            this.xb2 = FloatMatrixView.inBatchesCached(xb2, batchSize);
            this.hb = FloatMatrixView.inBatchesCached(hb, batchSize);
            this.hb2 = FloatMatrixView.inBatchesCached(hb2, batchSize);
            this.query = FloatMatrixView.inBatchesCached(query, batchSize);
            this.key = FloatMatrixView.inBatchesCached(key, batchSize);
            this.value = FloatMatrixView.inBatchesCached(value, batchSize);
            this.attentionScores = FloatMatrixView.inBatchesCached(attentionScores, batchSize);
            this.logits = logits;
            this.keyCache = keyCache;
            this.valueCache = valueCache;
            this.ingestedTokens = IntSequence.newBuilder();
        }
    }

    @Override
    public State createNewState(int batchSize) {
        if (Integer.bitCount(batchSize) != 1) {
            throw new IllegalArgumentException("batchSize must be a power of 2");
        }

        Configuration config = configuration();
        int kvDim = (config.keyHeadSize * config.numberOfKeyValueHeads); // / config.numberOfHeads;

        FloatSpan x = spanFactory.allocateBatches(batchSize, config.embeddingLength);
        FloatSpan xb = spanFactory.allocateBatches(batchSize, config.embeddingLength);

        FloatSpan attention_out =
                spanFactory.allocateBatches(batchSize, config.keyHeadSize * config.numberOfHeads);

        FloatSpan xb2 = spanFactory.allocateBatches(batchSize, config.embeddingLength);

        FloatSpan hb = spanFactory.allocateBatches(batchSize, config.ffnLength);
        FloatSpan hb2 = spanFactory.allocateBatches(batchSize, config.ffnLength);
        FloatSpan query =
                spanFactory.allocateBatches(batchSize, config.keyHeadSize * config.numberOfHeads);
        FloatSpan key = spanFactory.allocateBatches(batchSize, kvDim);
        FloatSpan value = spanFactory.allocateBatches(batchSize, kvDim);
        FloatSpan attentionScores =
                spanFactory.allocateBatches(batchSize, config.numberOfHeads, config.contextLength);

        // These not need to be batched.
        FloatSpan logits = spanFactory.allocate(config.vocabularySize);
        FloatSpan[] keyCache =
                Stream.generate(() -> spanFactory.allocate(config.contextLength, kvDim))
                        .limit(config.numberOfLayers)
                        .toArray(FloatSpan[]::new);
        FloatSpan[] valueCache =
                Stream.generate(() -> spanFactory.allocate(kvDim, config.contextLength))
                        .limit(config.numberOfLayers)
                        .toArray(FloatSpan[]::new);

        return new State(
                batchSize,
                x,
                xb,
                attention_out,
                xb2,
                hb,
                hb2,
                query,
                key,
                value,
                attentionScores,
                logits,
                keyCache,
                valueCache);
    }

    private void batchedForwardImpl(
            Weights weights, State state, int[] tokens, int position, boolean computeLogits) {

        if (computeLogits && tokens.length != 1) {
            throw new IllegalArgumentException("cannot compute logits of multiple previous tokens");
        }

        // a few convenience variables
        Configuration config = configuration();
        int headSize = config.valueHeadSize;
        int kvDim = config.numberOfKeyValueHeads * config.keyHeadSize; // (config.embeddingLength *
        // config.numberOfKeyValueHeads) /
        // config.numberOfHeads;

        /*
         * numberOfKeyValueHeads == numberOfHeads => Multi-head attention (MHA)
         * numberOfKeyValueHeads == 1             => Multi-query attention (MQA)
         * else                                   => Grouped-query attention (GQA)
         */
        assert config.numberOfHeads % config.numberOfKeyValueHeads == 0;
        assert !Float.isNaN(config.attentionScale);

        int batchSize = tokens.length;
        assert batchSize <= state.batchSize;
        // assert Integer.bitCount(batchSize) == 1;

        if (!computeLogits) {
            // copy the token embedding into x
            Parallel.parallelFor(
                    0,
                    batchSize,
                    t -> kernelOps.copyTo(weights.tokenEmbeddings.row(tokens[t]), state.x.row(t)));
            if (TraceDebug.enabled() && batchSize > 0) {
                TraceDebug.vector("x_embed", -1, position, state.x.row(0));
            }
            // For Granite models.
            if (!Float.isNaN(config.embeddingScale)) {
                Parallel.parallelFor(
                        0,
                        batchSize,
                        t ->
                                kernelOps.scale(
                                        state.x.row(t), config.embeddingScale, state.x.row(t)));
            }
        }

        // forward all the layers
        for (int currentLayer = computeLogits ? config.numberOfLayers - 1 : 0;
                currentLayer < config.numberOfLayers;
                currentLayer++) {
            final int li = currentLayer;
            if (!computeLogits) {
                // attention rmsnorm
                Parallel.parallelFor(
                        0,
                        batchSize,
                        t ->
                                kernelOps.rmsNorm(
                                        state.x.row(t),
                                        weights.rmsAttentionWeights[li],
                                        config.rmsNormEps,
                                        state.xb.row(t)));
                if (TraceDebug.enabled() && batchSize > 0) {
                    TraceDebug.vector("xb", li, position, state.xb.row(0));
                }

                // qkv matmuls for this position
                kernelOps.matrixMultiply(
                        batchSize,
                        config.keyHeadSize * config.numberOfHeads,
                        config.embeddingLength,
                        state.xb,
                        weights.queryWeights[li],
                        state.query);
                kernelOps.matrixMultiply(
                        batchSize,
                        kvDim,
                        config.embeddingLength,
                        state.xb,
                        weights.keyWeights[li],
                        state.key);
                kernelOps.matrixMultiply(
                        batchSize,
                        kvDim,
                        config.embeddingLength,
                        state.xb,
                        weights.valueWeights[li],
                        state.value);
                if (TraceDebug.enabled() && batchSize > 0) {
                    TraceDebug.vector("q_pre", li, position, state.query.row(0));
                    TraceDebug.vector("k_pre", li, position, state.key.row(0));
                    TraceDebug.vector("v_pre", li, position, state.value.row(0));
                }

                // Bias correction.
                if (weights.queryBias != null && weights.queryBias[li] != null) {
                    Parallel.parallelFor(
                            0,
                            batchSize,
                            t ->
                                    kernelOps.add(
                                            state.query.row(t),
                                            weights.queryBias[li],
                                            state.query.row(t)));
                }
                if (weights.keyBias != null && weights.keyBias[li] != null) {
                    Parallel.parallelFor(
                            0,
                            batchSize,
                            t ->
                                    kernelOps.add(
                                            state.key.row(t),
                                            weights.keyBias[li],
                                            state.key.row(t)));
                }
                if (weights.valueBias != null && weights.valueBias[li] != null) {
                    Parallel.parallelFor(
                            0,
                            batchSize,
                            t ->
                                    kernelOps.add(
                                            state.value.row(t),
                                            weights.valueBias[li],
                                            state.value.row(t)));
                }

                if (weights.queryNormWeights != null) {
                    // attention rmsnorm
                    Parallel.parallelFor(
                            0,
                            batchSize,
                            t -> {
                                for (int h = 0; h < config.numberOfHeads; ++h) {
                                    FloatSpan queryHead =
                                            state.query.row(t).slice(headSize * h, headSize);
                                    kernelOps.rmsNorm(
                                            queryHead,
                                            weights.queryNormWeights[li],
                                            config.rmsNormEps,
                                            queryHead);
                                }
                            });
                }

                if (weights.keyNormWeights != null) {
                    // attention rmsnorm
                    Parallel.parallelFor(
                            0,
                            batchSize,
                            t -> {
                                for (int h = 0; h < config.numberOfKeyValueHeads; ++h) {
                                    FloatSpan keyHead =
                                            state.key
                                                    .row(t)
                                                    .slice(
                                                            config.keyHeadSize * h,
                                                            config.keyHeadSize);
                                    kernelOps.rmsNorm(
                                            keyHead,
                                            weights.keyNormWeights[li],
                                            config.rmsNormEps,
                                            keyHead);
                                }
                            });
                }

                // RoPE relative positional encoding: complex-valued rotate q and k in each head
                Parallel.parallelFor(
                        0,
                        batchSize,
                        t ->
                                kernelOps.rotate(
                                        config.ropeIsNeoxStyle,
                                        state.query.row(t),
                                        weights.ropeReal,
                                        weights.ropeImag,
                                        position + t,
                                        config.numberOfHeads,
                                        headSize,
                                        state.query.row(t)));
                Parallel.parallelFor(
                        0,
                        batchSize,
                        t ->
                                kernelOps.rotate(
                                        config.ropeIsNeoxStyle,
                                        state.key.row(t),
                                        weights.ropeReal,
                                        weights.ropeImag,
                                        position + t,
                                        config.numberOfKeyValueHeads,
                                        headSize,
                                        state.key.row(t)));
                if (TraceDebug.enabled() && batchSize > 0) {
                    TraceDebug.vector("q", li, position, state.query.row(0));
                    TraceDebug.vector("k", li, position, state.key.row(0));
                }

                // save key,value at this time step (position) to our kv cache
                // int loff = li * config.seq_len * kvDim; // kv cache layer offset for convenience
                Parallel.parallelFor(
                        0,
                        batchSize,
                        t ->
                                updateKeyValueCache(
                                        position + t,
                                        config.contextLength,
                                        kvDim,
                                        state.key.row(t),
                                        state.keyCache[li],
                                        state.value.row(t),
                                        state.valueCache[li]));
            }

            if (!computeLogits && li == config.numberOfLayers - 1) {
                // Logits are not needed, can skip attention and FFN of the last layer.
                return;
            }

            int batchIndex = -1;
            if (computeLogits && li == config.numberOfLayers - 1) {
                batchIndex = state.latestIngestedTokenBatchIndex;
            } else if (batchSize == 1) {
                batchIndex = 0;
            }

            // multihead attention
            Parallel.parallelFor(
                    0,
                    batchSize,
                    batchIndex,
                    t ->
                            attention(
                                    li,
                                    position + t,
                                    config.numberOfHeads,
                                    headSize,
                                    config.contextLength,
                                    kvDim,
                                    config.numberOfKeyValueHeads,
                                    config.attentionScale,
                                    state.query.row(t),
                                    state.keyCache[li],
                                    state.valueCache[li],
                                    state.attentionScores.row(t),
                                    state.attention_out.row(t)));
            if (TraceDebug.enabled() && batchSize > 0) {
                int ti = batchIndex >= 0 ? batchIndex : 0;
                TraceDebug.vector("attn_out", li, position + ti, state.attention_out.row(ti));
            }

            // final matmul to get the output of the attention
            if (batchIndex >= 0) {
                kernelOps.matrixVectorMultiply(
                        weights.outputWeights[li], /*dim, dim,*/
                        state.attention_out.row(batchIndex),
                        state.xb2.row(batchIndex));
            } else {
                kernelOps.matrixMultiply(
                        batchSize,
                        config.embeddingLength,
                        config.embeddingLength,
                        state.attention_out,
                        weights.outputWeights[li],
                        state.xb2);
            }

            // For Granite models.
            if (!Float.isNaN(config.residualScale)) {
                Parallel.parallelFor(
                        0,
                        batchSize,
                        batchIndex,
                        t ->
                                kernelOps.scale(
                                        state.xb2.row(t), config.residualScale, state.xb2.row(t)));
            }

            // residual connection back into x
            Parallel.parallelFor(
                    0,
                    batchSize,
                    batchIndex,
                    t -> kernelOps.add(state.x.row(t), state.xb2.row(t), state.x.row(t)));

            // ffn rmsnorm
            Parallel.parallelFor(
                    0,
                    batchSize,
                    batchIndex,
                    t ->
                            kernelOps.rmsNorm(
                                    state.x.row(t),
                                    weights.rmsFFNWeights[li],
                                    config.rmsNormEps,
                                    state.xb.row(t)));

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            if (weights.ffnGate != null) {
                if (batchIndex >= 0) {
                    kernelOps.matrixVectorMultiply(
                            weights.ffnGate[li], /*config.ffnLength, dim,*/
                            state.xb.row(batchIndex),
                            state.hb.row(batchIndex));
                } else {
                    // kernelOps.matrixMultiply(weights.ffnGate[li], /*config.ffnLength, dim,*/
                    // batchSize, state.xb, state.hb);
                    kernelOps.matrixMultiply(
                            batchSize,
                            config.ffnLength,
                            config.embeddingLength,
                            state.xb,
                            weights.ffnGate[li],
                            state.hb);
                }

                if (batchIndex >= 0) {
                    kernelOps.matrixVectorMultiply(
                            weights.ffnUp[li], /*config.ffnLength, dim,*/
                            state.xb.row(batchIndex),
                            state.hb2.row(batchIndex));
                } else {
                    kernelOps.matrixMultiply(
                            batchSize,
                            config.ffnLength,
                            config.embeddingLength,
                            state.xb,
                            weights.ffnUp[li],
                            state.hb2);
                }

                // SwiGLU non-linearity
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                FloatUnaryOperator activationFunction =
                        config.activationFunction != null
                                ? config.activationFunction
                                : config.activationFunctionPerLayer[currentLayer];
                Parallel.parallelFor(
                        0,
                        batchSize,
                        batchIndex,
                        t ->
                                kernelOps.elementWise(
                                        state.hb.row(t), activationFunction, state.hb.row(t)));

                // elementwise multiply with w3(x)
                Parallel.parallelFor(
                        0,
                        batchSize,
                        batchIndex,
                        t ->
                                kernelOps.multiply(
                                        state.hb.row(t), state.hb2.row(t), state.hb.row(t)));
            } else {
                if (batchIndex >= 0) {
                    kernelOps.matrixVectorMultiply(
                            weights.ffnUp[li], /*config.ffnLength, dim,*/
                            state.xb.row(batchIndex),
                            state.hb.row(batchIndex));
                } else {
                    kernelOps.matrixMultiply(
                            batchSize,
                            config.ffnLength,
                            config.embeddingLength,
                            state.xb,
                            weights.ffnUp[li],
                            state.hb);
                }
                // SwiGLU non-linearity
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                FloatUnaryOperator activationFunction =
                        config.activationFunction != null
                                ? config.activationFunction
                                : config.activationFunctionPerLayer[currentLayer];
                Parallel.parallelFor(
                        0,
                        batchSize,
                        batchIndex,
                        t ->
                                kernelOps.elementWise(
                                        state.hb.row(t), activationFunction, state.hb.row(t)));
            }

            // final matmul to get the output of the ffn
            if (batchIndex >= 0) {
                kernelOps.matrixVectorMultiply(
                        weights.ffnDown[li], /*dim, config.ffnLength,*/
                        state.hb.row(batchIndex),
                        state.xb.row(batchIndex));
            } else {
                kernelOps.matrixMultiply(
                        batchSize,
                        config.embeddingLength,
                        config.ffnLength,
                        state.hb,
                        weights.ffnDown[li],
                        state.xb);
            }

            // For Granite models.
            if (!Float.isNaN(config.residualScale)) {
                Parallel.parallelFor(
                        0,
                        batchSize,
                        batchIndex,
                        t ->
                                kernelOps.scale(
                                        state.xb.row(t), config.residualScale, state.xb.row(t)));
            }

            // residual connection
            Parallel.parallelFor(
                    0,
                    batchSize,
                    batchIndex,
                    t -> kernelOps.add(state.x.row(t), state.xb.row(t), state.x.row(t)));
            if (TraceDebug.enabled() && batchSize > 0) {
                int ti = batchIndex >= 0 ? batchIndex : 0;
                TraceDebug.vector("x", li, position + ti, state.x.row(ti));
            }
        }

        // Only compute logits for the latest batch.
        int latestBatch = state.latestIngestedTokenBatchIndex;

        // final rmsnorm
        kernelOps.rmsNorm(
                state.x.row(latestBatch),
                weights.rmsFinalWeights,
                config.rmsNormEps,
                state.x.row(latestBatch));

        // classifier into logits
        kernelOps.matrixVectorMultiply(
                weights.classifierWeights, /*config.vocabularySize, dim,*/
                state.x.row(latestBatch),
                state.logits);
        if (TraceDebug.enabled()) {
            TraceDebug.vector("logits", config.numberOfLayers, position + latestBatch, state.logits);
        }

        // For Granite models.
        if (!Float.isNaN(config.logitScale)) {
            kernelOps.scale(state.logits, 1f / config.logitScale, state.logits);
        }
    }

    protected void attention(
            int layerIndex,
            int position,
            int numberOfHeads,
            int headSize,
            int contextLength,
            int kvDim,
            int keyValueHeads,
            float attentionScale,
            FloatSpan query,
            FloatSpan keyCache,
            FloatSpan valueCache,
            FloatSpan attentionScores,
            FloatSpan attentionOutput) {
        assert numberOfHeads % keyValueHeads == 0;
        int kvMul = numberOfHeads / keyValueHeads;
        // Process each group of kvMul heads that share the same KV cache
        Parallel.parallelFor(
                0,
                keyValueHeads,
                kvHead -> {
                    int h = kvHead * kvMul; // first query head of the group of kvMul heads.
                    FloatMatrixView keyMatrix =
                            FloatMatrixView.asMatrix(
                                    keyCache, kvHead * headSize, position + 1, headSize, kvDim);
                    FloatMatrixView queryMatrix =
                            FloatMatrixView.asMatrix(query, h * headSize, kvMul, headSize);
                    FloatMatrixView scoresMatrix =
                            FloatMatrixView.asMatrix(
                                    attentionScores,
                                    h * contextLength,
                                    kvMul,
                                    position + 1,
                                    contextLength);

                    // 1. Compute Q @ K^T for all heads in the group at once
                    kernelOps.matrixMultiply(
                            kvMul, position + 1, headSize, queryMatrix, keyMatrix, scoresMatrix);

                    // 2. Scale attention scores
                    for (int i = 0; i < kvMul; i++) {
                        // Get the scores for one head
                        FloatSpan headScores = scoresMatrix.row(i);
                        kernelOps.scale(headScores, attentionScale, headScores);
                        // Apply softmax to get attention weights
                        kernelOps.softMax(headScores, headScores);
                    }

                    // [keyValueHead, headSize, contextLength]
                    FloatMatrixView valueMatrix =
                            FloatMatrixView.asMatrix(
                                    valueCache,
                                    (kvHead * headSize) * contextLength,
                                    headSize,
                                    position + 1,
                                    contextLength);
                    FloatMatrixView outputMatrix =
                            FloatMatrixView.asMatrix(
                                    attentionOutput, h * headSize, kvMul, headSize);

                    // 3. Compute attention @ V for all heads in the group at once
                    // [R, K]   -> kvMul, position+1
                    // [K, C]^T -> headSize, position+1
                    // [R, C]   -> kvMul, headSize
                    kernelOps.matrixMultiply(
                            kvMul, headSize, position + 1, scoresMatrix, valueMatrix, outputMatrix);
                });
    }

    protected void updateKeyValueCache(
            int position,
            int contextLength,
            int kvDim,
            FloatSpan key,
            FloatSpan keyCache,
            FloatSpan value,
            FloatSpan valueCache) {
        kernelOps.copyTo(key, keyCache.slice(position * (long) kvDim, kvDim));
        // Copy to valueCache column, since valueCache is transposed.
        kernelOps.copyToStrided(value, valueCache, position, contextLength);
    }

    @Override
    public void ingestTokens(Weights weights, State state, int[] tokens) {
        assert tokens.length > 0;
        batchedForwardImpl(weights, state, tokens, state.ingestedTokens.length(), false);
        state.ingestedTokens.addAll(IntSequence.of(tokens));
        state.latestIngestedTokenBatchIndex = tokens.length - 1;
    }

    @Override
    public void computeLogits(Weights weights, State state) {
        batchedForwardImpl(
                weights,
                state,
                new int[] {state.ingestedTokens.getLast()},
                state.ingestedTokens.length() - 1,
                true);
    }
}
