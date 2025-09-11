package com.llm4j.model.phi3;

import com.llm4j.gguf.GGUF;
import com.llm4j.model.SpanLoader;
import com.llm4j.model.generic.ChatMLFormat;
import com.llm4j.model.llama.ArraySpan;
import com.llm4j.model.llama.BaseLlamaLoader;
import com.llm4j.model.llama.RoPE;
import com.llm4j.model.llama.Timer;
import com.llm4j.span.FloatMatrixView;
import com.llm4j.span.FloatSpan;
import com.llm4j.tokenizers.Tokenizer;

import java.util.Arrays;

public class Phi3Loader extends BaseLlamaLoader {

    static final String PHI3_ARCH = "phi3";

    public Phi3Loader(GGUF gguf) {
        super(gguf);
    }

    @Override
    public ChatMLFormat createChatFormat(Tokenizer tokenizer) {
        return new ChatMLFormat(tokenizer);
    }

    @Override
    public Phi3.Configuration loadConfiguration(int maxTokens, SpanLoader spanLoader) {
        String arch = gguf.getValue(String.class, "general.architecture");
        if (!PHI3_ARCH.equals(arch)) {
            throw new IllegalArgumentException("general.architecture expected " + PHI3_ARCH + " but found " + arch);
        }
        return super.loadConfiguration(maxTokens, spanLoader).with(b -> b.ropeIsNeoxStyle(true));
    }

    @Override
    public Phi3.Weights loadWeights(Phi3.Configuration config, SpanLoader spanLoader) {
        try (var timer = Timer.log("Load weights")) {
            int dim = config.embeddingLength;
            int kvDim = config.numberOfKeyValueHeads * config.headSize;

            float[][] ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta);

            assert ropeFreqs.length == 2;
            float[] ropeFreqsReal = ropeFreqs[0];
            float[] ropeFreqsImag = ropeFreqs[1];

            FloatMatrixView tokenEmbeddings = tensorAsMatrix(gguf.getTensor("token_embd.weight"), spanLoader);

            // If "output.weight" is not present then the embedding weights are tied/shared with the decoder.
            // This is commonly referred as "tie word embeddings".
            FloatMatrixView classifierWeights = gguf.containsTensor("output.weight")
                    ? tensorAsMatrix(gguf.getTensor("output.weight"), spanLoader)
                    : tokenEmbeddings;

            // queryWeights + keyWeights + valueWeights
            FloatSpan[] qkvWeights = loadSpanArray(config.numberOfLayers, i -> gguf.getTensor("blk." + i + ".attn_qkv.weight"), spanLoader);
            // ffnGate + ffnUp
            FloatSpan[] ffnGateUp = loadSpanArray(config.numberOfLayers, i -> gguf.getTensor("blk." + i + ".ffn_up.weight"), spanLoader);

            return new Phi3.Weights(
                    tokenEmbeddings,
                    loadSpanArray(config.numberOfLayers, i -> gguf.getTensor("blk." + i + ".attn_norm.weight"), spanLoader),

                    Arrays.stream(qkvWeights).map(span -> FloatMatrixView.asMatrix(span, 0, dim, dim)).toArray(FloatMatrixView[]::new), // queryWeights,
                    Arrays.stream(qkvWeights).map(span -> FloatMatrixView.asMatrix(span, (long) dim * dim, kvDim, dim)).toArray(FloatMatrixView[]::new), // keyWeights,
                    Arrays.stream(qkvWeights).map(span -> FloatMatrixView.asMatrix(span, (long) (dim + kvDim) * dim, kvDim, dim)).toArray(FloatMatrixView[]::new), // valueWeights,

                    null,
                    null,

                    // qkv bias can be null.
                    loadSpanArray(true, config.numberOfLayers, i -> gguf.getTensor("blk." + i + ".attn_q.bias"), spanLoader),
                    loadSpanArray(true, config.numberOfLayers, i -> gguf.getTensor("blk." + i + ".attn_k.bias"), spanLoader),
                    loadSpanArray(true, config.numberOfLayers, i -> gguf.getTensor("blk." + i + ".attn_v.bias"), spanLoader),

                    loadMatrixArray(false, config.numberOfLayers, i -> gguf.getTensor("blk." + i + ".attn_output.weight"), spanLoader),
                    loadSpanArray(config.numberOfLayers, i -> gguf.getTensor("blk." + i + ".ffn_norm.weight"), spanLoader),

                    Arrays.stream(ffnGateUp).map(span -> FloatMatrixView.asMatrix(span, 0, config.ffnLength, dim)).toArray(FloatMatrixView[]::new), // ffnGate
                    loadMatrixArray(false, config.numberOfLayers, i -> gguf.getTensor("blk." + i + ".ffn_down.weight"), spanLoader), // w2
                    Arrays.stream(ffnGateUp).map(span -> FloatMatrixView.asMatrix(span, (long) config.ffnLength * dim, config.ffnLength, dim)).toArray(FloatMatrixView[]::new), // ffnUp

                    spanLoader.apply(gguf.getTensor("output_norm.weight")),
                    ArraySpan.wrap(ropeFreqsReal),
                    ArraySpan.wrap(ropeFreqsImag),
                    classifierWeights
            );
        }
    }
}
