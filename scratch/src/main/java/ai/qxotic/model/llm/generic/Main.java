package ai.qxotic.model.llm.generic;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.*;
import ai.qxotic.model.llm.llama.Llama;
import ai.qxotic.model.llm.llama.Sampler2;
import ai.qxotic.tokenizers.Normalizer;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.Tokenizer;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public class Main {

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Path modelPath = options.modelPath();
        GGUF gguf = GGUF.read(modelPath);

        Model<Object, Object, ? extends Llama.State> model;
        Object weights;
        GenericGGUFLoader loader = new GenericGGUFLoader(gguf);
        try (FileChannel fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            SpanLoader spanLoader =
                    TensorLoader.loaderFromTensorDataMM(gguf.getTensorDataOffset(), fileChannel);
            var configuration = loader.loadConfiguration(options.maxTokens(), spanLoader);
            model = loader.loadModel(configuration);
            weights = loader.loadWeights(configuration, spanLoader);
        }

        TextSplitter textSplitter = new TextSplitterLoader().loadTextSplitter(gguf);
        Tokenizer tokenizer =
                new TokenizerLoader().loadTokenizer(gguf, Normalizer.IDENTITY, textSplitter);

        Sampler2 sampler =
                Sampler2.selectSampler(
                        tokenizer.vocabulary().size(),
                        options.temperature(),
                        options.topp(),
                        options.seed());

        ChatFormat chatFormat = loader.createChatFormat(tokenizer);

        RunInteractive.runInteractive(
                model, weights, chatFormat, sampler.fromLogits(s -> s.logits), options);
    }
}
