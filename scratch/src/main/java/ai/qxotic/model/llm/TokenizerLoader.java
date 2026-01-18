package ai.qxotic.model.llm;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.mistral.MistralTokenizerFactory;
import ai.qxotic.tokenizers.Normalizer;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.Tokenizer;
import java.util.List;
import java.util.Optional;
import java.util.ServiceLoader;

// TokenizerLoader class that uses ServiceLoader
public class TokenizerLoader {
    private final List<TokenizerFactory> factories;

    public TokenizerLoader() {
        this.factories = knownTokenizerFactories(); // fromServiceLoader()
    }

    private static List<TokenizerFactory> knownTokenizerFactories() {
        return List.of(new GPT2TokenizerFactory(), new MistralTokenizerFactory());
    }

    private static List<TokenizerFactory> fromServiceLoader() {
        ServiceLoader<TokenizerFactory> loader = ServiceLoader.load(TokenizerFactory.class);
        return loader.stream().map(ServiceLoader.Provider::get).toList();
    }

    public Tokenizer loadTokenizer(GGUF gguf, Normalizer normalizer, TextSplitter textSplitter) {
        String model = gguf.getValue(String.class, "tokenizer.ggml.model");
        Optional<TokenizerFactory> first =
                factories.stream()
                        .filter(
                                f ->
                                        "gguf".equals(f.getSourceName())
                                                && model.equals(f.getTokenizerName()))
                        .findFirst();
        TokenizerFactory tokenizerFactory =
                first.orElseThrow(
                        () ->
                                new UnsupportedOperationException(
                                        "Unsupported tokenizer.ggml.model: " + model));
        return tokenizerFactory.createTokenizer(gguf, normalizer, textSplitter);
    }
}
