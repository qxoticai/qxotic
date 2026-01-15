package ai.qxotic.model.llm;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.apertus.TekkenTextSplitterFactory;
import ai.qxotic.model.llm.generic.DBRXTextSplitterFactory;
import ai.qxotic.model.llm.generic.DefaultTextSplitterFactory;
import ai.qxotic.model.llm.granite.RefactTextSplitterFactory;
import ai.qxotic.model.llm.llama.LlamaBPETextSplitterFactory;
import ai.qxotic.model.llm.qwen2.Qwen2TextSplitterFactory;
import ai.qxotic.model.llm.smollm2.SmolLMTextSplitterFactory;
import ai.qxotic.tokenizers.TextSplitter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;

// TextSplitterLoader class that uses ServiceLoader
public class TextSplitterLoader {
    private final Map<String, TextSplitterFactory> factories;

    public TextSplitterLoader() {
        this.factories = knownTextSplitterFactories(); // fromServiceLoader();
    }

    private static Map<String, TextSplitterFactory> knownTextSplitterFactories() {
        Map<String, TextSplitterFactory> factories = new HashMap<>();
        List<TextSplitterFactory> knownFactories = List.of(
            new LlamaBPETextSplitterFactory(),
            new DBRXTextSplitterFactory(),
            new SmolLMTextSplitterFactory(),
            new DefaultTextSplitterFactory(),
            new Qwen2TextSplitterFactory(),
            new RefactTextSplitterFactory(),
            new TekkenTextSplitterFactory()
        );
        for (var f : knownFactories) {
            factories.put(f.getTextSplitterName(), f);
        }
        return factories;
    }

    private static Map<String, TextSplitterFactory> fromServiceLoader() {
        Map<String, TextSplitterFactory> factories = new HashMap<>();
        ServiceLoader<TextSplitterFactory> loader = ServiceLoader.load(TextSplitterFactory.class);
        for (TextSplitterFactory factory : loader) {
            factories.put(factory.getTextSplitterName(), factory);
        }
        return factories;
    }

    public TextSplitter loadTextSplitter(GGUF gguf) {
        String pre = gguf.getValue(String.class, "tokenizer.ggml.pre");
        TextSplitterFactory factory = factories.get(pre);
        if (factory == null) {
            throw new UnsupportedOperationException("Unexpected tokenizer.ggml.pre: " + pre);
        }
        return factory.createTextSplitter(gguf);
    }
}
