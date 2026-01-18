package ai.qxotic.model.llm;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.apertus.ApertusLoaderFactory;
import ai.qxotic.model.llm.gemma3.Gemma3LoaderFactory;
import ai.qxotic.model.llm.granite.GraniteLoaderFactory;
import ai.qxotic.model.llm.llama.LlamaLoaderFactory;
import ai.qxotic.model.llm.phi3.Phi3LoaderFactory;
import ai.qxotic.model.llm.qwen2.Qwen2LoaderFactory;
import ai.qxotic.model.llm.qwen3.Qwen3LoaderFactory;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;

// ModeLoader class that uses ServiceLoader
public class Loader {
    private final Map<String, ModelLoaderFactory> factories;

    public Loader() {
        this.factories = knownTextSplitterFactories(); // fromServiceLoader();
    }

    private static Map<String, ModelLoaderFactory> knownTextSplitterFactories() {
        Map<String, ModelLoaderFactory> factories = new HashMap<>();
        List<ModelLoaderFactory> knownFactories =
                List.of(
                        new Qwen2LoaderFactory(),
                        new Qwen3LoaderFactory(),
                        new ApertusLoaderFactory(),
                        new Gemma3LoaderFactory(),
                        new Phi3LoaderFactory(),
                        new LlamaLoaderFactory(),
                        new GraniteLoaderFactory());
        for (var f : knownFactories) {
            factories.put(f.getArchitectureName(), f);
        }
        return factories;
    }

    private static Map<String, ModelLoaderFactory> fromServiceLoader() {
        Map<String, ModelLoaderFactory> factories = new HashMap<>();
        ServiceLoader<ModelLoaderFactory> loader = ServiceLoader.load(ModelLoaderFactory.class);
        for (ModelLoaderFactory factory : loader) {
            factories.put(factory.getArchitectureName(), factory);
        }
        return factories;
    }

    public ModelLoader loadModelLoader(GGUF gguf) {
        String arch = gguf.getValue(String.class, "general.architecture");
        ModelLoaderFactory factory = factories.get(arch);
        if (factory == null) {
            throw new UnsupportedOperationException("Unsupported general.architecture: " + arch);
        }
        return factory.createLoader(gguf);
    }
}
