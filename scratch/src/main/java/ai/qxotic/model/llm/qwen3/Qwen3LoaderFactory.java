package ai.qxotic.model.llm.qwen3;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFModelLoaderFactory;
import ai.qxotic.model.llm.ModelLoader;
import ai.qxotic.model.llm.ModelLoaderFactory;
import com.google.auto.service.AutoService;

@AutoService(ModelLoaderFactory.class)
public class Qwen3LoaderFactory implements GGUFModelLoaderFactory {
    @Override
    public String getArchitectureName() {
        return "qwen3";
    }

    @Override
    public ModelLoader createLoader(GGUF gguf) {
        return new Qwen3Loader(gguf);
    }
}
