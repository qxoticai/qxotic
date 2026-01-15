package ai.qxotic.model.llm.qwen2;

import com.google.auto.service.AutoService;
import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFModelLoaderFactory;
import ai.qxotic.model.llm.ModelLoader;
import ai.qxotic.model.llm.ModelLoaderFactory;

@AutoService(ModelLoaderFactory.class)
public class Qwen2LoaderFactory implements GGUFModelLoaderFactory {
    @Override
    public String getArchitectureName() {
        return "qwen2";
    }

    @Override
    public ModelLoader createLoader(GGUF gguf) {
        return new Qwen2Loader(gguf);
    }
}
