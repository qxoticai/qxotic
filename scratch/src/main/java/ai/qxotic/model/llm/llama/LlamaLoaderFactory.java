package ai.qxotic.model.llm.llama;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFModelLoaderFactory;
import ai.qxotic.model.llm.ModelLoader;
import ai.qxotic.model.llm.ModelLoaderFactory;
import com.google.auto.service.AutoService;

@AutoService(ModelLoaderFactory.class)
public class LlamaLoaderFactory implements GGUFModelLoaderFactory {
    @Override
    public String getArchitectureName() {
        return "llama";
    }

    @Override
    public ModelLoader createLoader(GGUF source) {
        return new BaseLlamaLoader(source);
    }
}
