package ai.qxotic.model.llm.gemma3;

import com.google.auto.service.AutoService;
import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFModelLoaderFactory;
import ai.qxotic.model.llm.ModelLoader;
import ai.qxotic.model.llm.ModelLoaderFactory;

@AutoService(ModelLoaderFactory.class)
public class Gemma3LoaderFactory implements GGUFModelLoaderFactory {
    @Override
    public String getArchitectureName() {
        return "gemma3";
    }

    @Override
    public ModelLoader createLoader(GGUF gguf) {
        return new Gemma3Loader(gguf);
    }
}
