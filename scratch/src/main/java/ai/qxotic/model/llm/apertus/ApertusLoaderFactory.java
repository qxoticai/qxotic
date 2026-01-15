package ai.qxotic.model.llm.apertus;

import com.google.auto.service.AutoService;
import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFModelLoaderFactory;
import ai.qxotic.model.llm.ModelLoader;
import ai.qxotic.model.llm.ModelLoaderFactory;

@AutoService(ModelLoaderFactory.class)
public class ApertusLoaderFactory implements GGUFModelLoaderFactory {
    @Override
    public String getArchitectureName() {
        return "apertus";
    }

    @Override
    public ModelLoader createLoader(GGUF gguf) {
        return new ApertusLoader(gguf);
    }
}
