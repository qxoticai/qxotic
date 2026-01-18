package ai.qxotic.model.llm.phi3;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFModelLoaderFactory;
import ai.qxotic.model.llm.ModelLoader;
import ai.qxotic.model.llm.ModelLoaderFactory;
import com.google.auto.service.AutoService;

@AutoService(ModelLoaderFactory.class)
public class Phi3LoaderFactory implements GGUFModelLoaderFactory {
    @Override
    public String getArchitectureName() {
        return "phi3";
    }

    @Override
    public ModelLoader createLoader(GGUF gguf) {
        return new Phi3Loader(gguf);
    }
}
