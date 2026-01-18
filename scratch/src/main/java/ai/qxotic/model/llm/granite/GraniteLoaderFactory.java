package ai.qxotic.model.llm.granite;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFModelLoaderFactory;
import ai.qxotic.model.llm.ModelLoader;
import ai.qxotic.model.llm.ModelLoaderFactory;
import com.google.auto.service.AutoService;

@AutoService(ModelLoaderFactory.class)
public class GraniteLoaderFactory implements GGUFModelLoaderFactory {
    @Override
    public String getArchitectureName() {
        return "granite";
    }

    @Override
    public ModelLoader createLoader(GGUF gguf) {
        return new GraniteLoader(gguf);
    }
}
