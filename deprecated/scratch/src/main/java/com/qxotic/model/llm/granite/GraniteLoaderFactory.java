package com.qxotic.model.llm.granite;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFModelLoaderFactory;
import com.qxotic.model.llm.ModelLoader;
import com.qxotic.model.llm.ModelLoaderFactory;

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
