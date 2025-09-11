package com.llm4j.model.granite;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFModelLoaderFactory;
import com.llm4j.model.ModelLoader;
import com.llm4j.model.ModelLoaderFactory;

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
