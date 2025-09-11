package com.llm4j.model.apertus;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFModelLoaderFactory;
import com.llm4j.model.ModelLoader;
import com.llm4j.model.ModelLoaderFactory;

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
