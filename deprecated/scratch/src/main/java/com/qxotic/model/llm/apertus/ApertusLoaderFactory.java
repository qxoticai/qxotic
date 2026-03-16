package com.qxotic.model.llm.apertus;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFModelLoaderFactory;
import com.qxotic.model.llm.ModelLoader;
import com.qxotic.model.llm.ModelLoaderFactory;

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
