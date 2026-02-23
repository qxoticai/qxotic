package com.qxotic.model.llm.llama;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFModelLoaderFactory;
import com.qxotic.model.llm.ModelLoader;
import com.qxotic.model.llm.ModelLoaderFactory;

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
