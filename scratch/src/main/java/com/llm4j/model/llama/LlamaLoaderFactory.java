package com.llm4j.model.llama;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFModelLoaderFactory;
import com.llm4j.model.ModelLoader;
import com.llm4j.model.ModelLoaderFactory;

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
