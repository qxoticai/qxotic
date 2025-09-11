package com.llm4j.model.gemma3;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFModelLoaderFactory;
import com.llm4j.model.ModelLoader;
import com.llm4j.model.ModelLoaderFactory;

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
