package com.llm4j.model.phi3;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFModelLoaderFactory;
import com.llm4j.model.ModelLoader;
import com.llm4j.model.ModelLoaderFactory;

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
