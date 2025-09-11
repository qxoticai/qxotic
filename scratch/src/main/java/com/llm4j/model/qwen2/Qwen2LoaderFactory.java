package com.llm4j.model.qwen2;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFModelLoaderFactory;
import com.llm4j.model.ModelLoader;
import com.llm4j.model.ModelLoaderFactory;

@AutoService(ModelLoaderFactory.class)
public class Qwen2LoaderFactory implements GGUFModelLoaderFactory {
    @Override
    public String getArchitectureName() {
        return "qwen2";
    }

    @Override
    public ModelLoader createLoader(GGUF gguf) {
        return new Qwen2Loader(gguf);
    }
}
