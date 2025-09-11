package com.llm4j.model;

import com.llm4j.gguf.GGUF;

public interface GGUFModelLoaderFactory extends ModelLoaderFactory<GGUF> {
    @Override
    default String getFormatName() {
        return "gguf";
    }
}

