package com.qxotic.model.llm;

import com.qxotic.format.gguf.GGUF;

public interface GGUFModelLoaderFactory extends ModelLoaderFactory<GGUF> {
    @Override
    default String getFormatName() {
        return "gguf";
    }
}
