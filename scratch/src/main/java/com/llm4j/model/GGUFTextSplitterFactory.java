package com.llm4j.model;

import com.llm4j.gguf.GGUF;

public interface GGUFTextSplitterFactory extends TextSplitterFactory<GGUF> {
    @Override
    default String getSourceName() {
        return "gguf";
    }
}
