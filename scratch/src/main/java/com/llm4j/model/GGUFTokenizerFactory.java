package com.llm4j.model;

import com.llm4j.gguf.GGUF;

public interface GGUFTokenizerFactory extends TokenizerFactory<GGUF> {
    @Override
    default String getSourceName() {
        return "gguf";
    }
}
