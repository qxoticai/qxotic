package com.qxotic.model.llm;

import com.qxotic.format.gguf.GGUF;

public interface GGUFTokenizerFactory extends TokenizerFactory<GGUF> {
    @Override
    default String getSourceName() {
        return "gguf";
    }
}
