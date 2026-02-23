package com.qxotic.model.llm;

import com.qxotic.format.gguf.GGUF;

public interface GGUFTextSplitterFactory extends TextSplitterFactory<GGUF> {
    @Override
    default String getSourceName() {
        return "gguf";
    }
}
