package com.qxotic.model.llm.generic;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFTextSplitterFactory;
import com.qxotic.model.llm.TextSplitterFactory;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.loaders.ModelSplitters;

// Implementation for default pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class DefaultTextSplitterFactory implements GGUFTextSplitterFactory {
    @Override
    public String getTextSplitterName() {
        return "default";
    }

    @Override
    public Splitter createTextSplitter(GGUF gguf) {
        return ModelSplitters.LLAMA3;
    }
}
