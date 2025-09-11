package com.llm4j.model.generic;

import com.google.auto.service.AutoService;
import com.llm4j.model.TextSplitterFactory;
import com.llm4j.model.llama.LlamaBPETextSplitterFactory;

@AutoService(TextSplitterFactory.class)
public class DBRXTextSplitterFactory extends LlamaBPETextSplitterFactory {
    @Override
    public String getTextSplitterName() {
        return "dbrx";
    }
}
