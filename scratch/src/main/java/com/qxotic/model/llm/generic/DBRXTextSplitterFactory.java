package com.qxotic.model.llm.generic;

import com.google.auto.service.AutoService;
import com.qxotic.model.llm.TextSplitterFactory;
import com.qxotic.model.llm.llama.LlamaBPETextSplitterFactory;

@AutoService(TextSplitterFactory.class)
public class DBRXTextSplitterFactory extends LlamaBPETextSplitterFactory {
    @Override
    public String getTextSplitterName() {
        return "dbrx";
    }
}
