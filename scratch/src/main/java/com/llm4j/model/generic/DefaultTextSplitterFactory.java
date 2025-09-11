package com.llm4j.model.generic;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFTextSplitterFactory;
import com.llm4j.model.TextSplitterFactory;
import com.llm4j.tokenizers.TextSplitter;
import com.llm4j.tokenizers.impl.RegexSplitter;

import java.util.Arrays;

// Implementation for default pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class DefaultTextSplitterFactory implements GGUFTextSplitterFactory {
    @Override
    public String getTextSplitterName() {
        return "default";
    }

    @Override
    public TextSplitter createTextSplitter(GGUF gguf) {
        TextSplitter[] preTokenizers = Arrays.stream(RegexSplitter.DEFAULT_BPE_SPLITS)
                .map(RegexSplitter::create)
                .toArray(TextSplitter[]::new);
        return TextSplitter.compose(preTokenizers);
    }
}
