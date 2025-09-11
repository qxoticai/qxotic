package com.llm4j.model.granite;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFTextSplitterFactory;
import com.llm4j.model.TextSplitterFactory;
import com.llm4j.tokenizers.TextSplitter;
import com.llm4j.tokenizers.impl.RegexSplitter;

import java.util.Arrays;

// Implementation for Refact pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class RefactTextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String[] GRANITE_PATTERNS = {
            "\\p{N}",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    };

    @Override
    public String getTextSplitterName() {
        return "refact";
    }

    @Override
    public TextSplitter createTextSplitter(GGUF gguf) {
        TextSplitter[] preTokenizers = Arrays.stream(GRANITE_PATTERNS)
                .map(RegexSplitter::create)
                .toArray(TextSplitter[]::new);
        return TextSplitter.compose(preTokenizers);
    }
}
