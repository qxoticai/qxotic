package com.llm4j.model.smollm2;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFTextSplitterFactory;
import com.llm4j.model.TextSplitterFactory;
import com.llm4j.tokenizers.TextSplitter;
import com.llm4j.tokenizers.impl.RegexSplitter;

import java.util.Arrays;

// Implementation for SmolLM pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class SmolLMTextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String[] SMOLLM2_PATTERNS = {
            "\\p{N}",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };

    @Override
    public String getTextSplitterName() {
        return "smollm";
    }

    @Override
    public TextSplitter createTextSplitter(GGUF gguf) {
        TextSplitter[] preTokenizers = Arrays.stream(SMOLLM2_PATTERNS)
                .map(RegexSplitter::create)
                .toArray(TextSplitter[]::new);
        return TextSplitter.compose(preTokenizers);
    }
}
