package com.llm4j.model.apertus;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFTextSplitterFactory;
import com.llm4j.model.TextSplitterFactory;
import com.llm4j.tokenizers.TextSplitter;
import com.llm4j.tokenizers.impl.RegexSplitter;

import java.util.Arrays;

// Implementation for Tekken pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class TekkenTextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String[] TEKKEN_PATTERN = {
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    };

    @Override
    public String getTextSplitterName() {
        return "tekken";
    }

    @Override
    public TextSplitter createTextSplitter(GGUF gguf) {
        TextSplitter[] preTokenizers = Arrays.stream(TEKKEN_PATTERN)
                .map(RegexSplitter::create)
                .toArray(TextSplitter[]::new);
        return TextSplitter.compose(preTokenizers);
    }
}
