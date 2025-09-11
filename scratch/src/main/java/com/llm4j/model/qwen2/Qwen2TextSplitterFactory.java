package com.llm4j.model.qwen2;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFTextSplitterFactory;
import com.llm4j.model.TextSplitterFactory;
import com.llm4j.tokenizers.TextSplitter;
import com.llm4j.tokenizers.impl.RegexSplitter;

@AutoService(TextSplitterFactory.class)
public class Qwen2TextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String QWEN2_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

    @Override
    public String getTextSplitterName() {
        return "qwen2";
    }

    @Override
    public TextSplitter createTextSplitter(GGUF gguf) {
        return RegexSplitter.create(QWEN2_PATTERN);
    }
}
