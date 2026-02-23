package com.qxotic.model.llm.qwen2;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFTextSplitterFactory;
import com.qxotic.model.llm.TextSplitterFactory;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.RegexSplitter;

@AutoService(TextSplitterFactory.class)
public class Qwen2TextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String QWEN2_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

    @Override
    public String getTextSplitterName() {
        return "qwen2";
    }

    @Override
    public Splitter createTextSplitter(GGUF gguf) {
        return RegexSplitter.create(QWEN2_PATTERN);
    }
}
