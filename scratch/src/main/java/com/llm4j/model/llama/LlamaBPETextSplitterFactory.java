package com.llm4j.model.llama;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFTextSplitterFactory;
import com.llm4j.model.TextSplitterFactory;
import com.llm4j.tokenizers.TextSplitter;
import com.llm4j.tokenizers.impl.RegexSplitter;

// Implementation for Llama BPE pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class LlamaBPETextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String LLAMA3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

    @Override
    public String getTextSplitterName() {
        return "llama-bpe";
    }

    @Override
    public TextSplitter createTextSplitter(GGUF gguf) {
        return RegexSplitter.create(LLAMA3_PATTERN);
    }
}
