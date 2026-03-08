package com.qxotic.model.llm.llama;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFTextSplitterFactory;
import com.qxotic.model.llm.TextSplitterFactory;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.RegexSplitter;

// Implementation for Llama BPE pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class LlamaBPETextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String LLAMA3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\r\n"
                    + "]*|\\s*[\r\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    @Override
    public String getTextSplitterName() {
        return "llama-bpe";
    }

    @Override
    public Splitter createTextSplitter(GGUF gguf) {
        return RegexSplitter.create(LLAMA3_PATTERN);
    }
}
