package com.qxotic.model.llm.smollm2;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFTextSplitterFactory;
import com.qxotic.model.llm.TextSplitterFactory;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.RegexSplitter;
import java.util.Arrays;

// Implementation for SmolLM pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class SmolLMTextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String[] SMOLLM2_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };

    @Override
    public String getTextSplitterName() {
        return "smollm";
    }

    @Override
    public Splitter createTextSplitter(GGUF gguf) {
        Splitter[] splitters =
                Arrays.stream(SMOLLM2_PATTERNS).map(RegexSplitter::create).toArray(Splitter[]::new);
        return Splitter.sequence(splitters);
    }
}
