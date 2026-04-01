package com.qxotic.model.llm.granite;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFTextSplitterFactory;
import com.qxotic.model.llm.TextSplitterFactory;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.impl.RegexSplitter;
import java.util.Arrays;

// Implementation for Refact pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class RefactTextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String[] GRANITE_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    };

    @Override
    public String getTextSplitterName() {
        return "refact";
    }

    @Override
    public Splitter createTextSplitter(GGUF gguf) {
        Splitter[] splitters =
                Arrays.stream(GRANITE_PATTERNS).map(RegexSplitter::create).toArray(Splitter[]::new);
        return Splitter.sequence(splitters);
    }
}
