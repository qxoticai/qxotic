package com.qxotic.model.llm.generic;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFTextSplitterFactory;
import com.qxotic.model.llm.TextSplitterFactory;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.RegexSplitter;
import java.util.Arrays;

// Implementation for default pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class DefaultTextSplitterFactory implements GGUFTextSplitterFactory {
    @Override
    public String getTextSplitterName() {
        return "default";
    }

    @Override
    public Splitter createTextSplitter(GGUF gguf) {
        Splitter[] splitters =
                Arrays.stream(RegexSplitter.DEFAULT_BPE_SPLITS)
                        .map(RegexSplitter::create)
                        .toArray(Splitter[]::new);
        return Splitter.sequence(splitters);
    }
}
