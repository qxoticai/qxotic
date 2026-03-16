package com.qxotic.model.llm.apertus;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFTextSplitterFactory;
import com.qxotic.model.llm.TextSplitterFactory;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.RegexSplitter;
import java.util.Arrays;

// Implementation for Tekken pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class TekkenTextSplitterFactory implements GGUFTextSplitterFactory {
    public static final String[] TEKKEN_PATTERN = {
        "[^\\r"
            + "\\n"
            + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r"
            + "\\n"
            + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}|"
            + " ?[^\\s\\p{L}\\p{N}]+[\\r"
            + "\\n"
            + "/]*|\\s*[\\r"
            + "\\n"
            + "]+|\\s+(?!\\S)|\\s+"
    };

    @Override
    public String getTextSplitterName() {
        return "tekken";
    }

    @Override
    public Splitter createTextSplitter(GGUF gguf) {
        Splitter[] splitters =
                Arrays.stream(TEKKEN_PATTERN).map(RegexSplitter::create).toArray(Splitter[]::new);
        return Splitter.sequence(splitters);
    }
}
