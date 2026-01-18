package ai.qxotic.model.llm.llama;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFTextSplitterFactory;
import ai.qxotic.model.llm.TextSplitterFactory;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.impl.RegexSplitter;
import com.google.auto.service.AutoService;

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
