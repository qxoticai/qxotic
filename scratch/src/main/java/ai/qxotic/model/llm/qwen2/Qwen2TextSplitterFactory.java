package ai.qxotic.model.llm.qwen2;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFTextSplitterFactory;
import ai.qxotic.model.llm.TextSplitterFactory;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.impl.RegexSplitter;
import com.google.auto.service.AutoService;

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
