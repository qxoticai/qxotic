package ai.qxotic.model.llm.smollm2;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFTextSplitterFactory;
import ai.qxotic.model.llm.TextSplitterFactory;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.impl.RegexSplitter;
import com.google.auto.service.AutoService;
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
    public TextSplitter createTextSplitter(GGUF gguf) {
        TextSplitter[] preTokenizers =
                Arrays.stream(SMOLLM2_PATTERNS)
                        .map(RegexSplitter::create)
                        .toArray(TextSplitter[]::new);
        return TextSplitter.compose(preTokenizers);
    }
}
