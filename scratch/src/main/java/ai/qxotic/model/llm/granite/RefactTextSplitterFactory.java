package ai.qxotic.model.llm.granite;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFTextSplitterFactory;
import ai.qxotic.model.llm.TextSplitterFactory;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.impl.RegexSplitter;
import com.google.auto.service.AutoService;
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
    public TextSplitter createTextSplitter(GGUF gguf) {
        TextSplitter[] preTokenizers =
                Arrays.stream(GRANITE_PATTERNS)
                        .map(RegexSplitter::create)
                        .toArray(TextSplitter[]::new);
        return TextSplitter.compose(preTokenizers);
    }
}
