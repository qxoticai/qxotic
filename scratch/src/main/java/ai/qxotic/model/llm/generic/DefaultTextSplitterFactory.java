package ai.qxotic.model.llm.generic;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFTextSplitterFactory;
import ai.qxotic.model.llm.TextSplitterFactory;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.impl.RegexSplitter;
import com.google.auto.service.AutoService;
import java.util.Arrays;

// Implementation for default pre-tokenizer
@AutoService(TextSplitterFactory.class)
public class DefaultTextSplitterFactory implements GGUFTextSplitterFactory {
    @Override
    public String getTextSplitterName() {
        return "default";
    }

    @Override
    public TextSplitter createTextSplitter(GGUF gguf) {
        TextSplitter[] preTokenizers =
                Arrays.stream(RegexSplitter.DEFAULT_BPE_SPLITS)
                        .map(RegexSplitter::create)
                        .toArray(TextSplitter[]::new);
        return TextSplitter.compose(preTokenizers);
    }
}
