package ai.qxotic.model.llm;

import ai.qxotic.format.gguf.GGUF;

public interface GGUFTokenizerFactory extends TokenizerFactory<GGUF> {
    @Override
    default String getSourceName() {
        return "gguf";
    }
}
