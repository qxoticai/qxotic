package ai.qxotic.model.llm;

import ai.qxotic.format.gguf.GGUF;

public interface GGUFTextSplitterFactory extends TextSplitterFactory<GGUF> {
    @Override
    default String getSourceName() {
        return "gguf";
    }
}
