package ai.qxotic.model.llm;

import ai.qxotic.format.gguf.GGUF;

public interface GGUFModelLoaderFactory extends ModelLoaderFactory<GGUF> {
    @Override
    default String getFormatName() {
        return "gguf";
    }
}

