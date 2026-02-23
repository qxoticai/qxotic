package com.qxotic.model.llm;

public interface ModelLoaderFactory<Source> {
    default String getFormatName() {
        // TODO(peterssen): Remove.
        return "gguf";
    }

    String getArchitectureName();

    ModelLoader createLoader(Source source);
}
