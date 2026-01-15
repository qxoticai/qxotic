package ai.qxotic.model.llm;

import ai.qxotic.tokenizers.TextSplitter;

// TextSplitterFactory interface
public interface TextSplitterFactory<Source> {
    String getSourceName();

    String getTextSplitterName();

    TextSplitter createTextSplitter(Source source);
}
