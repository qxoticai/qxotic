package com.llm4j.model;

import com.llm4j.tokenizers.TextSplitter;

// TextSplitterFactory interface
public interface TextSplitterFactory<Source> {
    String getSourceName();

    String getTextSplitterName();

    TextSplitter createTextSplitter(Source source);
}
