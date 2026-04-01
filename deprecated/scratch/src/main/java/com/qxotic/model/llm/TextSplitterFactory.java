package com.qxotic.model.llm;

import com.qxotic.toknroll.advanced.Splitter;

// TextSplitterFactory interface
public interface TextSplitterFactory<Source> {
    String getSourceName();

    String getTextSplitterName();

    Splitter createTextSplitter(Source source);
}
