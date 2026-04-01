package com.qxotic.model.llm;

import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;

// TokenizerFactory interface that implementations will need to implement
public interface TokenizerFactory<Source> {

    String getSourceName();

    String getTokenizerName();

    Tokenizer createTokenizer(Source source, Normalizer normalizer, Splitter splitter);
}
