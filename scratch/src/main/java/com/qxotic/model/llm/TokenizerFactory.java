package com.qxotic.model.llm;

import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;

// TokenizerFactory interface that implementations will need to implement
public interface TokenizerFactory<Source> {

    String getSourceName();

    String getTokenizerName();

    Tokenizer createTokenizer(Source source, Normalizer normalizer, Splitter splitter);
}
