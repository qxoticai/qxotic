package com.llm4j.model;

import com.llm4j.tokenizers.Normalizer;
import com.llm4j.tokenizers.TextSplitter;
import com.llm4j.tokenizers.Tokenizer;

// TokenizerFactory interface that implementations will need to implement
public interface TokenizerFactory<Source> {

    String getSourceName();

    String getTokenizerName();

    Tokenizer createTokenizer(Source source, Normalizer normalizer, TextSplitter textSplitter);
}
