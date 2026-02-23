package com.qxotic.tokenizers.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.advanced.Splitter;

@FunctionalInterface
public interface GGUFTokenizerFactory {

    Tokenizer create(GGUF gguf, Splitter splitter);
}
