package com.llm4j.model;

import com.llm4j.api.model.Model;
import com.llm4j.tokenizers.Tokenizer;

public interface ModelLoader<M extends Model<C, W, S>, C, W, S> {
    M loadModel(C configuration);

    ChatFormat createChatFormat(Tokenizer tokenizer);

    C loadConfiguration(int maxTokens, SpanLoader spanLoader);

    W loadWeights(C configuration, SpanLoader spanLoader);
}
