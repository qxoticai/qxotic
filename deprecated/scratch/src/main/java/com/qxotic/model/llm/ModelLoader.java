package com.qxotic.model.llm;

import com.qxotic.toknroll.Tokenizer;

public interface ModelLoader<M extends Model<C, W, S>, C, W, S> {
    M loadModel(C configuration);

    ChatFormat createChatFormat(Tokenizer tokenizer);

    C loadConfiguration(int maxTokens, SpanLoader spanLoader);

    W loadWeights(C configuration, SpanLoader spanLoader);
}
