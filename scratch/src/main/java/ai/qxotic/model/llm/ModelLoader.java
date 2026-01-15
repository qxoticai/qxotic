package ai.qxotic.model.llm;

import ai.qxotic.model.llm.Model;
import ai.qxotic.tokenizers.Tokenizer;

public interface ModelLoader<M extends Model<C, W, S>, C, W, S> {
    M loadModel(C configuration);

    ChatFormat createChatFormat(Tokenizer tokenizer);

    C loadConfiguration(int maxTokens, SpanLoader spanLoader);

    W loadWeights(C configuration, SpanLoader spanLoader);
}
