package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import java.util.Set;

/**
 * A {@link LanguageModel} together with the two token-level facts a GGUF file carries about its
 * text: the tokenizer and the ids that end a turn. Everything here speaks tokens - the model's chat
 * framing lives one layer up, on {@code chat.ChatModel}.
 *
 * <p>These are data, not behaviour, so they are a record rather than an interface on the model. A
 * caller that wants different stop tokens simply builds another record; a caller that only needs
 * logits passes {@link #model()} and never sees the rest. {@link Generator} takes the model
 * explicitly for exactly that reason.
 *
 * <p>Produced by each model class ({@code loaded()}); the architecture-dispatching loaders (the
 * server's {@code Models.load}) bundle it into a {@code ChatModel}.
 */
public record LoadedModel<S extends RuntimeState>(
        LanguageModel<?, ?, S> model, GgufTokenizer tokenizer, Set<Integer> stopTokens) {

    public LoadedModel {
        if (model == null) throw new IllegalArgumentException("null model");
        if (tokenizer == null) throw new IllegalArgumentException("null tokenizer");
        stopTokens = Set.copyOf(stopTokens);
    }

    /** The same model with a caller-supplied stop-token set (user {@code stop} overrides). */
    public LoadedModel<S> withStopTokens(Set<Integer> stops) {
        return new LoadedModel<>(model, tokenizer, stops);
    }
}
