package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.TurnTemplate;
import java.util.Optional;
import java.util.Set;

/**
 * A {@link LanguageModel} together with the three facts a GGUF file carries about its text: the
 * tokenizer, the ids that end a turn, and the model's curated chat framing (when one exists).
 *
 * <p>These are data, not behaviour, so they are a record rather than an interface on the model. A
 * caller that wants different stop tokens simply builds another record; a caller that only needs
 * logits passes {@link #model()} and never sees the rest. {@link Generator} takes the model and the
 * tokenizer explicitly for exactly that reason.
 *
 * <p>Produced by the architecture-dispatching loaders (the server's {@code Models.load}), which are
 * the only callers that hold a model of unknown family and still need its text facts.
 */
public record LoadedModel<S extends RuntimeState>(
        LanguageModel<?, ?, S> model,
        GgufTokenizer tokenizer,
        Set<Integer> stopTokens,
        Optional<TurnTemplate> chatTemplate) {

    public LoadedModel {
        if (model == null) throw new IllegalArgumentException("null model");
        if (tokenizer == null) throw new IllegalArgumentException("null tokenizer");
        stopTokens = Set.copyOf(stopTokens);
        if (chatTemplate == null) throw new IllegalArgumentException("null chatTemplate");
    }

    /** The same model with a caller-supplied stop-token set (user {@code stop} overrides). */
    public LoadedModel<S> withStopTokens(Set<Integer> stops) {
        return new LoadedModel<>(model, tokenizer, stops, chatTemplate);
    }
}
