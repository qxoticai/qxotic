package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.llm.LoadedModel;
import java.util.Optional;

/**
 * The chat-layer binding: a token-level {@link LoadedModel} plus the model's {@link ChatTemplate}
 * codec (when one exists - models without one fall back to whole-render). This record is the ONLY
 * place the two layers meet; everything below it speaks tokens, everything above it speaks
 * messages.
 *
 * <p>Produced by each model class ({@code chatModel()}) and by the architecture-dispatching loaders
 * (the server's {@code Models.load}).
 */
public record ChatModel<S extends RuntimeState>(
        LoadedModel<S> base, Optional<ChatTemplate> template) {

    public ChatModel {
        if (base == null) throw new IllegalArgumentException("null base");
        if (template == null) throw new IllegalArgumentException("null template");
    }

    /**
     * The binding for an unported model: its per-turn {@link TurnTemplate} bridged onto the codec
     * through {@link TurnTemplateAdapter} (models with a native codec construct directly).
     */
    public static <S extends RuntimeState> ChatModel<S> adapt(
            LoadedModel<S> base, Optional<TurnTemplate> turn) {
        return new ChatModel<>(
                base, turn.map(t -> (ChatTemplate) new TurnTemplateAdapter(t, base.tokenizer())));
    }
}
