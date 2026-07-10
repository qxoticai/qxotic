package com.qxotic.jinfer;

import com.qxotic.jinfer.cache.StateCodec;
import java.util.Optional;

/**
 * An LLM: a {@link com.qxotic.jinfer.Model} backbone whose head projects retained hidden states to
 * a vocabulary distribution (of width {@code config().vocabularySize()}).
 *
 * <p>Tokens in, logits out. This interface knows nothing of text: no tokenizer, no stop tokens, no
 * chat framing. Those live one layer up, on {@code com.qxotic.jinfer.llm.LoadedModel}, the record
 * the architecture-dispatching loaders return.
 */
public interface LanguageModel<C extends Config, W, S extends RuntimeState> extends Model<C, W, S> {

    /** The prompt-cache resume-state codec for this model, when caching is supported. Stateless. */
    default Optional<StateCodec<S>> stateCodec() {
        return Optional.empty();
    }

    /** Vocabulary logits for the {@code output}-th retained hidden state (0 .. outputCount()-1). */
    FloatTensor logits(S state, int output);

    /** The last retained output — the next-token distribution after the last input row. */
    default FloatTensor logits(S state) {
        return logits(state, state.outputCount() - 1);
    }
}
