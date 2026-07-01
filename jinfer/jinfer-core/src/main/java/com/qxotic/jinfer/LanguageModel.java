package com.qxotic.jinfer;

import com.qxotic.jinfer.Model;

/** An LLM: a {@link com.qxotic.jinfer.Model} backbone whose head projects retained hidden states to a vocabulary
 *  distribution (of width {@code config().vocabularySize()}). */
public interface LanguageModel<C extends Config, W, S extends RuntimeState> extends Model<C, W, S> {

    /** Vocabulary logits for the {@code output}-th retained hidden state (0 .. outputCount()-1). */
    FloatTensor logits(S state, int output);

    /** The last retained output — the next-token distribution after the last input row. */
    default FloatTensor logits(S state) { return logits(state, state.outputCount() - 1); }
}
