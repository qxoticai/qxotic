package com.qxotic.jinfer;

import com.qxotic.jinfer.Model;

/** An LLM: a {@link com.qxotic.jinfer.Model} backbone whose head projects retained hidden states to a vocabulary
 *  distribution (of width {@code config().vocabularySize()}). */
public interface LanguageModel<C extends Config, W, S extends RuntimeState> extends Model<C, W, S> {

    /** GGUF-loaded tokenizer: vocabulary, special tokens and the (optional) chat template. Needed by
     *  the generation driver to detokenize the stream, match text stops, and detect stop tokens. */
    LFMTokenizer tokenizer();

    /** The end-of-turn / eos ids that terminate generation (the model's default stop tokens). */
    java.util.Set<Integer> stopTokens();

    /** Vocabulary logits for the {@code output}-th retained hidden state (0 .. outputCount()-1). */
    FloatTensor logits(S state, int output);

    /** The last retained output — the next-token distribution after the last input row. */
    default FloatTensor logits(S state) { return logits(state, state.outputCount() - 1); }
}
