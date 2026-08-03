package com.qxotic.jinfer;

import com.qxotic.jinfer.cache.StateCodec;
import java.util.Optional;

/**
 * An LLM: a {@link com.qxotic.jinfer.Model} backbone whose head projects retained hidden states to
 * a vocabulary distribution (of width {@code config().vocabularySize()}).
 *
 * <p>Tokens in, logits out. This interface knows nothing of text: no tokenizer, no stop tokens, no
 * chat framing. Those live one layer up, on {@code com.qxotic.jinfer.chat.LoadedModel}, the record
 * the architecture-dispatching loaders return.
 */
public interface LanguageModel<C extends Config, W, S extends RuntimeState> extends Model<C, W, S> {

    /** The prompt-cache resume-state codec for this model, when caching is supported. Stateless. */
    default Optional<StateCodec<S>> stateCodec() {
        return Optional.empty();
    }

    /**
     * A state sized for one generation over a {@code promptLen}-token prompt: full context, batch
     * capacity clamped to the prompt (min 16 rows - the loop needs a tail token with fresh logits;
     * max {@link RuntimeFlags#BATCH_CAPACITY}). Owns the sizing policy consumers used to hand-roll.
     */
    default S stateFor(int promptLen) {
        return newState(
                config().contextLength(),
                Math.min(Math.max(promptLen, 16), RuntimeFlags.BATCH_CAPACITY));
    }

    /** Vocabulary logits for the {@code output}-th retained hidden state (0 .. outputCount()-1). */
    default FloatTensor logits(S state, int output) {
        BaseState base = (BaseState) state;
        base.enter();
        FloatTensor logits;
        try {
            logits = head(state, output);
        } finally {
            base.exit();
        }
        java.lang.ref.Reference.reachabilityFence(this);
        return logits;
    }

    /** The LM-head projection behind {@link #logits} - the implementation seam. */
    FloatTensor head(S state, int output);

    /** The last retained output — the next-token distribution after the last input row. */
    default FloatTensor logits(S state) {
        return logits(state, state.outputCount() - 1);
    }
}
