package com.qxotic.jinfer.x.boundary;

import com.qxotic.jota.memory.MemoryView;
import java.lang.ref.Reference;

/**
 * An LLM: a {@link Model} backbone whose head projects retained hidden states to a vocabulary
 * distribution (of width {@code config().vocabularySize()}).
 *
 * <p>Tokens in, logits out. This interface knows nothing of text: no tokenizer, no stop tokens, no
 * chat framing. The boundary speaks {@link MemoryView}: the returned view is FP32 of shape {@code
 * [vocab]}, a ZERO-COPY slice of the state's scratch — a REUSED per-state buffer, valid until the
 * next head call.
 */
public interface LanguageModel<C extends Config, W, S extends RuntimeState> extends Model<C, W, S> {

    /** Vocabulary logits for the {@code output}-th retained hidden state (0 .. outputCount()-1). */
    default MemoryView<?> logits(S state, int output) {
        BaseState base = (BaseState) state;
        base.enter();
        MemoryView<?> logits;
        try {
            logits = head(state, output);
        } finally {
            base.exit();
        }
        Reference.reachabilityFence(this);
        return logits;
    }

    /** The LM-head projection behind {@link #logits} - the implementation seam. */
    MemoryView<?> head(S state, int output);

    /** The last retained output — the next-token distribution after the last input row. */
    default MemoryView<?> logits(S state) {
        return logits(state, state.outputCount() - 1);
    }
}
