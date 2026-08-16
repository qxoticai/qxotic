package com.qxotic.jinfer.x.boundary;

import com.qxotic.jota.memory.MemoryView;

/** A context model with a vocabulary-logits head. */
public interface LanguageModel<C extends ContextConfiguration, W, S extends ContextState>
        extends ContextModel<C, W, S> {

    /** Logits for the {@code output}-th retained row. Safe for direct use. */
    MemoryView<?> logits(S state, int output);

    /** Logits for the final retained row. */
    default MemoryView<?> logits(S state) {
        return state.exclusively(() -> logits(state, state.outputCount() - 1));
    }
}
