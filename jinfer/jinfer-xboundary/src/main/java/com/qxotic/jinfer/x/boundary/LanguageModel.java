package com.qxotic.jinfer.x.boundary;

import com.qxotic.jota.memory.MemoryView;

/** A context model with a vocabulary-logits head. */
public interface LanguageModel<C extends ContextConfiguration, W, S extends ContextState>
        extends ContextModel<C, W, S> {

    /**
     * Logits for the {@code output}-th retained row. The returned view is borrowed from the state:
     * consume it before the next model operation on that state, and before closing the state.
     *
     * <p><b>Implementation contract:</b> validate and project the output while holding {@link
     * RuntimeState#exclusively(java.util.function.Supplier) exclusive access}. Keep the model
     * strongly reachable until projection completes, normally through {@link
     * java.lang.ref.Reference#reachabilityFence(Object)}.
     */
    MemoryView<?> logits(S state, int output);

    /** Logits for the final retained row. */
    default MemoryView<?> logits(S state) {
        return state.exclusively(() -> logits(state, state.outputCount() - 1));
    }
}
