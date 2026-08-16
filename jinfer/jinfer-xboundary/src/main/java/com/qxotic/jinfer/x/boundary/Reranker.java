package com.qxotic.jinfer.x.boundary;

import java.util.List;
import java.util.function.DoubleConsumer;

/** A task recipe that frames and scores inputs with a context model. */
public interface Reranker<S extends ContextState> {

    ContextModel<?, ?, S> model();

    String defaultInstruction();

    default boolean hasInstructionSlot() {
        return true;
    }

    /** Scores all documents safely; implementations own exclusive access to {@code state}. */
    int scoreAll(
            S state, String instruction, String query, List<String> documents, DoubleConsumer sink);
}
