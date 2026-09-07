package com.qxotic.jinfer;

import java.util.List;
import java.util.function.DoubleConsumer;

/** A task recipe that frames and scores inputs with a context model. */
public interface Reranker<S extends ContextState> {

    ContextModel<?, ?, S> model();

    String defaultInstruction();

    default boolean hasInstructionSlot() {
        return true;
    }

    /**
     * Scores all documents safely; implementations own exclusive access to {@code state}.
     *
     * @throws IllegalArgumentException when a document does not fit the context - the message says
     *     which document, and how to fit it: a larger context, or a smaller document
     */
    int scoreAll(
            S state, String instruction, String query, List<String> documents, DoubleConsumer sink);
}
