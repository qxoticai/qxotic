package com.qxotic.jinfer;

/** Configuration shared by models that ingest a bounded positional context. */
public interface ContextConfiguration {

    int vocabularySize();

    /**
     * Model-declared maximum context length. This describes the model, not the capacity allocated
     * by any particular state.
     */
    int contextLength();
}
