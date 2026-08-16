package com.qxotic.jinfer.x.boundary;

/** Configuration shared by models that ingest a bounded positional context. */
public interface ContextConfiguration {

    int vocabularySize();

    int contextLength();
}
