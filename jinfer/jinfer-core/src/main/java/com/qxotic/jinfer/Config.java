package com.qxotic.jinfer;

/**
 * The facts universal to every model — independent of its output head. The token space it operates
 * over (the input embedding-table size — every token-ingesting model has one, generative or not)
 * and the position ceiling (to bound state allocation). Nothing about tokenization itself.
 */
public interface Config {
    int vocabularySize(); // the model's token space (input embedding table)

    int contextLength(); // largest context a state may allocate; newState must not exceed it
}
