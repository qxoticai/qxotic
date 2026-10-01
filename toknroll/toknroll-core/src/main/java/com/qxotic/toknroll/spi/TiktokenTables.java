package com.qxotic.toknroll.spi;

/**
 * What a tiktoken model is, for a {@link TiktokenEngine}: the bytes of every token and the merges
 * between them. A word encodes as its bytes' single-byte tokens, merged pair by pair, the pair with
 * the lowest rank first and the leftmost among equal ranks, until no pair merges.
 */
public interface TiktokenTables {

    /** One more than the largest token id. */
    int tokenCount();

    /** The bytes of a token, or null for an id without one. The array must not be modified. */
    byte[] tokenBytes(int id);

    /** How many merges there are; each is numbered from 0 here, which is not its rank. */
    int mergeCount();

    int mergeLeft(int merge);

    int mergeRight(int merge);

    /** The token the pair merges into. */
    int mergeResult(int merge);

    int mergeRank(int merge);

    /**
     * Whether a word that is a token as a whole encodes as that token without merging, as Llama 3
     * has it ({@code ignore_merges}).
     */
    boolean ignoreMerges();
}
