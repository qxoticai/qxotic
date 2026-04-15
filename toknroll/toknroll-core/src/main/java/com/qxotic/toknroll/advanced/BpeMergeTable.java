package com.qxotic.toknroll.advanced;

/** Lookup table for merge candidates in BPE. */
public interface BpeMergeTable {

    /** Returns packed merge info or {@code IntPair.NONE}-equivalent when absent. */
    long mergeInfo(int leftTokenId, int rightTokenId);
}
