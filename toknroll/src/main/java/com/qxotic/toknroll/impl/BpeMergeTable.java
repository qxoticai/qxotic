package com.qxotic.toknroll.impl;

/** Lookup table for merge candidates in BPE. */
public interface BpeMergeTable {

    /** Returns packed merge info or {@link IntPair#NONE} when absent. */
    long mergeInfo(int leftTokenId, int rightTokenId);
}
