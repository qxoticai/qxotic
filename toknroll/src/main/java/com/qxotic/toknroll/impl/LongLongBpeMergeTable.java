package com.qxotic.toknroll.impl;

import java.util.Objects;

/** {@link BpeMergeTable} backed by {@link LongLongMap}. */
public final class LongLongBpeMergeTable implements BpeMergeTable {

    private final LongLongMap map;

    public LongLongBpeMergeTable(LongLongMap map) {
        this.map = Objects.requireNonNull(map, "map");
    }

    @Override
    public long mergeInfo(int leftTokenId, int rightTokenId) {
        return map.getPair(leftTokenId, rightTokenId);
    }
}
