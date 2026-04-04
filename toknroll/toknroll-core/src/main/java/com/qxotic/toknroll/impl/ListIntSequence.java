package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Objects;

final class ListIntSequence extends AbstractIntSequence {
    private final List<Integer> list;

    ListIntSequence(List<Integer> list) {
        this.list = Objects.requireNonNull(list);
    }

    @Override
    public int intAt(int index) {
        return list.get(index);
    }

    @Override
    public int length() {
        return list.size();
    }

    @Override
    public IntSequence subSequence(int startInclusive, int endExclusive) {
        return new ListIntSequence(list.subList(startInclusive, endExclusive));
    }
}
