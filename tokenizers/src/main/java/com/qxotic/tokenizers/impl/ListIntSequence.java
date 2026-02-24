package com.qxotic.tokenizers.impl;

import com.qxotic.tokenizers.IntSequence;
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
    public IntSequence subSequence(int start, int end) {
        return new ListIntSequence(list.subList(start, end));
    }
}
