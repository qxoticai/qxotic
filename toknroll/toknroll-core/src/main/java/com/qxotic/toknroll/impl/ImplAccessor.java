package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import java.util.List;

public final class ImplAccessor {

    private static final int[] EMPTY_ARRAY = new int[0];
    private static final IntSequence EMPTY_SEQUENCE = wrap(EMPTY_ARRAY);

    public static IntSequence empty() {
        return EMPTY_SEQUENCE;
    }

    public static IntSequence.Builder newBuilder() {
        return new IntSequenceBuilder();
    }

    public static IntSequence.Builder newBuilder(int initialCapacity) {
        return new IntSequenceBuilder(initialCapacity);
    }

    public static IntSequence wrap(int[] array) {
        return new ArrayIntSequence(array);
    }

    public static IntSequence wrap(List<Integer> list) {
        return new ListIntSequence(list);
    }
}
