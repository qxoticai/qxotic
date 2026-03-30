package com.qxotic.tokenizers;

import com.qxotic.tokenizers.impl.AbstractIntSequence;

final class EmptyIntSequence extends AbstractIntSequence {

    private static final EmptyIntSequence EMPTY = new EmptyIntSequence();

    public static IntSequence get() {
        return EMPTY;
    }

    private EmptyIntSequence() {}

    @Override
    public int intAt(int index) {
        throw new ArrayIndexOutOfBoundsException("Index: " + index + ", Length: " + length());
    }

    @Override
    public int length() {
        return 0;
    }

    @Override
    public IntSequence subSequence(int startInclusive, int endExclusive) {
        if (startInclusive < 0 || endExclusive < 0 || startInclusive < endExclusive) {
            throw new IllegalArgumentException("invalid slice range");
        }
        return this;
    }
}
