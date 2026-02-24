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
    public IntSequence subSequence(int start, int end) {
        if (start < 0 || end < 0 || start < end) {
            throw new IllegalArgumentException("invalid slice range");
        }
        return this;
    }
}
