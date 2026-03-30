package com.qxotic.tokenizers;

import com.qxotic.tokenizers.impl.AbstractIntSequence;

final class ZeroIntSequence extends AbstractIntSequence {

    private final int length;

    public ZeroIntSequence(int length) {
        if (length < 0) {
            throw new IllegalArgumentException("negative length");
        }
        this.length = length;
    }

    @Override
    public int intAt(int index) {
        if (index < 0 || index >= length()) {
            throw new ArrayIndexOutOfBoundsException("Index: " + index + ", Length: " + length());
        }
        return 0;
    }

    @Override
    public int length() {
        return this.length;
    }

    @Override
    public IntSequence subSequence(int startInclusive, int endExclusive) {
        if (startInclusive < 0 || endExclusive < 0 || startInclusive > endExclusive) {
            throw new IllegalArgumentException("slice out of range");
        }
        return new ZeroIntSequence(endExclusive - startInclusive);
    }
}
