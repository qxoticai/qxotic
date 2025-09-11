package com.llm4j.tokenizers;

import com.llm4j.tokenizers.impl.AbstractIntSequence;

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
    public IntSequence subSequence(int start, int end) {
        if (start < 0 || end < 0 || start > end) {
            throw new IllegalArgumentException("slice out of range");
        }
        return new ZeroIntSequence(end - start);
    }
}
