package com.qxotic.toknroll;

final class ZeroIntSequence implements IntSequence {

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

    @Override
    public int compareTo(IntSequence other) {
        return IntSequence.compare(this, other);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        return other instanceof IntSequence && IntSequence.contentEquals(this, (IntSequence) other);
    }

    @Override
    public int hashCode() {
        int hash = 1;
        for (int i = 0; i < length; i++) {
            hash = hash * 31;
        }
        return hash;
    }
}
