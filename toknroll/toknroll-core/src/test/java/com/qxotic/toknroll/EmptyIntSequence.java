package com.qxotic.toknroll;

final class EmptyIntSequence implements IntSequence {

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
        return 1;
    }
}
