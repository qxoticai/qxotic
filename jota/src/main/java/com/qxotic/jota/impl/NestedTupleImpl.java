package com.qxotic.jota.impl;

import com.qxotic.jota.Util;

import java.util.Arrays;
import java.util.Objects;
import java.util.StringJoiner;

abstract class NestedTupleImpl<T extends NestedTuple<T>> implements NestedTuple<T> {

    protected static final long[] EMPTY = new long[0];

    final long[] flat;
    final int[] parent;

    NestedTupleImpl(long[] flat, int[] parent) {
        this.flat = flat;
        this.parent = parent;
    }

    @Override
    public int flatRank() {
        return flat.length;
    }

    @Override
    public long flatAt(int _flatIndex) {
        int flatIndex = Util.wrapAround(_flatIndex, flatRank());
        return flat[flatIndex];
    }

    @Override
    public long[] toArray() {
        return flat.clone();
    }

    @Override
    public String toString() {
        if (isFlat()) {
            return Arrays.toString(this.flat);
        }
        return nestedToString();
    }

    private String nestedToString() {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (int i = 0; i < rank(); ++i) {
            NestedTuple<?> mode = modeAt(i);
            if (mode.flatRank() == 1) { // [x] -> x
                joiner.add("" + mode.flatAt(0));
            } else {
                joiner.add(mode.toString());
            }
        }
        return joiner.toString();
    }

    @Override
    public boolean isCongruentWith(NestedTuple<?> other) {
        Objects.requireNonNull(other);

        // Must have same rank and flatRank
        if (this.rank() != other.rank() || this.flatRank() != other.flatRank()) {
            return false;
        }

        // Both flat -> congruent
        if (this.isFlat() && other.isFlat()) {
            return true;
        }

        // One flat, one nested -> not congruent
        if (this.isFlat() != other.isFlat()) {
            return false;
        }

        // Both nested -> check parent structure
        if (other instanceof NestedTupleImpl<?> otherImpl) {
            return Arrays.equals(this.parent, otherImpl.parent);
        }

        throw new IllegalArgumentException("Unsupported NestedTuple implementation");
    }
}
