package com.qxotic.jota.impl;

import com.qxotic.jota.Util;

import java.util.Arrays;
import java.util.Objects;
import java.util.StringJoiner;

abstract class NestedTupleImpl<T extends NestedTuple<T>> implements NestedTuple<T> {

    protected static final long[] EMPTY = new long[0];

    final long[] flat;
    final int[] nest;

    NestedTupleImpl(long[] flat, int[] nest) {
        assert flat != null : "Flat array cannot be null";

        // Nest can be null (indicates flat structure)
        // If nest is not null, it must match the flat array length
        assert nest == null || flat.length == nest.length :
            "Flat and nest arrays must have the same length: flat.length=" + flat.length + ", nest.length=" + nest.length;

        // Validate nest array structure (if not null)
        if (nest != null) {
            assertValidNestArray(nest);
        }

        this.flat = flat;
        this.nest = nest;
    }

    /**
     * Assert that the nest array structure is valid.
     * This is called from the constructor to catch structural errors early.
     */
    private static void assertValidNestArray(int[] nest) {
        int depth = 0;
        for (int value : nest) {
            int open = Math.max(value, 0);
            int close = Math.max(-value, 0);
            depth += open;
            assert depth >= 0 : "Invalid nest array: negative depth after opening";
            depth -= close;
            assert depth >= 0 : "Invalid nest array: negative depth after closing";
        }
        assert depth == 0 : "Invalid nest array: unbalanced nesting";
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
        if (isScalar()) {
            return "()";
        }
        StringBuilder builder = new StringBuilder();
        builder.append('(');
        for (int i = 0; i < flatRank(); i++) {
            if (i > 0) {
                builder.append(", ");
            }
            int value = nest == null ? 0 : nest[i];
            int open = Math.max(value, 0);
            int close = Math.max(-value, 0);
            builder.append("(".repeat(open));
            builder.append(flat[i]);
            builder.append(")".repeat(close));
        }
        builder.append(')');
        return builder.toString();
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

        // Both nested -> check nest structure
        if (other instanceof NestedTupleImpl<?> otherImpl) {
            return Arrays.equals(this.nest, otherImpl.nest);
        }

        throw new IllegalArgumentException("Unsupported NestedTuple implementation");
    }
}
