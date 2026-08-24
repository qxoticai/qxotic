package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Shape;
import java.util.Arrays;

/**
 * Describes the kind of view transformation applied to a tensor. Each variant captures the
 * parameters needed to invert the transformation when computing index expressions.
 */
public sealed interface ViewKind {

    /** Transpose: permutes dimensions according to the given permutation. */
    record Transpose(int[] permutation) implements ViewKind {
        public Transpose {
            if (permutation == null || permutation.length == 0) {
                throw new IllegalArgumentException("permutation cannot be null or empty");
            }
            boolean[] seen = new boolean[permutation.length];
            for (int axis : permutation) {
                if (axis < 0 || axis >= permutation.length || seen[axis]) {
                    throw new IllegalArgumentException("invalid permutation");
                }
                seen[axis] = true;
            }
            permutation = permutation.clone();
        }

        @Override
        public int[] permutation() {
            return permutation.clone();
        }

        /** Returns the inverse permutation. */
        public int[] inverse() {
            int[] inv = new int[permutation.length];
            for (int i = 0; i < permutation.length; i++) {
                inv[permutation[i]] = i;
            }
            return inv;
        }

        @Override
        public boolean equals(Object other) {
            return this == other
                    || other instanceof Transpose transpose
                            && Arrays.equals(permutation, transpose.permutation);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(permutation);
        }

        @Override
        public String toString() {
            return "Transpose[permutation=" + Arrays.toString(permutation) + "]";
        }
    }

    /** Reshape: changes the shape without changing element order. */
    record Reshape(Shape fromShape, Shape toShape) implements ViewKind {
        public Reshape {
            if (fromShape == null || toShape == null) {
                throw new IllegalArgumentException("shapes cannot be null");
            }
            if (fromShape.size() != toShape.size()) {
                throw new IllegalArgumentException(
                        "reshape requires same number of elements: "
                                + fromShape
                                + " vs "
                                + toShape);
            }
        }
    }

    /** Broadcast: expands singleton dimensions to match target shape. */
    record Broadcast(Shape fromShape, Shape toShape) implements ViewKind {
        public Broadcast {
            if (fromShape == null || toShape == null) {
                throw new IllegalArgumentException("shapes cannot be null");
            }
        }
    }

    /** Expand: similar to broadcast but for explicit expansion of size-1 dims. */
    record Expand(Shape fromShape, Shape toShape) implements ViewKind {
        public Expand {
            if (fromShape == null || toShape == null) {
                throw new IllegalArgumentException("shapes cannot be null");
            }
        }
    }

    /** Slice: extracts a range from a dimension with optional step. */
    record Slice(int axis, long start, long step) implements ViewKind {
        public Slice {
            if (step == 0) {
                throw new IllegalArgumentException("step cannot be zero");
            }
        }
    }
}
