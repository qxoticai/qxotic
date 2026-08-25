package com.qxotic.jota.ir.tir;

import com.qxotic.jota.Shape;
import java.util.Arrays;

/** Describes a view operation and the parameters needed to invert it during lowering. */
public sealed interface ViewOperation {

    /** Permutes tensor axes. */
    record Transpose(int[] permutation) implements ViewOperation {
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
            int[] inverse = new int[permutation.length];
            for (int i = 0; i < permutation.length; i++) {
                inverse[permutation[i]] = i;
            }
            return inverse;
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

    /** Changes shape without changing element order. */
    record Reshape(Shape fromShape, Shape toShape) implements ViewOperation {
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

    /** Adds or expands broadcast dimensions. */
    record Broadcast(Shape fromShape, Shape toShape) implements ViewOperation {
        public Broadcast {
            if (fromShape == null || toShape == null) {
                throw new IllegalArgumentException("shapes cannot be null");
            }
        }
    }

    /** Expands singleton dimensions within an existing shape. */
    record Expand(Shape fromShape, Shape toShape) implements ViewOperation {
        public Expand {
            if (fromShape == null || toShape == null) {
                throw new IllegalArgumentException("shapes cannot be null");
            }
        }
    }

    /** Selects a strided range along one axis. */
    record Slice(int axis, long start, long step) implements ViewOperation {
        public Slice {
            if (step == 0) {
                throw new IllegalArgumentException("step cannot be zero");
            }
        }
    }
}
