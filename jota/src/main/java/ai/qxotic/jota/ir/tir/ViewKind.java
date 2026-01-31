package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.Shape;

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
        }

        /** Returns the inverse permutation. */
        public int[] inverse() {
            int[] inv = new int[permutation.length];
            for (int i = 0; i < permutation.length; i++) {
                inv[permutation[i]] = i;
            }
            return inv;
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
