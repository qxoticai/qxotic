package com.qxotic.toknroll;

/** A BPE merge rule: merge token {@code leftId} with {@code rightId} at the given rank. */
public final class MergeRule {
    private final int leftId;
    private final int rightId;
    private final int rank;

    public static MergeRule of(int leftId, int rightId, int rank) {
        return new MergeRule(leftId, rightId, rank);
    }

    MergeRule(int leftId, int rightId, int rank) {
        if (leftId < 0) {
            throw new IllegalArgumentException("leftId must be non-negative: " + leftId);
        }
        if (rightId < 0) {
            throw new IllegalArgumentException("rightId must be non-negative: " + rightId);
        }
        this.leftId = leftId;
        this.rightId = rightId;
        this.rank = rank;
    }

    public int leftId() {
        return leftId;
    }

    public int rightId() {
        return rightId;
    }

    public int rank() {
        return rank;
    }
}
