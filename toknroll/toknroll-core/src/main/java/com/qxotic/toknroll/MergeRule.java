package com.qxotic.toknroll;

/** A BPE merge rule: merge token {@code leftId} with {@code rightId} at the given rank. */
public final class MergeRule {
    private final int leftId;
    private final int rightId;
    private final int rank;

    /**
     * Creates a merge rule.
     *
     * @param leftId non-negative left token ID
     * @param rightId non-negative right token ID
     * @param rank merge priority rank (lower ranks are applied first)
     * @throws IllegalArgumentException if {@code leftId} or {@code rightId} is negative
     */
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

    /**
     * @return left-side token ID
     */
    public int leftId() {
        return leftId;
    }

    /**
     * @return right-side token ID
     */
    public int rightId() {
        return rightId;
    }

    /**
     * @return merge priority rank
     */
    public int rank() {
        return rank;
    }
}
