package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.util.Objects;

/** Binary operation node in IR-T. */
public record BinaryOp(BinaryOperator op, TIRNode left, TIRNode right, Shape shape)
        implements TIRNode {

    public BinaryOp(BinaryOperator op, TIRNode left, TIRNode right) {
        this(op, left, right, broadcastShapes(left.shape(), right.shape()));
    }

    public BinaryOp {
        Objects.requireNonNull(op);
        Objects.requireNonNull(left);
        Objects.requireNonNull(right);
        Objects.requireNonNull(shape);
    }

    @Override
    public DataType dataType() {
        return switch (op) {
            case EQUAL, LESS_THAN -> com.qxotic.jota.DataType.BOOL;
            default -> left.dataType();
        };
    }

    @Override
    public Shape shape() {
        return shape;
    }

    /** Computes the broadcast result shape following numpy broadcasting rules. */
    static Shape broadcastShapes(Shape a, Shape b) {
        int rankA = a.flatRank();
        int rankB = b.flatRank();
        int maxRank = Math.max(rankA, rankB);

        long[] result = new long[maxRank];
        for (int i = 0; i < maxRank; i++) {
            // Index from the right (like numpy broadcasting)
            long dimA = (i < rankA) ? a.flatAt(rankA - 1 - i) : 1;
            long dimB = (i < rankB) ? b.flatAt(rankB - 1 - i) : 1;

            if (dimA == dimB) {
                result[maxRank - 1 - i] = dimA;
            } else if (dimA == 1) {
                result[maxRank - 1 - i] = dimB;
            } else if (dimB == 1) {
                result[maxRank - 1 - i] = dimA;
            } else {
                throw new IllegalArgumentException("Shapes not broadcastable: " + a + " and " + b);
            }
        }
        return Shape.flat(result);
    }
}
