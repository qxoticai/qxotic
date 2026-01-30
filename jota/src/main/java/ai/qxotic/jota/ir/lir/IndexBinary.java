package ai.qxotic.jota.ir.lir;

import java.util.Objects;

/** Binary operation on index expressions. */
public record IndexBinary(IndexBinaryOp op, IndexExpr left, IndexExpr right) implements IndexExpr {

    public IndexBinary {
        Objects.requireNonNull(op, "op cannot be null");
        Objects.requireNonNull(left, "left cannot be null");
        Objects.requireNonNull(right, "right cannot be null");
    }

    /** Binary operators for index expressions. */
    public enum IndexBinaryOp {
        ADD,
        SUBTRACT,
        MULTIPLY,
        DIVIDE,
        MODULO
    }

    /** Creates an addition of two index expressions. */
    public static IndexBinary add(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.ADD, left, right);
    }

    /** Creates a subtraction of two index expressions. */
    public static IndexBinary subtract(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.SUBTRACT, left, right);
    }

    /** Creates a multiplication of two index expressions. */
    public static IndexBinary multiply(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.MULTIPLY, left, right);
    }

    /** Creates a division of two index expressions. */
    public static IndexBinary divide(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.DIVIDE, left, right);
    }

    /** Creates a modulo of two index expressions. */
    public static IndexBinary modulo(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.MODULO, left, right);
    }
}
