package ai.qxotic.jota.ir.irl;

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
        SUB,
        MUL,
        DIV,
        MOD
    }

    /** Creates an addition of two index expressions. */
    public static IndexBinary add(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.ADD, left, right);
    }

    /** Creates a subtraction of two index expressions. */
    public static IndexBinary sub(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.SUB, left, right);
    }

    /** Creates a multiplication of two index expressions. */
    public static IndexBinary mul(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.MUL, left, right);
    }

    /** Creates a division of two index expressions. */
    public static IndexBinary div(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.DIV, left, right);
    }

    /** Creates a modulo of two index expressions. */
    public static IndexBinary mod(IndexExpr left, IndexExpr right) {
        return new IndexBinary(IndexBinaryOp.MOD, left, right);
    }
}
