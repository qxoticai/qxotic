package ai.qxotic.jota.ir.lir;

import java.util.Objects;

/** Binary operation on index expressions with algebraic simplification. */
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
        MODULO,
        BITWISE_AND,
        SHIFT_LEFT,
        SHIFT_RIGHT
    }

    private static final IndexConst MINUS_ONE = new IndexConst(-1);

    /** Creates an addition of two index expressions (with simplification). */
    public static IndexExpr add(IndexExpr left, IndexExpr right) {
        // Fast path: try to simplify without recursing
        if (right.isZero()) return left;
        if (left.isZero()) return right;
        if (left.isConstant() && right.isConstant()) {
            return IndexConst.of(left.constantValue() + right.constantValue());
        }
        return new IndexBinary(IndexBinaryOp.ADD, left, right);
    }

    /** Creates a subtraction of two index expressions (with simplification). */
    public static IndexExpr subtract(IndexExpr left, IndexExpr right) {
        // Fast path
        if (right.isZero()) return left;
        if (left.isZero()) return new IndexBinary(IndexBinaryOp.MULTIPLY, right, MINUS_ONE);
        if (left.isConstant() && right.isConstant()) {
            return IndexConst.of(left.constantValue() - right.constantValue());
        }
        return new IndexBinary(IndexBinaryOp.SUBTRACT, left, right);
    }

    /** Creates a multiplication of two index expressions (with simplification). */
    public static IndexExpr multiply(IndexExpr left, IndexExpr right) {
        // Fast path
        if (right.isZero() || left.isZero()) return IndexConst.ZERO;
        if (right.isOne()) return left;
        if (left.isOne()) return right;
        if (left.isConstant() && right.isConstant()) {
            return IndexConst.of(left.constantValue() * right.constantValue());
        }
        return new IndexBinary(IndexBinaryOp.MULTIPLY, left, right);
    }

    /** Creates a division of two index expressions (with simplification). */
    public static IndexExpr divide(IndexExpr left, IndexExpr right) {
        // Fast path
        if (right.isOne()) return left;
        if (left.isZero()) return IndexConst.ZERO;
        if (left.isConstant() && right.isConstant() && right.constantValue() != 0) {
            return IndexConst.of(left.constantValue() / right.constantValue());
        }
        return new IndexBinary(IndexBinaryOp.DIVIDE, left, right);
    }

    /** Creates a modulo of two index expressions (with simplification). */
    public static IndexExpr modulo(IndexExpr left, IndexExpr right) {
        // Fast path
        if (left.isZero()) return IndexConst.ZERO;
        if (right.isOne()) return IndexConst.ZERO;
        if (left.isConstant() && right.isConstant() && right.constantValue() != 0) {
            return IndexConst.of(left.constantValue() % right.constantValue());
        }
        return new IndexBinary(IndexBinaryOp.MODULO, left, right);
    }

    @Override
    public IndexExpr simplify() {
        // Recursively simplify children first (bottom-up)
        IndexExpr leftSimplified = left.simplify();
        IndexExpr rightSimplified = right.simplify();

        // If children unchanged and we're already simplified, return this
        if (leftSimplified == left && rightSimplified == right) {
            // Try once more with current state
            return trySimplify(this);
        }

        // Create new node with simplified children
        IndexBinary simplified = new IndexBinary(op, leftSimplified, rightSimplified);
        return trySimplify(simplified);
    }

    private static IndexExpr trySimplify(IndexBinary bin) {
        return switch (bin.op()) {
            case ADD -> {
                if (bin.right().isZero()) yield bin.left();
                if (bin.left().isZero()) yield bin.right();
                if (bin.left().isConstant() && bin.right().isConstant()) {
                    yield IndexConst.of(bin.left().constantValue() + bin.right().constantValue());
                }
                yield bin;
            }
            case SUBTRACT -> {
                if (bin.right().isZero()) yield bin.left();
                if (bin.left().isConstant() && bin.right().isConstant()) {
                    yield IndexConst.of(bin.left().constantValue() - bin.right().constantValue());
                }
                yield bin;
            }
            case MULTIPLY -> {
                if (bin.left().isZero() || bin.right().isZero()) yield IndexConst.ZERO;
                if (bin.right().isOne()) yield bin.left();
                if (bin.left().isOne()) yield bin.right();
                // x * 2^n -> x << n (strength reduction)
                if (bin.right().isConstant() && bin.right().constantValue() > 0) {
                    long mult = bin.right().constantValue();
                    if (isPowerOfTwo(mult)) {
                        yield new IndexBinary(
                                IndexBinaryOp.SHIFT_LEFT,
                                bin.left(),
                                IndexConst.of(Long.numberOfTrailingZeros(mult)));
                    }
                }
                if (bin.left().isConstant() && bin.right().isConstant()) {
                    yield IndexConst.of(bin.left().constantValue() * bin.right().constantValue());
                }
                yield bin;
            }
            case DIVIDE -> {
                if (bin.right().isOne()) yield bin.left();
                if (bin.left().isZero()) yield IndexConst.ZERO;
                // x / 2^n -> x >> n (strength reduction) for unsigned/positive
                if (bin.right().isConstant() && bin.right().constantValue() > 0) {
                    long div = bin.right().constantValue();
                    if (isPowerOfTwo(div)) {
                        yield new IndexBinary(
                                IndexBinaryOp.SHIFT_RIGHT,
                                bin.left(),
                                IndexConst.of(Long.numberOfTrailingZeros(div)));
                    }
                }
                if (bin.left().isConstant()
                        && bin.right().isConstant()
                        && bin.right().constantValue() != 0) {
                    yield IndexConst.of(bin.left().constantValue() / bin.right().constantValue());
                }
                yield bin;
            }
            case MODULO -> {
                if (bin.left().isZero()) yield IndexConst.ZERO;
                if (bin.right().isOne()) yield IndexConst.ZERO;
                // (x % a) % b -> x % a if a <= b (redundant modulo)
                if (bin.left() instanceof IndexBinary leftBin
                        && leftBin.op() == IndexBinaryOp.MODULO
                        && leftBin.right().isConstant()
                        && bin.right().isConstant()
                        && leftBin.right().constantValue() <= bin.right().constantValue()) {
                    yield leftBin;
                }
                // x % 2^n -> x & (2^n - 1) for positive constants
                if (bin.right().isConstant() && bin.right().constantValue() > 0) {
                    long mod = bin.right().constantValue();
                    if (isPowerOfTwo(mod)) {
                        yield new IndexBinary(
                                IndexBinaryOp.BITWISE_AND, bin.left(), IndexConst.of(mod - 1));
                    }
                }
                if (bin.left().isConstant()
                        && bin.right().isConstant()
                        && bin.right().constantValue() != 0) {
                    yield IndexConst.of(bin.left().constantValue() % bin.right().constantValue());
                }
                yield bin;
            }
            case BITWISE_AND -> bin;
            case SHIFT_LEFT -> bin;
            case SHIFT_RIGHT -> bin;
        };
    }

    private static boolean isPowerOfTwo(long n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}
