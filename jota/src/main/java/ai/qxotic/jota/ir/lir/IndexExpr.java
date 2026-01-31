package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;

/** Sealed interface for index expressions in IR-L. Indices are always 64-bit longs. */
public sealed interface IndexExpr extends LIRNode permits IndexVar, IndexConst, IndexBinary {

    @Override
    default DataType dataType() {
        return DataType.I64;
    }

    /**
     * Simplifies this expression using algebraic rules. Returns the same instance if no
     * simplification possible (structural sharing).
     */
    default IndexExpr simplify() {
        return this;
    }

    /** Returns true if this is a constant with value 0. */
    default boolean isZero() {
        return false;
    }

    /** Returns true if this is a constant with value 1. */
    default boolean isOne() {
        return false;
    }

    /** Returns true if this is a constant expression. */
    default boolean isConstant() {
        return false;
    }

    /** Returns the constant value if this is a constant, otherwise throws. */
    default long constantValue() {
        throw new IllegalStateException("Not a constant: " + this);
    }
}
