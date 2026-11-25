package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Transpose;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.triangular_solve}. Generated from the op table; see {@link Op}.
 */
public record TriangularSolveOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.triangular_solve";

    public TriangularSolveOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value a() {
        return operation.operand(0);
    }

    public Value b() {
        return operation.operand(1);
    }

    /** Attribute {@code left_side}. */
    public boolean leftSide() {
        return OpSupport.bool(operation, "left_side");
    }

    /** Attribute {@code lower}. */
    public boolean lower() {
        return OpSupport.bool(operation, "lower");
    }

    /** Attribute {@code unit_diagonal}. */
    public boolean unitDiagonal() {
        return OpSupport.bool(operation, "unit_diagonal");
    }

    /** Attribute {@code transpose_a}. */
    public Transpose transposeA() {
        return Transpose.from(OpSupport.required(operation, "transpose_a"));
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitTriangularSolve(this);
    }
}
