package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import java.util.Objects;

/** Unary scalar operation. */
public record ScalarUnary(UnaryOperator op, ScalarExpr input) implements ScalarExpr {

    public ScalarUnary {
        Objects.requireNonNull(op, "op cannot be null");
        Objects.requireNonNull(input, "input cannot be null");
    }

    @Override
    public DataType dataType() {
        // Logical operations return BOOL regardless of input type
        return switch (op) {
            case LOGICAL_NOT -> DataType.BOOL;
            default -> input.dataType();
        };
    }
}
