package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.util.Objects;

/** Binary scalar operation. */
public record ScalarBinary(BinaryOperator op, ScalarExpr left, ScalarExpr right)
        implements ScalarExpr {

    public ScalarBinary {
        Objects.requireNonNull(op, "op cannot be null");
        Objects.requireNonNull(left, "left cannot be null");
        Objects.requireNonNull(right, "right cannot be null");
    }

    @Override
    public DataType dataType() {
        // Logical operations return BOOL regardless of input types
        return switch (op) {
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_XOR -> DataType.BOOL;
            case LESS_THAN, EQUAL -> DataType.BOOL;
            default -> left.dataType();
        };
    }
}
