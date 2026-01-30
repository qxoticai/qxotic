package ai.qxotic.jota.ir.irl;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.irt.BinaryOperator;
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
        return left.dataType();
    }
}
