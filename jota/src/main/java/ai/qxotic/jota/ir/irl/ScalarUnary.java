package ai.qxotic.jota.ir.irl;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.irt.UnaryOperator;
import java.util.Objects;

/** Unary scalar operation. */
public record ScalarUnary(UnaryOperator op, ScalarExpr input) implements ScalarExpr {

    public ScalarUnary {
        Objects.requireNonNull(op, "op cannot be null");
        Objects.requireNonNull(input, "input cannot be null");
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }
}
