package ai.qxotic.jota.ir.irl;

import ai.qxotic.jota.DataType;
import java.util.Objects;

/** Ternary scalar operation (select/where). */
public record ScalarTernary(ScalarExpr condition, ScalarExpr trueValue, ScalarExpr falseValue)
        implements ScalarExpr {

    public ScalarTernary {
        Objects.requireNonNull(condition, "condition cannot be null");
        Objects.requireNonNull(trueValue, "trueValue cannot be null");
        Objects.requireNonNull(falseValue, "falseValue cannot be null");
    }

    @Override
    public DataType dataType() {
        return trueValue.dataType();
    }
}
