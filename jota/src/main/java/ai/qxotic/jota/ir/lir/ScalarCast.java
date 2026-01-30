package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import java.util.Objects;

/** Type cast for scalar expressions. */
public record ScalarCast(ScalarExpr input, DataType targetType) implements ScalarExpr {

    public ScalarCast {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(targetType, "targetType cannot be null");
    }

    @Override
    public DataType dataType() {
        return targetType;
    }
}
