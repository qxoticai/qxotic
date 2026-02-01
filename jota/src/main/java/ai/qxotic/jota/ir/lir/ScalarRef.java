package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import java.util.Objects;

/**
 * References a scalar value bound by a {@link ScalarLet}.
 *
 * <p>This node is used after hoisting to reference a precomputed loop-invariant value.
 *
 * @param name the name of the bound scalar (must match a ScalarLet in an enclosing scope)
 * @param dataType the data type of the referenced value
 */
public record ScalarRef(String name, DataType dataType) implements ScalarExpr {

    public ScalarRef {
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(dataType, "dataType cannot be null");
    }

    @Override
    public DataType dataType() {
        return dataType;
    }
}
