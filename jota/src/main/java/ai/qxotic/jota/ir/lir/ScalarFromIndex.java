package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import java.util.Objects;

/**
 * Converts an IndexExpr to a ScalarExpr. Used for operations like iota/arange where the loop index
 * becomes the data value.
 */
public record ScalarFromIndex(IndexExpr index) implements ScalarExpr {

    public ScalarFromIndex {
        Objects.requireNonNull(index, "index cannot be null");
    }

    @Override
    public DataType dataType() {
        // Indices are always I64
        return DataType.I64;
    }
}
