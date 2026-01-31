package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import java.util.Objects;

/**
 * Represents a scalar input parameter passed by value at runtime.
 *
 * <p>Unlike {@link ScalarLiteral} which represents compile-time constants that are inlined,
 * ScalarInput represents a scalar value that is passed as a kernel parameter at runtime. The value
 * is not known at IR construction time.
 *
 * <p>Unlike {@link ScalarLoad} which loads from a buffer in memory, ScalarInput directly references
 * a scalar value passed by value. No memory access is performed.
 */
public record ScalarInput(int id, DataType dataType) implements ScalarExpr, LIRInput {

    public ScalarInput {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        if (id < 0) {
            throw new IllegalArgumentException("id must be non-negative, got: " + id);
        }
    }
}
