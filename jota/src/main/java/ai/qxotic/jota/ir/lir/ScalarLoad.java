package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import java.util.Objects;

/** Load a scalar value from a buffer at the given offset. */
public record ScalarLoad(BufferRef buffer, IndexExpr offset) implements ScalarExpr {

    public ScalarLoad {
        Objects.requireNonNull(buffer, "buffer cannot be null");
        Objects.requireNonNull(offset, "offset cannot be null");
    }

    @Override
    public DataType dataType() {
        return buffer.dataType();
    }
}
