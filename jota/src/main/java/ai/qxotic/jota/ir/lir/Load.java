package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import java.util.Objects;

/** Load a scalar value from a buffer at the given byte offset. */
public record Load(BufferRef buffer, IndexExpr offset) implements LIRNode {

    public Load {
        Objects.requireNonNull(buffer, "buffer cannot be null");
        Objects.requireNonNull(offset, "offset cannot be null");
    }

    @Override
    public DataType dataType() {
        return buffer.dataType();
    }

    /** Creates a ScalarLoad expression equivalent to this Load. */
    public ScalarLoad toScalarLoad() {
        return new ScalarLoad(buffer, offset);
    }
}
