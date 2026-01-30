package ai.qxotic.jota.ir.lir;

import java.util.Objects;

/** Store a scalar value to a buffer at the given byte offset. */
public record Store(BufferRef buffer, IndexExpr offset, ScalarExpr value) implements LIRNode {

    public Store {
        Objects.requireNonNull(buffer, "buffer cannot be null");
        Objects.requireNonNull(offset, "offset cannot be null");
        Objects.requireNonNull(value, "value cannot be null");
    }
}
