package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.*;
import java.util.Objects;

/**
 * Contiguous operation in IR-T. Represents a semantic requirement that the tensor should have
 * contiguous row-major layout. IR-L will decide whether to emit a no-op (if already contiguous) or
 * allocate+copy.
 */
public record Contiguous(IRTNode input) implements IRTNode {

    public Contiguous {
        Objects.requireNonNull(input);
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }

    @Override
    public Layout layout() {
        Shape shape = input.layout().shape();
        return Layout.of(shape, Stride.rowMajor(shape));
    }
}
