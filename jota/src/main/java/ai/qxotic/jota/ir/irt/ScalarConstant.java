package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import java.util.Objects;

/**
 * Scalar constant tensor in IR-T. Represents a constant scalar value that can be broadcasted
 * without allocation. The stride must be all zeros, allowing the same scalar value to be read from
 * any index.
 */
public record ScalarConstant(long rawBits, DataType dataType, Layout layout) implements IRTNode {

    public ScalarConstant {
        Objects.requireNonNull(dataType);
        Objects.requireNonNull(layout);
        if (!layout.stride().isAllZeros()) {
            throw new IllegalArgumentException(
                    "ScalarConstant stride must be all zeros, got: " + layout.stride());
        }
    }

    /** Creates a scalar constant with rank-0 shape (original scalar). */
    public static ScalarConstant of(long rawBits, DataType dtype) {
        return new ScalarConstant(rawBits, dtype, Layout.scalar());
    }

    /** Creates a scalar constant broadcasted to the given shape. */
    public static ScalarConstant broadcast(long rawBits, DataType dtype, Shape shape) {
        return new ScalarConstant(rawBits, dtype, Layout.of(shape, Stride.zeros(shape)));
    }
}
