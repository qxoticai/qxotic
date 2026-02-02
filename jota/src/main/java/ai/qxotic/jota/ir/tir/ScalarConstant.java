package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import java.util.Objects;

/**
 * Scalar constant tensor in IR-T. Represents a constant scalar value that can be broadcasted
 * without allocation.
 */
public record ScalarConstant(long rawBits, DataType dataType, Shape shape) implements TIRNode {

    public ScalarConstant {
        Objects.requireNonNull(dataType);
        Objects.requireNonNull(shape);
    }

    /** Creates a scalar constant with rank-0 shape (original scalar). */
    public static ScalarConstant of(long rawBits, DataType dataType) {
        return new ScalarConstant(rawBits, dataType, Shape.scalar());
    }

    /** Creates a scalar constant broadcasted to the given shape. */
    public static ScalarConstant broadcast(long rawBits, DataType dataType, Shape shape) {
        return new ScalarConstant(rawBits, dataType, shape);
    }

    @Override
    public Shape shape() {
        return shape;
    }
}
