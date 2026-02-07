package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import java.util.Objects;

/** Scalar input passed by value (broadcasted to the given shape). */
public record ScalarInput(int id, DataType dataType, Shape shape) implements TIRNode {

    public ScalarInput {
        if (id < 0) {
            throw new IllegalArgumentException("ScalarInput id must be non-negative, got: " + id);
        }
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(shape, "shape");
    }
}
