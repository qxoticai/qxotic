package ai.qxotic.jota.ir.lir;

import java.util.Objects;

/** Update an accumulator with a new value using its combine operation. */
public record AccumulatorUpdate(String name, ScalarExpr value) implements LIRNode {

    public AccumulatorUpdate {
        Objects.requireNonNull(name, "name cannot be null");
        if (name.isEmpty()) {
            throw new IllegalArgumentException("name cannot be empty");
        }
        Objects.requireNonNull(value, "value cannot be null");
    }
}
