package ai.qxotic.jota.ir.irl;

import ai.qxotic.jota.DataType;
import java.util.Objects;

/** Read the current value of an accumulator. */
public record AccumulatorRead(String name, DataType dataType) implements IRLNode {

    public AccumulatorRead {
        Objects.requireNonNull(name, "name cannot be null");
        if (name.isEmpty()) {
            throw new IllegalArgumentException("name cannot be empty");
        }
        Objects.requireNonNull(dataType, "dataType cannot be null");
    }
}
