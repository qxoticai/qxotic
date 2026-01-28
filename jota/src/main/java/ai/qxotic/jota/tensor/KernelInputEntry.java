package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import java.util.Objects;

public record KernelInputEntry(KernelInputKind kind, Object value, DataType dataType) {
    public KernelInputEntry {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(value, "value");
    }
}
