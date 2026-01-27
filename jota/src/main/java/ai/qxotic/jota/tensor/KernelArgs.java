package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.memory.MemoryView;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class KernelArgs {

    public enum Kind {
        BUFFER,
        SCALAR,
        METADATA
    }

    public record Entry(Kind kind, Object value, DataType dataType) {}

    private final List<Entry> entries = new ArrayList<>();

    public KernelArgs addBuffer(MemoryView<?> view) {
        Objects.requireNonNull(view, "view");
        entries.add(new Entry(Kind.BUFFER, view, view.dataType()));
        return this;
    }

    public KernelArgs addScalar(Number value, DataType dataType) {
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(dataType, "dataType");
        entries.add(new Entry(Kind.SCALAR, value, dataType));
        return this;
    }

    public KernelArgs addMetadata(Object metadata) {
        Objects.requireNonNull(metadata, "metadata");
        entries.add(new Entry(Kind.METADATA, metadata, null));
        return this;
    }

    public List<Entry> entries() {
        return List.copyOf(entries);
    }
}
