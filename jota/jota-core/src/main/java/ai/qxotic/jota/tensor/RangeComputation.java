package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Lazy range computation that materializes on demand. */
record RangeComputation(long count, Device device) implements LazyComputation {

    RangeComputation {
        if (count < 0) {
            throw new IllegalArgumentException("count must be non-negative, got: " + count);
        }
        Objects.requireNonNull(device, "device");
    }

    @Override
    public List<Tensor> inputs() {
        return List.of();
    }

    @Override
    public Map<String, Object> attributes() {
        return Map.of("count", count, "dataType", DataType.I64);
    }

    @Override
    public MemoryView<?> execute() {
        MemoryDomain<?> memoryDomain = Environment.current().runtimeFor(device).memoryDomain();
        return MemoryHelpers.arange(memoryDomain, DataType.I64, count);
    }
}
