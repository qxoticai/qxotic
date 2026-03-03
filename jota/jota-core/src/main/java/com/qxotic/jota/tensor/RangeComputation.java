package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Lazy range computation that materializes on demand. */
record RangeComputation(long count, Device device) implements LazyComputation {

    public RangeComputation {
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
        MemoryDomain<?> memoryDomain = Environment.current().memoryDomainFor(device);
        return MemoryHelpers.arange(memoryDomain, DataType.I64, count);
    }
}
