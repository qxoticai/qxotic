package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryContext;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

public final class JavaComputeEngine implements ComputeEngine {

    private final KernelCache cache;
    private final ComputeBackend backend;

    public JavaComputeEngine(MemoryContext<MemorySegment> context) {
        this(context, DiskKernelCache.defaultCache());
    }

    public JavaComputeEngine(MemoryContext<MemorySegment> context, KernelCache cache) {
        Objects.requireNonNull(context, "context");
        this.cache = Objects.requireNonNull(cache, "cache");
        this.backend = new JavaComputeBackend(context, cache);
    }

    @Override
    public ComputeBackend backendFor(Device device) {
        if (!backend.device().equals(device)) {
            throw new IllegalArgumentException("No backend registered for device: " + device);
        }
        return backend;
    }

    @Override
    public KernelCache cache() {
        return cache;
    }
}
