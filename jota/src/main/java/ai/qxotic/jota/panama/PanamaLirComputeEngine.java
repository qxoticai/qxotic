package ai.qxotic.jota.panama;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.tensor.ComputeBackend;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.KernelCache;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

public final class PanamaLirComputeEngine implements ComputeEngine {

    private final ComputeBackend backend;
    private final KernelCache cache;

    public PanamaLirComputeEngine(MemoryContext<MemorySegment> context, DiskKernelCache cache) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(cache, "cache");
        this.backend = new PanamaLirComputeBackend(context, cache);
        this.cache = cache;
    }

    @Override
    public ComputeBackend backendFor(Device device) {
        if (!device.equals(backend.device())) {
            throw new IllegalArgumentException(
                    "Unsupported device for Panama compute engine: " + device);
        }
        return backend;
    }

    @Override
    public KernelCache cache() {
        return cache;
    }
}
