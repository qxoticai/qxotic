package ai.qxotic.jota.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.tensor.ComputeBackend;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.KernelCache;
import java.util.Objects;

public final class HipComputeEngine implements ComputeEngine {

    private final HipComputeBackend backend;
    private final KernelCache cache;

    public HipComputeEngine(HipMemoryContext context) {
        Objects.requireNonNull(context, "context");
        this.backend = new HipComputeBackend(context.device());
        this.cache = DiskKernelCache.defaultCache();
    }

    @Override
    public ComputeBackend backendFor(Device device) {
        if (!device.equals(backend.device())) {
            throw new IllegalArgumentException("No HIP backend for device: " + device);
        }
        return backend;
    }

    @Override
    public KernelCache cache() {
        return cache;
    }
}
