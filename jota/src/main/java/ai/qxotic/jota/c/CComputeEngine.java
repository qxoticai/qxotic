package ai.qxotic.jota.c;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.tensor.ComputeBackend;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.KernelCache;
import java.util.Objects;

final class CComputeEngine implements ComputeEngine {

    private final CComputeBackend backend;
    private final KernelCache cache;

    CComputeEngine(CMemoryContext context) {
        Objects.requireNonNull(context, "context");
        this.backend = new CComputeBackend(context);
        this.cache = DiskKernelCache.defaultCache();
    }

    @Override
    public ComputeBackend backendFor(Device device) {
        if (!device.equals(backend.device())) {
            throw new IllegalArgumentException("No C backend for device: " + device);
        }
        return backend;
    }

    @Override
    public KernelCache cache() {
        return cache;
    }
}
