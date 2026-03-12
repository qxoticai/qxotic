package com.qxotic.jota.runtime.cuda;

import java.util.Objects;

public final class CudaModule implements AutoCloseable {

    private final long handle;

    private CudaModule(long handle) {
        this.handle = handle;
    }

    public static CudaModule load(byte[] ptx) {
        Objects.requireNonNull(ptx, "ptx");
        CudaRuntime.requireAvailable();
        return new CudaModule(CudaRuntime.loadModule(ptx));
    }

    public CudaFunction function(String name) {
        Objects.requireNonNull(name, "name");
        CudaRuntime.requireAvailable();
        return new CudaFunction(CudaRuntime.getFunction(handle, name));
    }

    @Override
    public void close() {
        CudaRuntime.requireAvailable();
        CudaRuntime.unloadModule(handle);
    }
}
