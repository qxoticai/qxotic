package com.qxotic.jota.runtime.metal;

import java.util.Objects;

final class MetalModule implements AutoCloseable {

    private final long handle;

    private MetalModule(long handle) {
        this.handle = handle;
    }

    static MetalModule load(byte[] metallib) {
        Objects.requireNonNull(metallib, "metallib");
        MetalRuntime.requireAvailable();
        return new MetalModule(MetalRuntime.loadLibrary(metallib));
    }

    MetalFunction function(String name) {
        Objects.requireNonNull(name, "name");
        MetalRuntime.requireAvailable();
        return new MetalFunction(MetalRuntime.createPipeline(handle, name));
    }

    @Override
    public void close() {
        MetalRuntime.requireAvailable();
        MetalRuntime.unloadLibrary(handle);
    }
}
