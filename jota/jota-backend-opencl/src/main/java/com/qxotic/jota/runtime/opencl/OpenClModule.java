package com.qxotic.jota.runtime.opencl;

import java.util.Objects;

final class OpenClModule implements AutoCloseable {

    private final long handle;
    private boolean closed;

    private OpenClModule(long handle) {
        this.handle = handle;
    }

    static OpenClModule load(byte[] source) {
        Objects.requireNonNull(source, "source");
        OpenClRuntime.requireAvailable();
        String flags = System.getProperty(OpenClRuntime.COMPILE_FLAGS_PROPERTY, "");
        long handle = OpenClRuntime.loadLibrary(source, flags);
        if (handle == 0L) {
            throw new IllegalStateException("OpenCL module load returned null handle");
        }
        return new OpenClModule(handle);
    }

    OpenClFunction function(String name) {
        Objects.requireNonNull(name, "name");
        if (closed) {
            throw new IllegalStateException("OpenCL module is already closed");
        }
        OpenClRuntime.requireAvailable();
        long functionHandle = OpenClRuntime.createPipeline(handle, name);
        if (functionHandle == 0L) {
            throw new IllegalStateException(
                    "OpenCL kernel pipeline creation returned null handle for '" + name + "'");
        }
        return new OpenClFunction(functionHandle);
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        OpenClRuntime.requireAvailable();
        OpenClRuntime.unloadLibrary(handle);
    }
}
