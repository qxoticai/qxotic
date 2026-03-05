package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.LaunchConfig;
import java.util.Objects;

final class OpenClKernelExecutable implements KernelExecutable {

    private final OpenClFunction function;
    private boolean closed;

    OpenClKernelExecutable(OpenClFunction function) {
        this.function = Objects.requireNonNull(function, "function");
    }

    @Override
    public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(stream, "stream");
        if (closed) {
            throw new IllegalStateException("OpenCL kernel executable is already closed");
        }
        OpenClRuntime.requireAvailable();
        long argsHandle = OpenClKernelParams.pack(args);
        try {
            OpenClRuntime.launchKernel(
                    function.handle(),
                    config.gridDimX(),
                    config.gridDimY(),
                    config.gridDimZ(),
                    config.blockDimX(),
                    config.blockDimY(),
                    config.blockDimZ(),
                    argsHandle);
        } finally {
            OpenClKernelParams.release(argsHandle);
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        OpenClRuntime.requireAvailable();
        OpenClRuntime.releasePipeline(function.handle());
    }
}
