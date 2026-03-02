package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.LaunchConfig;
import java.util.Objects;

final class MetalKernelExecutable implements KernelExecutable {

    private final MetalFunction function;

    MetalKernelExecutable(MetalFunction function) {
        this.function = Objects.requireNonNull(function, "function");
    }

    @Override
    public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(stream, "stream");
        MetalRuntime.requireAvailable();
        long argsHandle = MetalKernelParams.pack(args);
        try {
            MetalRuntime.launchKernel(
                    function.handle(),
                    config.gridDimX(),
                    config.gridDimY(),
                    config.gridDimZ(),
                    config.blockDimX(),
                    config.blockDimY(),
                    config.blockDimZ(),
                    argsHandle);
        } finally {
            MetalKernelParams.release(argsHandle);
        }
    }

    @Override
    public void close() {
        MetalRuntime.requireAvailable();
        MetalRuntime.releasePipeline(function.handle());
    }
}
