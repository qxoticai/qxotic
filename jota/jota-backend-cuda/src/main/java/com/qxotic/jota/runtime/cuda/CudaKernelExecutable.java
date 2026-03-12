package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.LaunchConfig;
import java.util.Objects;

public final class CudaKernelExecutable implements KernelExecutable {

    private final CudaFunction function;

    public CudaKernelExecutable(CudaFunction function) {
        this.function = Objects.requireNonNull(function, "function");
    }

    @Override
    public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(stream, "stream");
        CudaRuntime.requireAvailable();
        long argsHandle = CudaKernelParams.pack(args);
        CudaRuntime.launchKernel(
                function.handle(),
                config.gridDimX(),
                config.gridDimY(),
                config.gridDimZ(),
                config.blockDimX(),
                config.blockDimY(),
                config.blockDimZ(),
                config.sharedMemBytes(),
                CudaKernelParams.streamHandle(stream),
                argsHandle);
        CudaKernelParams.release(argsHandle);
    }
}
