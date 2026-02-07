package ai.qxotic.jota.runtime.hip;

import ai.qxotic.jota.tensor.ExecutionStream;
import ai.qxotic.jota.tensor.KernelArgs;
import ai.qxotic.jota.tensor.KernelExecutable;
import ai.qxotic.jota.tensor.LaunchConfig;
import java.util.Objects;

public final class HipKernelExecutable implements KernelExecutable {

    private final HipFunction function;

    public HipKernelExecutable(HipFunction function) {
        this.function = Objects.requireNonNull(function, "function");
    }

    @Override
    public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(stream, "stream");
        HipRuntime.requireAvailable();
        long argsHandle = HipKernelParams.pack(args);
        HipRuntime.launchKernel(
                function.handle(),
                config.gridDimX(),
                config.gridDimY(),
                config.gridDimZ(),
                config.blockDimX(),
                config.blockDimY(),
                config.blockDimZ(),
                config.sharedMemBytes(),
                HipKernelParams.streamHandle(stream),
                argsHandle);
        HipKernelParams.release(argsHandle);
    }
}
