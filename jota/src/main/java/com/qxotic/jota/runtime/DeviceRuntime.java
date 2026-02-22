package com.qxotic.jota.runtime;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.tensor.BinaryOp;
import com.qxotic.jota.tensor.ComputeEngine;
import com.qxotic.jota.tensor.ExecutionStream;
import com.qxotic.jota.tensor.KernelArgs;
import com.qxotic.jota.tensor.KernelCacheKey;
import com.qxotic.jota.tensor.KernelExecutable;
import com.qxotic.jota.tensor.KernelProgram;
import com.qxotic.jota.tensor.LaunchConfig;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.UnaryOp;
import java.util.Objects;
import java.util.Optional;

public interface DeviceRuntime {
    Device device();

    MemoryDomain<?> memoryDomain();

    ComputeEngine computeEngine();

    Optional<KernelService> kernelService();

    default Optional<EagerKernels> eagerKernels() {
        return Optional.empty();
    }

    default boolean supportsKernels() {
        return kernelService().isPresent();
    }

    default KernelCacheKey keyFor(KernelProgram program) {
        return KernelProgramHasher.keyFor(program);
    }

    default KernelExecutable registerKernel(KernelProgram program) {
        Objects.requireNonNull(program, "program");
        return registerKernel(program, keyFor(program));
    }

    default KernelExecutable registerKernel(KernelProgram program, KernelCacheKey key) {
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(key, "key");
        KernelService kernels = requireKernelService();
        return kernels.register(program, key);
    }

    default KernelExecutable registerKernel(String name, KernelProgram program) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(program, "program");
        return registerKernel(name, program, keyFor(program));
    }

    default KernelExecutable registerKernel(
            String name, KernelProgram program, KernelCacheKey key) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(key, "key");
        KernelService kernels = requireKernelService();
        return kernels.register(name, program, key);
    }

    default Optional<KernelProgram> loadRegisteredKernel(KernelCacheKey key) {
        Objects.requireNonNull(key, "key");
        KernelService kernels = requireKernelService();
        return kernels.loadRegisteredKernel(key);
    }

    default Optional<KernelProgram> loadRegisteredKernel(String name) {
        Objects.requireNonNull(name, "name");
        KernelService kernels = requireKernelService();
        return kernels.loadRegisteredKernel(name);
    }

    default Optional<KernelExecutable> loadRegisteredExecutable(String name) {
        Objects.requireNonNull(name, "name");
        KernelService kernels = requireKernelService();
        return kernels.loadRegisteredExecutable(name);
    }

    default Optional<KernelExecutable> loadRegisteredBinaryExecutable(String name) {
        Objects.requireNonNull(name, "name");
        KernelService kernels = requireKernelService();
        return kernels.loadRegisteredBinaryExecutable(name);
    }

    default void bindKernelName(String name, KernelCacheKey key) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(key, "key");
        KernelService kernels = requireKernelService();
        kernels.bindKernelName(name, key);
    }

    default void launchKernel(String name, Object... args) {
        launchKernel(name, LaunchConfig.auto(), args);
    }

    default void launchKernel(String name, LaunchConfig config, Object... args) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(config, "config");
        KernelExecutable exec =
                loadRegisteredExecutable(name)
                        .orElseThrow(
                                () ->
                                        new IllegalArgumentException(
                                                "No kernel registered with name: " + name));
        KernelArgs kernelArgs = KernelArgs.fromVarargs(args);
        exec.launch(config, kernelArgs, new ExecutionStream(device(), null, true));
    }

    private KernelService requireKernelService() {
        Optional<KernelService> service = kernelService();
        if (service.isEmpty()) {
            throw new UnsupportedOperationException(
                    "Runtime does not support kernels: " + device());
        }
        return service.get();
    }

    enum ReductionOp {
        SUM,
        PRODUCT,
        MIN,
        MAX
    }

    interface EagerKernels {

        Tensor unary(UnaryOp op, Tensor x);

        Tensor binary(BinaryOp op, Tensor a, Tensor b);

        Tensor compare(BinaryOp op, Tensor a, Tensor b);

        Tensor logical(BinaryOp op, Tensor a, Tensor b);

        Tensor cast(Tensor x, DataType targetType);

        Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue);

        Tensor reduce(
                ReductionOp op,
                Tensor x,
                DataType accumulatorType,
                boolean keepDims,
                int axis,
                int... axes);

        Tensor matmul(Tensor a, Tensor b);

        Tensor batchedMatmul(Tensor a, Tensor b);

        Tensor gather(Tensor input, Tensor indices, int axis);
    }
}
