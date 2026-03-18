package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryDomain;
import java.util.Objects;
import java.util.Optional;

public interface DeviceRuntime {
    Device device();

    MemoryDomain<?> memoryDomain();

    ComputeEngine computeEngine();

    Optional<KernelService> kernelService();

    default boolean supportsKernels() {
        return kernelService().isPresent();
    }

    default DeviceProperties properties() {
        return DeviceProperties.EMPTY;
    }

    default DeviceCapabilities capabilities() {
        return DeviceCapabilities.EMPTY;
    }

    /**
     * Indicates whether this runtime can safely back the {@link Device#NATIVE} alias.
     *
     * <p>By default runtimes are not eligible. CPU runtimes that operate on {@link
     * java.lang.foreign.MemorySegment} and are intended as process-default runtimes should override
     * this to {@code true}.
     */
    default boolean supportsNativeRuntimeAlias() {
        return false;
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
}
