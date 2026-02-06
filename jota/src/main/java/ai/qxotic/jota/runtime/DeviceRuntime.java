package ai.qxotic.jota.runtime;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelExecutable;
import ai.qxotic.jota.tensor.KernelProgram;
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

    default Optional<KernelProgram> loadRegisteredKernel(KernelCacheKey key) {
        Objects.requireNonNull(key, "key");
        KernelService kernels = requireKernelService();
        return kernels.loadRegisteredKernel(key);
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
