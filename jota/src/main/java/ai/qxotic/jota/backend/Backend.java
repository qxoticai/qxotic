package ai.qxotic.jota.backend;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelExecutable;
import ai.qxotic.jota.tensor.KernelProgram;
import java.util.Objects;
import java.util.Optional;

public interface Backend {
    Device device();

    MemoryContext<?> memoryContext();

    ComputeEngine computeEngine();

    Optional<KernelService> kernels();

    default boolean supportsKernels() {
        return kernels().isPresent();
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
        KernelService kernels = requireKernels();
        return kernels.register(program, key);
    }

    default Optional<KernelProgram> loadRegisteredKernel(KernelCacheKey key) {
        Objects.requireNonNull(key, "key");
        KernelService kernels = requireKernels();
        return kernels.loadRegisteredKernel(key);
    }

    private KernelService requireKernels() {
        Optional<KernelService> service = kernels();
        if (service.isEmpty()) {
            throw new UnsupportedOperationException(
                    "Backend does not support kernels: " + device());
        }
        return service.get();
    }
}
