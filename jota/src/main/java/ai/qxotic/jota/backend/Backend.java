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

    KernelPipeline kernelPipeline();

    default boolean supportsKernels() {
        return kernelPipeline() != null;
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
        KernelPipeline pipeline = requirePipeline();
        pipeline.programStore().store(program, key);
        KernelExecutable exec = pipeline.backend().getOrCompile(program, key);
        return exec;
    }

    default Optional<KernelProgram> loadRegisteredKernel(KernelCacheKey key) {
        Objects.requireNonNull(key, "key");
        KernelPipeline pipeline = requirePipeline();
        return pipeline.programStore().load(key);
    }

    private KernelPipeline requirePipeline() {
        KernelPipeline pipeline = kernelPipeline();
        if (pipeline == null) {
            throw new UnsupportedOperationException("Backend does not support kernels: " + device());
        }
        return pipeline;
    }
}
