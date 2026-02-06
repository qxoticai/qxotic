package ai.qxotic.jota.tensor;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryDomain;
import java.lang.foreign.MemorySegment;

/**
 * Interface for JIT-compiled kernels.
 *
 * <p>Kernels receive inputs and outputs via {@link KernelArgs}, and optionally a scratch buffer for
 * intermediate storage.
 */
public interface JitKernel {

    /**
     * Executes the kernel without scratch memory.
     *
     * <p>This method must be implemented by all kernels. Kernels that require scratch memory should
     * throw an exception from this method and override {@link #execute(MemoryDomain, KernelArgs,
     * Memory)} instead.
     *
     * @param domain the memory domain for device operations
     * @param args kernel arguments (inputs, outputs, scalars)
     */
    void execute(MemoryDomain<MemorySegment> domain, KernelArgs args);

    /**
     * Executes the kernel with optional scratch memory.
     *
     * <p>This is the preferred method for executors to call. The default implementation delegates
     * to {@link #execute(MemoryDomain, KernelArgs)}.
     *
     * @param memoryDomain the memory domain for device operations
     * @param args kernel arguments (inputs, outputs, scalars)
     * @param scratch scratch buffer for intermediate storage, or null if not needed
     */
    default void execute(
            MemoryDomain<MemorySegment> memoryDomain,
            KernelArgs args,
            Memory<MemorySegment> scratch) {
        execute(memoryDomain, args);
    }

    /**
     * Returns the scratch buffer size required by this kernel.
     *
     * @return required scratch size in bytes, or 0 if no scratch is needed
     */
    default long scratchByteSize() {
        return 0L;
    }
}
