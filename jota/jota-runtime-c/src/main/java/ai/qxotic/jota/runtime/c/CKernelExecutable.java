package ai.qxotic.jota.runtime.c;

import ai.qxotic.jota.tensor.ExecutionStream;
import ai.qxotic.jota.tensor.KernelArgs;
import ai.qxotic.jota.tensor.KernelExecutable;
import ai.qxotic.jota.tensor.LaunchConfig;
import java.util.List;

final class CKernelExecutable implements KernelExecutable {

    private final long functionPtr;

    CKernelExecutable(long functionPtr) {
        this.functionPtr = functionPtr;
    }

    @Override
    public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
        if (functionPtr == 0L) {
            throw new IllegalStateException("C kernel function is null");
        }
        long[] bufferPtrs = argsToBuffers(args);
        long[] scalarBits = argsToScalars(args);
        long scratchPtr = extractScratchPtr(args);
        CNative.invokeKernel(functionPtr, bufferPtrs, scalarBits, scratchPtr);
    }

    @Override
    public void close() {}

    private static long[] argsToBuffers(KernelArgs args) {
        return args.entries().stream()
                .filter(entry -> entry.kind() == KernelArgs.Kind.BUFFER)
                .mapToLong(entry -> CKernelExecutable.bufferPointer(entry))
                .toArray();
    }

    private static long[] argsToScalars(KernelArgs args) {
        return args.entries().stream()
                .filter(entry -> entry.kind() == KernelArgs.Kind.SCALAR)
                .mapToLong(entry -> (long) entry.value())
                .toArray();
    }

    private static long extractScratchPtr(KernelArgs args) {
        List<KernelArgs.Entry> entries = args.entries();
        if (entries.isEmpty()) {
            return 0L;
        }
        KernelArgs.Entry last = entries.getLast();
        if (last.kind() == KernelArgs.Kind.METADATA && last.value() instanceof Number value) {
            return value.longValue();
        }
        return 0L;
    }

    private static long bufferPointer(KernelArgs.Entry entry) {
        Object value = entry.value();
        if (!(value instanceof ai.qxotic.jota.memory.MemoryView<?> view)) {
            throw new IllegalArgumentException("Expected MemoryView, got " + value);
        }
        Object base = view.memory().base();
        if (!(base instanceof java.lang.foreign.MemorySegment segment)) {
            throw new IllegalArgumentException("Expected MemorySegment base for C backend");
        }
        return segment.address() + view.byteOffset();
    }
}
