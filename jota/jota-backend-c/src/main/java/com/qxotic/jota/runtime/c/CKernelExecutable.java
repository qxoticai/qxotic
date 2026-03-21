package com.qxotic.jota.runtime.c;

import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.LaunchConfig;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
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
        PreparedArgs prepared = prepareArgs(args);
        CNative.invokeKernel(
                functionPtr,
                prepared.bufferPointers(),
                prepared.scalarBits(),
                prepared.scratchPointer());
    }

    @Override
    public void close() {}

    private static PreparedArgs prepareArgs(KernelArgs args) {
        List<KernelArgs.Entry> entries = args.entries();
        if (entries.isEmpty()) {
            return new PreparedArgs(new long[0], new long[0], 0L);
        }
        long[] bufferPointers = new long[entries.size()];
        long[] scalarBits = new long[entries.size()];
        int bufferCount = 0;
        int scalarCount = 0;
        long scratchPointer = 0L;
        for (KernelArgs.Entry entry : entries) {
            if (entry.kind() == KernelArgs.Kind.BUFFER) {
                bufferPointers[bufferCount++] = bufferPointer(entry);
                continue;
            }
            if (entry.kind() == KernelArgs.Kind.SCALAR) {
                if (!(entry.value() instanceof Number value)) {
                    throw new IllegalArgumentException(
                            "Expected numeric scalar, got " + entry.value());
                }
                scalarBits[scalarCount++] = value.longValue();
                continue;
            }
            if (entry.kind() == KernelArgs.Kind.METADATA && entry.value() instanceof Number value) {
                scratchPointer = value.longValue();
            }
        }
        return new PreparedArgs(
                Arrays.copyOf(bufferPointers, bufferCount),
                Arrays.copyOf(scalarBits, scalarCount),
                scratchPointer);
    }

    private record PreparedArgs(long[] bufferPointers, long[] scalarBits, long scratchPointer) {}

    private static long bufferPointer(KernelArgs.Entry entry) {
        Object value = entry.value();
        if (!(value instanceof MemoryView<?> view)) {
            throw new IllegalArgumentException("Expected MemoryView, got " + value);
        }
        Object base = view.memory().base();
        if (!(base instanceof MemorySegment segment)) {
            throw new IllegalArgumentException("Expected MemorySegment base for C backend");
        }
        return segment.address() + view.byteOffset();
    }
}
