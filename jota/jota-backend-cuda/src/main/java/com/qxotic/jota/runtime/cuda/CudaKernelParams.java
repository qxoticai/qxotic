package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelArgs.Kind;

final class CudaKernelParams {

    private CudaKernelParams() {}

    static long pack(KernelArgs args) {
        CudaRuntime.requireAvailable();
        validateBuffers(args);
        return packNative(args);
    }

    private static void validateBuffers(KernelArgs args) {
        for (int i = 0; i < args.entries().size(); i++) {
            var entry = args.entry(i);
            if (entry.kind() != Kind.BUFFER) {
                continue;
            }
            MemoryView<?> view = (MemoryView<?>) entry.value();
            Object base = view.memory().base();
            if (!(base instanceof CudaDevicePtr) && !(base instanceof Number)) {
                throw new UnsupportedOperationException(
                        "KernelArgs buffer base is not device-addressable for CUDA at index "
                                + i
                                + ": baseClass="
                                + base.getClass().getName()
                                + ", device="
                                + view.memory().device());
            }
        }
    }

    static long streamHandle(ExecutionStream stream) {
        Object handle = stream.handle();
        if (handle instanceof Long value) {
            return value;
        }
        return 0L;
    }

    static void release(long handle) {
        CudaRuntime.requireAvailable();
        releaseNative(handle);
    }

    private static native long packNative(KernelArgs args);

    private static native void releaseNative(long handle);
}
