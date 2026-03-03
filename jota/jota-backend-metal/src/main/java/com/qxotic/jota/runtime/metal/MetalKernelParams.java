package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelArgs.Kind;

final class MetalKernelParams {

    private MetalKernelParams() {}

    static long pack(KernelArgs args) {
        MetalRuntime.requireAvailable();
        validateBuffers(args);
        return packNative(args);
    }

    static void release(long handle) {
        MetalRuntime.requireAvailable();
        releaseNative(handle);
    }

    private static void validateBuffers(KernelArgs args) {
        for (int i = 0; i < args.entries().size(); i++) {
            var entry = args.entry(i);
            if (entry.kind() != Kind.BUFFER) {
                continue;
            }
            MemoryView<?> view = (MemoryView<?>) entry.value();
            Object base = view.memory().base();
            if (!(base instanceof MetalDevicePtr)) {
                throw new UnsupportedOperationException(
                        "KernelArgs buffer base must be MetalDevicePtr at index "
                                + i
                                + ": baseClass="
                                + base.getClass().getName()
                                + ", device="
                                + view.memory().device());
            }
        }
    }

    private static native long packNative(KernelArgs args);

    private static native void releaseNative(long handle);
}
