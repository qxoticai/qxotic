package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelArgs.Kind;

final class OpenClKernelParams {

    private OpenClKernelParams() {}

    static long pack(KernelArgs args) {
        OpenClRuntime.requireAvailable();
        validateBuffers(args);
        return packNative(args);
    }

    static void release(long handle) {
        OpenClRuntime.requireAvailable();
        releaseNative(handle);
    }

    private static void validateBuffers(KernelArgs args) {
        for (int i = 0; i < args.entries().size(); i++) {
            var entry = args.entry(i);
            if (entry.kind() != Kind.BUFFER) {
                continue;
            }
            if (!(entry.value() instanceof MemoryView<?> view)) {
                throw new UnsupportedOperationException(
                        "KernelArgs buffer value must be MemoryView at index "
                                + i
                                + ": valueClass="
                                + (entry.value() == null
                                        ? "<null>"
                                        : entry.value().getClass().getName()));
            }
            Object base = view.memory().base();
            if (!(base instanceof OpenClDevicePtr)) {
                throw new UnsupportedOperationException(
                        "KernelArgs buffer base must be OpenClDevicePtr at index "
                                + i
                                + ": baseClass="
                                + (base == null ? "<null>" : base.getClass().getName())
                                + ", device="
                                + view.memory().device());
            }
        }
    }

    private static native long packNative(KernelArgs args);

    private static native void releaseNative(long handle);
}
