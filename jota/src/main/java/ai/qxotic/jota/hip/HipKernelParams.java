package ai.qxotic.jota.hip;

import ai.qxotic.jota.tensor.ExecutionStream;
import ai.qxotic.jota.tensor.KernelArgs;

final class HipKernelParams {

    private HipKernelParams() {}

    static long pack(KernelArgs args) {
        HipRuntime.requireAvailable();
        return packNative(args);
    }

    static long streamHandle(ExecutionStream stream) {
        Object handle = stream.handle();
        if (handle instanceof Long value) {
            return value;
        }
        return 0L;
    }

    static void release(long handle) {
        HipRuntime.requireAvailable();
        releaseNative(handle);
    }

    private static native long packNative(KernelArgs args);

    private static native void releaseNative(long handle);
}
