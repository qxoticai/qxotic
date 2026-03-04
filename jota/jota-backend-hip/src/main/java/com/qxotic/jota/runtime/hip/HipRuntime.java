package com.qxotic.jota.runtime.hip;

import java.util.concurrent.atomic.AtomicReference;

public final class HipRuntime {

    private static final String LIB_NAME = "jota_hip";
    private static final AtomicReference<Boolean> AVAILABLE = new AtomicReference<>();

    private HipRuntime() {}

    public static boolean isAvailable() {
        Boolean available = AVAILABLE.get();
        if (available != null) {
            return available;
        }
        try {
            System.loadLibrary(LIB_NAME);
            AVAILABLE.set(true);
        } catch (UnsatisfiedLinkError e) {
            AVAILABLE.set(false);
        }
        return AVAILABLE.get();
    }

    public static void requireAvailable() {
        if (!isAvailable()) {
            throw new IllegalStateException("HIP runtime not available (" + LIB_NAME + ")");
        }
    }

    public static int deviceCount() {
        requireAvailable();
        return nativeDeviceCount();
    }

    private static native int nativeDeviceCount();

    static native long createStream();

    static native void destroyStream(long handle);

    static native long loadModule(byte[] hsaco);

    static native void unloadModule(long moduleHandle);

    static native long getFunction(long moduleHandle, String name);

    static native void launchKernel(
            long functionHandle,
            int gridDimX,
            int gridDimY,
            int gridDimZ,
            int blockDimX,
            int blockDimY,
            int blockDimZ,
            int sharedMemBytes,
            long streamHandle,
            long argsHandle);

    static native long malloc(long byteSize);

    static native void free(long devicePtr);

    static native void memcpyHtoD(long dstPtr, long dstOffset, long srcAddress, long byteSize);

    static native void memcpyDtoH(long dstAddress, long srcPtr, long srcOffset, long byteSize);

    static native void memcpyDtoD(
            long dstPtr, long dstOffset, long srcPtr, long srcOffset, long byteSize);

    // Memory fill operations
    static native void memsetD8(long dstPtr, long dstOffset, long byteSize, byte value);

    static native void memsetD16(long dstPtr, long dstOffset, long elementCount, short value);

    static native void memsetD32(long dstPtr, long dstOffset, long elementCount, int value);

    static native void memsetD64(long dstPtr, long dstOffset, long elementCount, long value);
}
