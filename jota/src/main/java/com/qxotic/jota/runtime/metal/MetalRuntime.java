package com.qxotic.jota.runtime.metal;

import java.util.concurrent.atomic.AtomicReference;

public final class MetalRuntime {

    static final int STORAGE_MODE_SHARED = 0;
    static final int STORAGE_MODE_PRIVATE = 2;

    private static final String LIB_NAME = "jota_metal";
    private static final AtomicReference<Boolean> AVAILABLE = new AtomicReference<>();

    private MetalRuntime() {}

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
            throw new IllegalStateException("Metal runtime not available (" + LIB_NAME + ")");
        }
    }

    public static int deviceCount() {
        requireAvailable();
        return nativeDeviceCount();
    }

    private static native int nativeDeviceCount();

    static native long malloc(long byteSize, int storageMode);

    static native void free(long handle);

    static native void memcpyHtoD(long dstHandle, long dstOffset, long srcAddress, long byteSize);

    static native void memcpyDtoH(long dstAddress, long srcHandle, long srcOffset, long byteSize);

    static native void memcpyDtoD(
            long dstHandle, long dstOffset, long srcHandle, long srcOffset, long byteSize);

    static native void fillByte(long dstHandle, long dstOffset, long byteSize, byte value);

    static native void fillShort(long dstHandle, long dstOffset, long byteSize, short value);

    static native void fillInt(long dstHandle, long dstOffset, long byteSize, int value);

    static native void fillLong(long dstHandle, long dstOffset, long byteSize, long value);

    static native long loadLibrary(byte[] metallib);

    static native void unloadLibrary(long libraryHandle);

    static native long createPipeline(long libraryHandle, String functionName);

    static native void releasePipeline(long pipelineHandle);

    static native void launchKernel(
            long pipelineHandle,
            int gridDimX,
            int gridDimY,
            int gridDimZ,
            int blockDimX,
            int blockDimY,
            int blockDimZ,
            long argsHandle);
}
