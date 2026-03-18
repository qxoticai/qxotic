package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.runtime.NativeLibraryLoader;
import java.util.concurrent.atomic.AtomicReference;

/** JNI bindings for the HIP backend native runtime. */
public final class HipRuntime {

    private static final String LIB_NAME = "jota_hip";
    private static final AtomicReference<Boolean> AVAILABLE = new AtomicReference<>();

    private HipRuntime() {
        // Utility class
    }

    /** Returns true if the HIP backend native library is available. */
    public static boolean isAvailable() {
        Boolean available = AVAILABLE.get();
        if (available != null) {
            return available;
        }
        try {
            NativeLibraryLoader.load(LIB_NAME);
            AVAILABLE.set(true);
        } catch (UnsatisfiedLinkError e) {
            AVAILABLE.set(false);
        }
        return AVAILABLE.get();
    }

    /**
     * Requires that the HIP backend is available, throwing otherwise.
     *
     * @throws IllegalStateException if the HIP backend is not available
     */
    public static void requireAvailable() {
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "HIP backend not available ("
                            + LIB_NAME
                            + "). "
                            + "Ensure lib"
                            + LIB_NAME
                            + " is on java.library.path "
                            + "or the backend JAR includes natives for your platform ("
                            + NativeLibraryLoader.currentPlatform()
                            + "). "
                            + "Supported platforms: Linux x86_64/aarch64, Windows x86_64. "
                            + "Note: HIP requires AMD ROCm to be installed.");
        }
    }

    public static int deviceCount() {
        requireAvailable();
        return nativeDeviceCount();
    }

    public static int currentDevice() {
        requireAvailable();
        return nativeCurrentDevice();
    }

    public static String deviceArchName(int deviceIndex) {
        requireAvailable();
        if (deviceIndex < 0) {
            throw new IllegalArgumentException("deviceIndex must be >= 0");
        }
        return nativeDeviceArchName(deviceIndex);
    }

    public static String deviceName(int deviceIndex) {
        requireAvailable();
        return nativeDeviceName(deviceIndex);
    }

    public static long deviceTotalMem(int deviceIndex) {
        requireAvailable();
        return nativeDeviceTotalMem(deviceIndex);
    }

    public static long deviceSharedMemPerBlock(int deviceIndex) {
        requireAvailable();
        return nativeDeviceSharedMemPerBlock(deviceIndex);
    }

    public static int deviceComputeUnits(int deviceIndex) {
        requireAvailable();
        return nativeDeviceMultiProcessorCount(deviceIndex);
    }

    public static int deviceClockRateKHz(int deviceIndex) {
        requireAvailable();
        return nativeDeviceClockRateKHz(deviceIndex);
    }

    public static int deviceWarpSize(int deviceIndex) {
        requireAvailable();
        return nativeDeviceWarpSize(deviceIndex);
    }

    public static int deviceMaxThreadsPerBlock(int deviceIndex) {
        requireAvailable();
        return nativeDeviceMaxThreadsPerBlock(deviceIndex);
    }

    public static int[] deviceMaxBlockDim(int deviceIndex) {
        requireAvailable();
        return nativeDeviceMaxBlockDim(deviceIndex);
    }

    public static int[] deviceMaxGridDim(int deviceIndex) {
        requireAvailable();
        return nativeDeviceMaxGridDim(deviceIndex);
    }

    public static boolean deviceConcurrentKernels(int deviceIndex) {
        requireAvailable();
        return nativeDeviceConcurrentKernels(deviceIndex);
    }

    public static boolean deviceEccEnabled(int deviceIndex) {
        requireAvailable();
        return nativeDeviceEccEnabled(deviceIndex);
    }

    public static int deviceL2CacheSize(int deviceIndex) {
        requireAvailable();
        return nativeDeviceL2CacheSize(deviceIndex);
    }

    public static int deviceMemoryBusWidthBits(int deviceIndex) {
        requireAvailable();
        return nativeDeviceMemoryBusWidthBits(deviceIndex);
    }

    public static int deviceMemoryClockRateKHz(int deviceIndex) {
        requireAvailable();
        return nativeDeviceMemoryClockRateKHz(deviceIndex);
    }

    public static int deviceRegsPerBlock(int deviceIndex) {
        requireAvailable();
        return nativeDeviceRegsPerBlock(deviceIndex);
    }

    private static native int nativeDeviceCount();

    private static native int nativeCurrentDevice();

    private static native String nativeDeviceArchName(int deviceIndex);

    private static native String nativeDeviceName(int deviceIndex);

    private static native long nativeDeviceTotalMem(int deviceIndex);

    private static native long nativeDeviceSharedMemPerBlock(int deviceIndex);

    private static native int nativeDeviceMultiProcessorCount(int deviceIndex);

    private static native int nativeDeviceClockRateKHz(int deviceIndex);

    private static native int nativeDeviceWarpSize(int deviceIndex);

    private static native int nativeDeviceMaxThreadsPerBlock(int deviceIndex);

    private static native int[] nativeDeviceMaxBlockDim(int deviceIndex);

    private static native int[] nativeDeviceMaxGridDim(int deviceIndex);

    private static native boolean nativeDeviceConcurrentKernels(int deviceIndex);

    private static native boolean nativeDeviceEccEnabled(int deviceIndex);

    private static native int nativeDeviceL2CacheSize(int deviceIndex);

    private static native int nativeDeviceMemoryBusWidthBits(int deviceIndex);

    private static native int nativeDeviceMemoryClockRateKHz(int deviceIndex);

    private static native int nativeDeviceRegsPerBlock(int deviceIndex);

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
