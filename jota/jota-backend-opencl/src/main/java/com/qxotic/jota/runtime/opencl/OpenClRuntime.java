package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.NativeLibraryLoader;
import java.util.concurrent.atomic.AtomicReference;

/** JNI bindings for the OpenCL backend native runtime. */
public final class OpenClRuntime {

    static final int STORAGE_MODE_SHARED = 0;
    static final int STORAGE_MODE_PRIVATE = 1;

    public static final String DEVICE_TYPE_PROPERTY = "jota.opencl.device.type";
    public static final String PLATFORM_INDEX_PROPERTY = "jota.opencl.platform.index";
    public static final String DEVICE_INDEX_PROPERTY = "jota.opencl.device.index";
    public static final String DEVICE_NAME_CONTAINS_PROPERTY = "jota.opencl.device.name.contains";

    private static final String LIB_NAME = "jota_opencl";
    private static final AtomicReference<Boolean> AVAILABLE = new AtomicReference<>();
    private static final AtomicReference<String> LOAD_FAILURE = new AtomicReference<>();
    private static final Object LOAD_LOCK = new Object();

    private OpenClRuntime() {
        // Utility class
    }

    /** Returns true if the OpenCL backend native library is available. */
    public static boolean isAvailable() {
        Boolean available = AVAILABLE.get();
        if (available != null) {
            return available;
        }
        synchronized (LOAD_LOCK) {
            available = AVAILABLE.get();
            if (available != null) {
                return available;
            }
            try {
                NativeLibraryLoader.load(LIB_NAME);
                LOAD_FAILURE.set(null);
                AVAILABLE.set(true);
            } catch (UnsatisfiedLinkError e) {
                LOAD_FAILURE.set(e.getMessage());
                AVAILABLE.set(false);
            }
        }
        return AVAILABLE.get();
    }

    /**
     * Requires that the OpenCL backend is available, throwing otherwise.
     *
     * @throws IllegalStateException if the OpenCL backend is not available
     */
    public static void requireAvailable() {
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "OpenCL backend not available (" + LIB_NAME + "). " + availabilityDetails());
        }
    }

    public static String availabilityDetails() {
        String failure = LOAD_FAILURE.get();
        if (failure == null || failure.isBlank()) {
            return "Ensure lib"
                    + LIB_NAME
                    + " is on java.library.path or the backend JAR includes natives for your"
                    + " platform ("
                    + NativeLibraryLoader.currentPlatform()
                    + "). Supported platforms: Linux x86_64/aarch64, Windows x86_64, macOS aarch64."
                    + " Also ensure OpenCL ICD/runtime libraries are installed.";
        }
        return "Library load error: "
                + failure
                + ". Ensure lib"
                + LIB_NAME
                + " is on java.library.path or the backend JAR includes natives for your platform ("
                + NativeLibraryLoader.currentPlatform()
                + "). Supported platforms: Linux x86_64/aarch64, Windows x86_64, macOS aarch64. "
                + "Also ensure OpenCL ICD/runtime libraries are installed.";
    }

    public static int deviceCount() {
        requireAvailable();
        return nativeDeviceCount();
    }

    public static String selectedDeviceType() {
        requireAvailable();
        return nativeSelectedDeviceType();
    }

    public static String selectedDeviceName() {
        requireAvailable();
        return nativeSelectedDeviceName();
    }

    public static String selectedPlatformName() {
        requireAvailable();
        return nativeSelectedPlatformName();
    }

    public static String listDevices() {
        requireAvailable();
        return nativeListDevices();
    }

    public static String selectionPropertiesSummary() {
        return DEVICE_TYPE_PROPERTY
                + "="
                + System.getProperty(DEVICE_TYPE_PROPERTY, "<default>")
                + ", "
                + PLATFORM_INDEX_PROPERTY
                + "="
                + System.getProperty(PLATFORM_INDEX_PROPERTY, "<default>")
                + ", "
                + DEVICE_INDEX_PROPERTY
                + "="
                + System.getProperty(DEVICE_INDEX_PROPERTY, "<default>")
                + ", "
                + DEVICE_NAME_CONTAINS_PROPERTY
                + "="
                + System.getProperty(DEVICE_NAME_CONTAINS_PROPERTY, "<default>");
    }

    public static String initFailureReason() {
        if (!isAvailable()) {
            return availabilityDetails();
        }
        return nativeInitFailureReason();
    }

    private static native int nativeDeviceCount();

    private static native String nativeSelectedDeviceType();

    private static native String nativeSelectedDeviceName();

    private static native String nativeSelectedPlatformName();

    private static native String nativeListDevices();

    private static native String nativeInitFailureReason();

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

    static native long loadLibrary(byte[] source);

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
