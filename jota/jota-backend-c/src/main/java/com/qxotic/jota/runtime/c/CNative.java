package com.qxotic.jota.runtime.c;

import com.qxotic.jota.runtime.NativeLibraryLoader;
import java.util.concurrent.atomic.AtomicReference;

/** JNI bindings for the C backend native runtime. */
public final class CNative {

    private static final String LIB_NAME = "jota_c";
    private static final AtomicReference<Boolean> AVAILABLE = new AtomicReference<>();

    private CNative() {
        // Utility class
    }

    /** Returns true if the C backend native library is available. */
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
     * Requires that the C backend is available, throwing otherwise.
     *
     * @throws IllegalStateException if the C backend is not available
     */
    public static void requireAvailable() {
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "C backend not available ("
                            + LIB_NAME
                            + "). "
                            + "Ensure lib"
                            + LIB_NAME
                            + " is on java.library.path "
                            + "or the backend JAR includes natives for your platform ("
                            + NativeLibraryLoader.currentPlatform()
                            + "). Supported platforms: Linux x86_64/aarch64, Windows x86_64, macOS"
                            + " aarch64.");
        }
    }

    static native long loadKernel(String soPath, String symbol);

    static native void invokeKernel(
            long functionPtr, long[] bufferPtrs, long[] scalarBits, long scratchPtr);
}
