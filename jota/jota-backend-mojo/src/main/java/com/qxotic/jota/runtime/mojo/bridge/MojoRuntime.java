package com.qxotic.jota.runtime.mojo.bridge;

import com.qxotic.jota.NativeLibraryLoader;
import java.util.concurrent.atomic.AtomicReference;

/** JNI bindings for the libjota_mojo.so runtime bridge. */
public final class MojoRuntime {

    public static final int ABI_VERSION = 1;
    public static final String DEFAULT_BACKEND = "hip";
    private static final String LIB_NAME = "jota_mojo";
    private static final AtomicReference<Boolean> AVAILABLE = new AtomicReference<>();

    private MojoRuntime() {}

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

    public static void requireAvailable() {
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "Mojo backend not available ("
                            + LIB_NAME
                            + "). Ensure lib"
                            + LIB_NAME
                            + " is available on java.library.path or packaged in META-INF/native/");
        }
    }

    public static long init(int abiVersion, String fixedTarget) {
        return initWithBackend(abiVersion, fixedTarget, DEFAULT_BACKEND);
    }

    public static native long initWithBackend(
            int abiVersion, String fixedTarget, String fixedBackend);

    public static native void shutdown(long contextHandle);

    public static native String lastError(long contextHandle);
}
