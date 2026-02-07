package ai.qxotic.jota.runtime.c;

import java.util.concurrent.atomic.AtomicReference;

public final class CNative {

    private static final String LIB_NAME = "jota_c";
    private static final AtomicReference<Boolean> AVAILABLE = new AtomicReference<>();

    private CNative() {}

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
            throw new IllegalStateException("C JNI runtime not available (" + LIB_NAME + ")");
        }
    }

    static native long loadKernel(String soPath, String symbol);

    static native void invokeKernel(
            long functionPtr, long[] bufferPtrs, long[] scalarBits, long scratchPtr);
}
