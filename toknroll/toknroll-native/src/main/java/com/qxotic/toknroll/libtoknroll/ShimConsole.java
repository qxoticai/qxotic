package com.qxotic.toknroll.libtoknroll;

import com.qxotic.jinfer.hub.Console;
import org.graalvm.nativeimage.ImageInfo;
import org.graalvm.nativeimage.c.function.CFunction;

/**
 * The hub's console in the image: isatty, the window size and the console code page through the
 * shim's C helpers (toknroll_isolate.c), where the JVM's provider links them through the Foreign
 * Function API. The image keeps this provider and leaves the hub's out, so the Foreign Function API
 * and its method-handle machinery stay out of it. On a JVM these natives have no binding: no
 * console, as the hub without a provider.
 */
public final class ShimConsole implements Console {
    private static final boolean IMAGE = ImageInfo.inImageRuntimeCode();

    @CFunction("toknroll__tty")
    private static native int tty(int fd);

    @CFunction("toknroll__tty_ansi")
    private static native int ttyAnsi(int fd);

    @CFunction("toknroll__tty_columns")
    private static native int ttyColumns(int fd);

    @CFunction("toknroll__tty_utf8")
    private static native int ttyUtf8();

    public ShimConsole() {}

    @Override
    public boolean isTerminal(int fd) {
        return IMAGE && tty(fd) == 1;
    }

    @Override
    public int columns(int fd) {
        return IMAGE ? ttyColumns(fd) : 0;
    }

    @Override
    public boolean enableAnsi(int fd) {
        return IMAGE && ttyAnsi(fd) == 1;
    }

    @Override
    public boolean enableUtf8() {
        return IMAGE && ttyUtf8() == 1;
    }
}
