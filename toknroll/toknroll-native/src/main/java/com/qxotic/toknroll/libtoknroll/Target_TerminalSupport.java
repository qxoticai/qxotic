package com.qxotic.toknroll.libtoknroll;

import com.oracle.svm.core.annotate.Substitute;
import com.oracle.svm.core.annotate.TargetClass;
import java.util.Locale;
import java.util.Map;
import org.graalvm.nativeimage.c.function.CFunction;

/**
 * The hub's console probes, isatty and the window size, over the shim's C helpers instead of the
 * Foreign Function API: the one FFM user in the image, so the whole API and its method-handle
 * machinery stay out. Same answers as the original, see TerminalSupport in jinfer-hub.
 */
@TargetClass(className = "com.qxotic.jinfer.hub.TerminalSupport")
@Substitute
final class Target_TerminalSupport {
    @CFunction("toknroll__tty")
    private static native int tty(int fd);

    @CFunction("toknroll__tty_ansi")
    private static native int ttyAnsi(int fd);

    @CFunction("toknroll__tty_columns")
    private static native int ttyColumns(int fd);

    @CFunction("toknroll__tty_utf8")
    private static native int ttyUtf8();

    @Substitute
    public static boolean isTerminal(int fd) {
        return fd >= 0 && fd <= 2 && tty(fd) == 1;
    }

    @Substitute
    public static int columns(int fd) {
        int width = fd == 1 || fd == 2 ? ttyColumns(fd) : 0;
        if (width > 0) return width;
        try {
            width = Integer.parseInt(System.getenv().getOrDefault("COLUMNS", "").strip());
            if (width > 0) return width;
        } catch (NumberFormatException unset) {
            // the conventional width, as the original
        }
        return 80;
    }

    @Substitute
    public static boolean enableAnsi(int fd) {
        return (fd == 1 || fd == 2) && ttyAnsi(fd) == 1;
    }

    @Substitute
    public static boolean enableUtf8() {
        return ttyUtf8() == 1 || isUtf8(System.getenv());
    }

    @Substitute
    static boolean isUtf8(Map<String, String> env) {
        for (String name : new String[] {"LC_ALL", "LC_CTYPE", "LANG"}) {
            String value = env.getOrDefault(name, "");
            if (!value.isEmpty())
                return value.toUpperCase(Locale.ROOT).replace("-", "").contains("UTF8");
        }
        return "UTF-8".equals(System.getProperty("native.encoding"));
    }
}
