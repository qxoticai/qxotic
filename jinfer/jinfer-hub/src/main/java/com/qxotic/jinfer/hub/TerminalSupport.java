package com.qxotic.jinfer.hub;

import java.util.Locale;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * Best-effort console access shared by download progress and the CLI, over the {@link Console} the
 * runtime provides: the one {@link #use given}, else the first service provider found, else no
 * console at all, so plain output and an environment or default width. No shell commands or
 * inference dependencies.
 */
public final class TerminalSupport {
    private TerminalSupport() {}

    private static final boolean WINDOWS = System.getProperty("os.name").startsWith("Windows");
    private static volatile Console console;

    /**
     * The console this process probes with from now on, given before any download or terminal
     * output: a native image's own, where the JVM's provider is not in the image. Null returns to
     * the provider the service loader finds.
     */
    public static void use(Console given) {
        console = given;
    }

    private static Console console() {
        Console chosen = console;
        if (chosen != null) return chosen;
        for (Console provided : ServiceLoader.load(Console.class)) {
            return console = provided;
        }
        return console = new Console() {}; // no provider: no console
    }

    /** Whether standard descriptor 0, 1 or 2 is a terminal; false when unavailable. */
    public static boolean isTerminal(int fd) {
        return fd >= 0 && fd <= 2 && console().isTerminal(fd);
    }

    /** Current width of output descriptor 1 or 2, then positive {@code COLUMNS}, else 80. */
    public static int columns(int fd) {
        int width = fd == 1 || fd == 2 ? console().columns(fd) : 0;
        if (width > 0) return width;
        try {
            width = Integer.parseInt(System.getenv().getOrDefault("COLUMNS", "").strip());
            if (width > 0) return width;
        } catch (NumberFormatException unset) {
            // Use the conventional width only when no positive measurement is available.
        }
        return 80;
    }

    /** Enables ANSI output on a Windows console; on POSIX, verifies the output is a terminal. */
    public static boolean enableAnsi(int fd) {
        return (fd == 1 || fd == 2) && console().enableAnsi(fd);
    }

    /**
     * Prepares a console for a UTF-8 writer. Windows' original code page is restored at exit; POSIX
     * follows the locale. Call only for terminal-bound output already encoded as UTF-8.
     */
    public static boolean enableUtf8() {
        return WINDOWS ? console().enableUtf8() : isUtf8(System.getenv());
    }

    static boolean isUtf8(Map<String, String> env) {
        for (String name : new String[] {"LC_ALL", "LC_CTYPE", "LANG"}) {
            String value = env.getOrDefault(name, "");
            if (!value.isEmpty())
                return value.toUpperCase(Locale.ROOT).replace("-", "").contains("UTF8");
        }
        return "UTF-8".equals(System.getProperty("native.encoding"));
    }
}
