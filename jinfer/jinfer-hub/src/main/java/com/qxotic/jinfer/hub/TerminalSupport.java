package com.qxotic.jinfer.hub;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Best-effort console access shared by download progress and the CLI. Native access enables libc on
 * Linux/macOS and kernel32 on Windows; unavailable calls fall back to plain output and an
 * environment/default width. No shell commands or inference dependencies.
 */
public final class TerminalSupport {
    private TerminalSupport() {}

    private static final boolean WINDOWS = System.getProperty("os.name").startsWith("Windows");
    private static final boolean MAC = System.getProperty("os.name").startsWith("Mac");
    private static final int UTF8 = 65001, VIRTUAL_TERMINAL_PROCESSING = 0x0004;
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT;
    private static final SymbolLookup LIBRARY = library();
    private static final MethodHandle ISATTY = downcall("isatty", FunctionDescriptor.of(INT, INT));
    private static final MethodHandle IOCTL =
            downcall(
                    "ioctl",
                    FunctionDescriptor.of(INT, INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS),
                    Linker.Option.firstVariadicArg(2));
    private static final MethodHandle GET_STD_HANDLE =
            downcall("GetStdHandle", FunctionDescriptor.of(ValueLayout.ADDRESS, INT));
    private static final MethodHandle GET_CONSOLE_MODE =
            downcall(
                    "GetConsoleMode",
                    FunctionDescriptor.of(INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    private static final MethodHandle SET_CONSOLE_MODE =
            downcall("SetConsoleMode", FunctionDescriptor.of(INT, ValueLayout.ADDRESS, INT));
    private static final MethodHandle GET_SCREEN_BUFFER_INFO =
            downcall(
                    "GetConsoleScreenBufferInfo",
                    FunctionDescriptor.of(INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    private static final MethodHandle GET_OUTPUT_CODE_PAGE =
            downcall("GetConsoleOutputCP", FunctionDescriptor.of(INT));
    private static final MethodHandle SET_OUTPUT_CODE_PAGE =
            downcall("SetConsoleOutputCP", FunctionDescriptor.of(INT, INT));
    private static boolean utf8Enabled;

    /** Whether standard descriptor 0, 1 or 2 is a terminal; false when unavailable. */
    public static boolean isTerminal(int fd) {
        if (fd < 0 || fd > 2) return false;
        try {
            if (!WINDOWS) return (int) ISATTY.invokeExact(fd) == 1;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment mode = arena.allocate(INT);
                return (int) GET_CONSOLE_MODE.invokeExact(stdHandle(fd), mode) != 0;
            }
        } catch (Throwable unsupported) {
            return false;
        }
    }

    /** Current width of output descriptor 1 or 2, then positive {@code COLUMNS}, else 80. */
    public static int columns(int fd) {
        if (fd == 1 || fd == 2) {
            try (Arena arena = Arena.ofConfined()) {
                if (WINDOWS) {
                    MemorySegment info = arena.allocate(22, 2); // CONSOLE_SCREEN_BUFFER_INFO
                    if ((int) GET_SCREEN_BUFFER_INFO.invokeExact(stdHandle(fd), info) != 0) {
                        int width = windowsColumns(info);
                        if (width > 0) return width;
                    }
                } else {
                    MemorySegment size = arena.allocate(8, 2); // struct winsize
                    long request = MAC ? 0x40087468L : 0x5413L;
                    if ((int) IOCTL.invokeExact(fd, request, size) == 0) {
                        int width = Short.toUnsignedInt(size.get(ValueLayout.JAVA_SHORT, 2));
                        if (width > 0) return width;
                    }
                }
            } catch (Throwable unsupported) {
                // No native access, no terminal, or unavailable platform API.
            }
        }
        try {
            int width = Integer.parseInt(System.getenv().getOrDefault("COLUMNS", "").strip());
            if (width > 0) return width;
        } catch (NumberFormatException unset) {
            // Use the conventional width only when no positive measurement is available.
        }
        return 80;
    }

    /** Windows' visible window, not the potentially much wider scrollback buffer. */
    static int windowsColumns(MemorySegment info) {
        // SMALL_RECT srWindow: Left at byte 10, Right at byte 14 (both signed SHORT).
        return info.get(ValueLayout.JAVA_SHORT, 14) - info.get(ValueLayout.JAVA_SHORT, 10) + 1;
    }

    /** Enables ANSI output on a Windows console; on POSIX, verifies the output is a terminal. */
    public static boolean enableAnsi(int fd) {
        if (fd != 1 && fd != 2) return false;
        if (!WINDOWS) return isTerminal(fd);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment handle = stdHandle(fd), mode = arena.allocate(INT);
            if ((int) GET_CONSOLE_MODE.invokeExact(handle, mode) == 0) return false;
            return (int)
                            SET_CONSOLE_MODE.invokeExact(
                                    handle, mode.get(INT, 0) | VIRTUAL_TERMINAL_PROCESSING)
                    != 0;
        } catch (Throwable unsupported) {
            return false;
        }
    }

    /**
     * Prepares a console for a UTF-8 writer. Windows' original code page is restored at exit; POSIX
     * follows the locale. Call only for terminal-bound output already encoded as UTF-8.
     */
    public static synchronized boolean enableUtf8() {
        if (!WINDOWS) return isUtf8(System.getenv());
        if (utf8Enabled) return true;
        try {
            int original = (int) GET_OUTPUT_CODE_PAGE.invokeExact();
            if (original == 0) return false;
            if (original != UTF8) {
                if (!setCodePage(UTF8)) return false;
                try {
                    Runtime.getRuntime()
                            .addShutdownHook(
                                    new Thread(
                                            () -> setCodePage(original), "jinfer-console-restore"));
                } catch (RuntimeException unavailable) {
                    setCodePage(original);
                    return false;
                }
            }
            utf8Enabled = true;
            return true;
        } catch (Throwable unsupported) {
            return false;
        }
    }

    static boolean isUtf8(Map<String, String> env) {
        for (String name : new String[] {"LC_ALL", "LC_CTYPE", "LANG"}) {
            String value = env.getOrDefault(name, "");
            if (!value.isEmpty())
                return value.toUpperCase(Locale.ROOT).replace("-", "").contains("UTF8");
        }
        return "UTF-8".equals(System.getProperty("native.encoding"));
    }

    private static boolean setCodePage(int codePage) {
        try {
            return (int) SET_OUTPUT_CODE_PAGE.invokeExact(codePage) != 0;
        } catch (Throwable unsupported) {
            // Console may already be detached during process shutdown.
            return false;
        }
    }

    private static MemorySegment stdHandle(int fd) throws Throwable {
        return (MemorySegment) GET_STD_HANDLE.invokeExact(-10 - fd);
    }

    private static SymbolLookup library() {
        // Optional UI must not emit restricted-method warnings in a plain downloader application.
        if (TerminalSupport.class.getModule().isNativeAccessEnabled()) {
            try {
                return WINDOWS
                        ? SymbolLookup.libraryLookup("kernel32", Arena.global())
                        : Linker.nativeLinker().defaultLookup();
            } catch (RuntimeException | LinkageError unsupported) {
                // Native access can be unavailable on an embedding runtime.
            }
        }
        return symbol -> Optional.empty();
    }

    private static MethodHandle downcall(
            String name, FunctionDescriptor descriptor, Linker.Option... options) {
        try {
            return LIBRARY.find(name)
                    .map(
                            symbol ->
                                    Linker.nativeLinker()
                                            .downcallHandle(symbol, descriptor, options))
                    .orElse(null);
        } catch (RuntimeException unsupported) {
            return null;
        }
    }
}
