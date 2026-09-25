package com.qxotic.jinfer.cli;

import java.io.PrintStream;
import java.lang.foreign.AddressLayout;
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
 * The terminal on stderr, set up for the live transcription view: a buffered stream to write frames
 * to, whether it shows Unicode, and how many colors. Linux and macOS are asked through libc ({@code
 * isatty}, {@code ioctl(TIOCGWINSZ)}); on Windows the console is asked through kernel32, then
 * switched to ANSI escapes and UTF-8. Every native call is linked once and any failure, such as a
 * native image without its descriptor, reads as "no terminal": the caller falls back to plain
 * lines, never breaks.
 */
record Terminal(PrintStream out, boolean unicode, TranscriptHud.ColorDepth depth) {

    private static final boolean WINDOWS = System.getProperty("os.name").startsWith("Windows");
    private static final boolean MAC = System.getProperty("os.name").startsWith("Mac");
    private static final int VIRTUAL_TERMINAL_PROCESSING = 0x0004, UTF_8_CODE_PAGE = 65001;
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT;
    private static final AddressLayout POINTER = ValueLayout.ADDRESS;
    private static final SymbolLookup LIBRARY =
            WINDOWS ? library("kernel32") : Linker.nativeLinker().defaultLookup();

    // libc
    private static final MethodHandle ISATTY = downcall("isatty", FunctionDescriptor.of(INT, INT));
    private static final MethodHandle IOCTL =
            downcall(
                    "ioctl",
                    FunctionDescriptor.of(INT, INT, ValueLayout.JAVA_LONG, POINTER),
                    Linker.Option.firstVariadicArg(2));
    // kernel32
    private static final MethodHandle GET_STD_HANDLE =
            downcall("GetStdHandle", FunctionDescriptor.of(POINTER, INT));
    private static final MethodHandle GET_CONSOLE_MODE =
            downcall("GetConsoleMode", FunctionDescriptor.of(INT, POINTER, POINTER));
    private static final MethodHandle SET_CONSOLE_MODE =
            downcall("SetConsoleMode", FunctionDescriptor.of(INT, POINTER, INT));
    private static final MethodHandle GET_SCREEN_BUFFER_INFO =
            downcall("GetConsoleScreenBufferInfo", FunctionDescriptor.of(INT, POINTER, POINTER));
    private static final MethodHandle GET_OUTPUT_CODE_PAGE =
            downcall("GetConsoleOutputCP", FunctionDescriptor.of(INT));
    private static final MethodHandle SET_OUTPUT_CODE_PAGE =
            downcall("SetConsoleOutputCP", FunctionDescriptor.of(INT, INT));

    /**
     * The terminal on stderr, ready for escapes, or null where there is none that can move the
     * cursor: stderr redirected, {@code TERM=dumb}, or a Windows console that refuses ANSI.
     */
    static Terminal stderr() {
        return stderr(System.err, "auto");
    }

    static Terminal stderr(PrintStream out, String color) {
        if (!isTerminal(2) || "dumb".equals(System.getenv("TERM"))) return null;
        boolean unicode;
        TranscriptHud.ColorDepth depth =
                color.equals("off")
                        ? TranscriptHud.ColorDepth.NONE
                        : color.equals("on")
                                ? TranscriptHud.ColorDepth.TRUE
                                : TranscriptHud.ColorDepth.of(System.getenv());
        if (WINDOWS) {
            if (!enableAnsi()) return null;
            int codePage = outputCodePage();
            unicode = codePage != 0 && call(SET_OUTPUT_CODE_PAGE, UTF_8_CODE_PAGE) != 0;
            // the console outlives the process: give it back its code page, even on Ctrl-C
            if (unicode) {
                Thread restore = new Thread(() -> call(SET_OUTPUT_CODE_PAGE, codePage));
                Runtime.getRuntime().addShutdownHook(restore);
            }
            // every console that takes ANSI escapes (Windows 10 and on) takes 24-bit colors
            if (depth != TranscriptHud.ColorDepth.NONE) depth = TranscriptHud.ColorDepth.TRUE;
        } else {
            unicode = isUtf8(System.getenv());
        }
        return new Terminal(out, unicode, depth);
    }

    /** Whether file descriptor 0, 1 or 2 is an interactive terminal; false when unknown. */
    static boolean isTerminal(int fd) {
        if (WINDOWS) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment mode = arena.allocate(INT);
                return (int) GET_CONSOLE_MODE.invokeExact(stdHandle(fd), mode) != 0;
            } catch (Throwable unsupported) {
                return false;
            }
        }
        try {
            return (int) ISATTY.invokeExact(fd) == 1;
        } catch (Throwable unsupported) {
            return false;
        }
    }

    /** Columns of the terminal on stderr now; else {@code $COLUMNS}, else 80. */
    static int columns() {
        try (Arena arena = Arena.ofConfined()) {
            if (WINDOWS) {
                // CONSOLE_SCREEN_BUFFER_INFO: the visible window's left and right at 10 and 14
                MemorySegment info = arena.allocate(22);
                if ((int) GET_SCREEN_BUFFER_INFO.invokeExact(stdHandle(2), info) != 0)
                    return info.get(ValueLayout.JAVA_SHORT_UNALIGNED, 14)
                            - info.get(ValueLayout.JAVA_SHORT_UNALIGNED, 10)
                            + 1;
            } else {
                MemorySegment size = arena.allocate(8); // struct winsize: rows, columns, pixels
                long request = MAC ? 0x40087468L : 0x5413L; // TIOCGWINSZ
                if ((int) IOCTL.invokeExact(2, request, size) == 0) {
                    int columns = Short.toUnsignedInt(size.get(ValueLayout.JAVA_SHORT, 2));
                    if (columns > 0) return columns;
                }
            }
        } catch (Throwable unsupported) {
            // fall through to the environment
        }
        try {
            return Math.max(1, Integer.parseInt(System.getenv("COLUMNS")));
        } catch (NumberFormatException unset) {
            return 80;
        }
    }

    /**
     * Whether the locale's character set is UTF-8: the first of {@code LC_ALL}, {@code LC_CTYPE}
     * and {@code LANG} that is set says, else the JVM's native encoding.
     */
    static boolean isUtf8(Map<String, String> env) {
        for (String name : new String[] {"LC_ALL", "LC_CTYPE", "LANG"}) {
            String value = env.getOrDefault(name, "");
            if (!value.isEmpty())
                return value.toUpperCase(Locale.ROOT).replace("-", "").contains("UTF8");
        }
        return "UTF-8".equals(System.getProperty("native.encoding"));
    }

    /** Turns on ANSI escape processing for the console on stderr; false if it refuses. */
    private static boolean enableAnsi() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment handle = stdHandle(2);
            MemorySegment mode = arena.allocate(INT);
            if ((int) GET_CONSOLE_MODE.invokeExact(handle, mode) == 0) return false;
            int enabled = mode.get(INT, 0) | VIRTUAL_TERMINAL_PROCESSING;
            return (int) SET_CONSOLE_MODE.invokeExact(handle, enabled) != 0;
        } catch (Throwable unsupported) {
            return false;
        }
    }

    /** The console handle of file descriptor 0, 1 or 2: STD_INPUT_HANDLE is -10, and so on. */
    private static MemorySegment stdHandle(int fd) throws Throwable {
        return (MemorySegment) GET_STD_HANDLE.invokeExact(-10 - fd);
    }

    private static int outputCodePage() {
        try {
            return (int) GET_OUTPUT_CODE_PAGE.invokeExact();
        } catch (Throwable unsupported) {
            return 0;
        }
    }

    private static int call(MethodHandle handle, int argument) {
        try {
            return (int) handle.invokeExact(argument);
        } catch (Throwable unsupported) {
            return 0;
        }
    }

    private static SymbolLookup library(String name) {
        try {
            return SymbolLookup.libraryLookup(name, Arena.global());
        } catch (RuntimeException unsupported) {
            return symbol -> Optional.empty();
        }
    }

    /** The downcall, or null where this platform's library has no such function. */
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
