package com.qxotic.toknroll.libtoknroll;

import com.qxotic.jinfer.hub.TerminalSupport;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Locale;
import java.util.ResourceBundle;
import org.graalvm.nativeimage.c.function.CFunctionPointer;
import org.graalvm.nativeimage.c.function.InvokeCFunctionPointer;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.CTypeConversion.CCharPointerHolder;
import org.graalvm.nativeimage.c.type.VoidPointer;
import org.graalvm.word.WordFactory;

/**
 * The libraries' log lines, libtoknroll's and libjinfer's alike: {@code <PROGRAM>_LOG} says how
 * much (off, error, warn, info, debug; warn by default), the host's log_set callback says where
 * (stderr without one, nowhere for NULL). The code logs through System.Logger, and the image's
 * finder is {@link LogFinder}, so every line comes here, one line each: a stack trace, at debug,
 * folds onto it. No java.util.logging in the image: its manager, handlers and formatters are a
 * megabyte.
 */
final class Log {
    interface Fn extends CFunctionPointer {
        @InvokeCFunctionPointer
        void invoke(int level, CCharPointer message, VoidPointer userData);
    }

    static final int DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, OFF = 4;
    private static final String[] NAMES = {"debug", "info", "warn", "error", "off"};

    // set once by start, before any other call into the library
    private static String program = "toknroll";
    private static long fn, userData; // raw pointers: word types may not live in fields
    private static volatile int level = WARN; // a line below it is never formatted

    /**
     * The runtime's first act, from the isolate's start: given is whether the host called log_set,
     * fn its callback (NULL: nowhere).
     */
    static void start(boolean given, Fn callback, VoidPointer user) {
        program = System.getProperty("toknroll.program", "toknroll");
        String variable = program.toUpperCase(Locale.ROOT) + "_LOG";
        String value = System.getenv(variable);
        int wanted = parse(value);
        level = given && callback.isNull() ? OFF : wanted < 0 ? WARN : wanted;
        fn = callback.rawValue();
        userData = user.rawValue();
        progressKnob();
        TerminalSupport.use(
                new ShimConsole()); // the image's console: the JVM's provider is not in it
        if (level != OFF && wanted < 0 && value != null && !value.isBlank())
            emit(
                    WARN,
                    program
                            + ".native: "
                            + variable
                            + "="
                            + value
                            + " is not off, error, warn, info or debug: warn applies");
    }

    /**
     * This program's names for the hub's progress knob, -D<program>.progress and
     * <PROGRAM>_PROGRESS, mapped onto -Djinfer.progress before any fetch, the property over the
     * variable.
     */
    private static void progressKnob() {
        if (program.equals("jinfer")) return;
        String value = System.getProperty(program + ".progress");
        if (value == null) value = System.getenv(program.toUpperCase(Locale.ROOT) + "_PROGRESS");
        if (value != null) System.setProperty("jinfer.progress", value);
    }

    /**
     * off, error, warn, info or debug in any case; -1 for anything else, unset and empty included.
     */
    private static int parse(String value) {
        if (value == null) return -1;
        for (int i = DEBUG; i <= OFF; i++) if (NAMES[i].equalsIgnoreCase(value.strip())) return i;
        return -1;
    }

    private static int of(System.Logger.Level level) {
        int severity = level.getSeverity();
        return severity >= System.Logger.Level.ERROR.getSeverity()
                ? ERROR
                : severity >= System.Logger.Level.WARNING.getSeverity()
                        ? WARN
                        : severity >= System.Logger.Level.INFO.getSeverity() ? INFO : DEBUG;
    }

    /** One line: its breaks folded, a stack trace's frames joined by " | ". */
    static String oneLine(String text) {
        return text.strip().replaceAll("\\s*\\R\\s*", " | ");
    }

    static void emit(int at, String text) {
        text = oneLine(text);
        if (fn == 0) {
            System.err.println(program + ": " + NAMES[at] + ": " + text);
            return;
        }
        try (CCharPointerHolder message = CTypeConversion.toCString(text)) {
            Fn callback = WordFactory.pointer(fn);
            callback.invoke(at, message.get(), WordFactory.pointer(userData));
        }
    }

    /** The finder's answer: every System.getLogger in the image gets one of these. */
    static System.Logger logger(String name) {
        return new Logger(name);
    }

    /** A named logger: a line is the name, the message and the throwable, when the level allows. */
    private record Logger(String name) implements System.Logger {
        // the JDK traces every process start at debug, stack included: one per curl download
        // chunk, not a line the libraries' debug promises
        private static final String PROCESSES = "java.lang.ProcessBuilder";

        @Override
        public String getName() {
            return name;
        }

        @Override
        public boolean isLoggable(Level at) {
            return level != OFF && of(at) >= level && !name.equals(PROCESSES);
        }

        @Override
        public void log(Level at, ResourceBundle bundle, String message, Throwable thrown) {
            if (!isLoggable(at)) return;
            String text = name + ": " + message;
            if (thrown != null && of(at) == DEBUG) {
                StringWriter trace = new StringWriter();
                thrown.printStackTrace(new PrintWriter(trace));
                text += ": " + trace;
            } else if (thrown != null) text += " (" + thrown + ")";
            emit(of(at), text);
        }

        @Override
        public void log(Level at, ResourceBundle bundle, String format, Object... params) {
            if (!isLoggable(at)) return;
            emit(of(at), name + ": " + fill(format, params));
        }

        /** {0}, {1}... as MessageFormat fills them, which is all these libraries write. */
        private static String fill(String format, Object[] params) {
            if (params == null) return format;
            for (int i = 0; i < params.length; i++)
                format = format.replace("{" + i + "}", String.valueOf(params[i]));
            return format;
        }
    }

    private Log() {}
}
