package com.qxotic.toknroll.libtoknroll;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Locale;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
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
 * (stderr without one, nowhere for NULL). The code logs through System.Logger, which the JDK routes
 * to java.util.logging, so one root handler takes every line, one line each: a stack trace, at
 * debug, folds onto it.
 */
final class Log {
    interface Fn extends CFunctionPointer {
        @InvokeCFunctionPointer
        void invoke(int level, CCharPointer message, VoidPointer userData);
    }

    static final int DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, OFF = 4;
    private static final String[] NAMES = {"debug", "info", "warn", "error", "off"};
    private static final Level[] JUL = {
        Level.FINEST, Level.INFO, Level.WARNING, Level.SEVERE, Level.OFF
    };

    // set once by start, before any other call into the library
    private static String program = "toknroll";
    private static long fn, userData; // raw pointers: word types may not live in fields
    // held: java.util.logging forgets a level set on a logger it collects
    private static Logger processes;

    /**
     * The runtime's first act, from the isolate's start: given is whether the host called log_set,
     * fn its callback (NULL: nowhere).
     */
    static void start(boolean given, Fn callback, VoidPointer user) {
        program = System.getProperty("toknroll.program", "toknroll");
        String variable = program.toUpperCase(Locale.ROOT) + "_LOG";
        String value = System.getenv(variable);
        int wanted = parse(value);
        int level = given && callback.isNull() ? OFF : wanted < 0 ? WARN : wanted;
        fn = callback.rawValue();
        userData = user.rawValue();
        Logger root = Logger.getLogger("");
        for (Handler handler : root.getHandlers()) root.removeHandler(handler);
        root.setLevel(JUL[level]); // a line below it is never formatted
        // the JDK traces every process start at debug, stack included: one per curl download chunk,
        // not a line the libraries' debug promises
        processes = Logger.getLogger("java.lang.ProcessBuilder");
        processes.setLevel(Level.OFF);
        if (level == OFF) return;
        Handler handler =
                new Handler() {
                    final SimpleFormatter format = new SimpleFormatter();

                    @Override
                    public void publish(LogRecord record) {
                        if (!isLoggable(record)) return;
                        int at = of(record.getLevel());
                        String text = record.getLoggerName() + ": " + format.formatMessage(record);
                        Throwable thrown = record.getThrown();
                        if (thrown != null && at == DEBUG) {
                            StringWriter trace = new StringWriter();
                            thrown.printStackTrace(new PrintWriter(trace));
                            text += ": " + trace;
                        } else if (thrown != null) text += " (" + thrown + ")";
                        emit(at, text);
                    }

                    @Override
                    public void flush() {}

                    @Override
                    public void close() {}
                };
        // the handler's own level too: a logger given a finer one passes the root's by
        handler.setLevel(JUL[level]);
        root.addHandler(handler);
        if (wanted < 0 && value != null && !value.isBlank())
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
     * off, error, warn, info or debug in any case; -1 for anything else, unset and empty included.
     */
    private static int parse(String value) {
        if (value == null) return -1;
        for (int i = DEBUG; i <= OFF; i++) if (NAMES[i].equalsIgnoreCase(value.strip())) return i;
        return -1;
    }

    private static int of(Level level) {
        int v = level.intValue();
        return v >= Level.SEVERE.intValue()
                ? ERROR
                : v >= Level.WARNING.intValue() ? WARN : v >= Level.INFO.intValue() ? INFO : DEBUG;
    }

    /** One line: its breaks folded, a stack trace's frames joined by " | ". */
    static String oneLine(String text) {
        return text.strip().replaceAll("\\s*\\R\\s*", " | ");
    }

    private static void emit(int at, String text) {
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

    private Log() {}
}
