package com.qxotic.jinfer.libjinfer;

import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.graalvm.nativeimage.c.function.CFunctionPointer;
import org.graalvm.nativeimage.c.function.InvokeCFunctionPointer;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.CTypeConversion.CCharPointerHolder;
import org.graalvm.nativeimage.c.type.VoidPointer;
import org.graalvm.word.WordFactory;

/**
 * jinfer_log_set: silent until the host installs a callback. jinfer logs through System.Logger,
 * which the JDK routes to java.util.logging, so one root handler forwards every line.
 */
final class Log {
    interface Fn extends CFunctionPointer {
        @InvokeCFunctionPointer
        void invoke(int level, CCharPointer message, VoidPointer payload);
    }

    static final int DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3;

    static {
        // silent by default, as the header promises: java.util.logging would otherwise print every
        // INFO line (the hub's downloads) to stderr through its console handler
        Logger root = Logger.getLogger("");
        for (Handler handler : root.getHandlers()) root.removeHandler(handler);
    }

    /**
     * Forces this class's initialization, so the default silence is in place before anything logs.
     */
    static void init() {}

    private static volatile long fn; // raw pointers: word types may not live in object fields
    private static volatile long payload;
    private static Handler installed;

    static synchronized void set(Fn callback, VoidPointer userData) {
        Logger root = Logger.getLogger("");
        if (installed != null) {
            root.removeHandler(installed);
            installed = null;
        }
        fn = callback.isNull() ? 0 : callback.rawValue();
        payload = userData.rawValue();
        if (fn == 0) return;
        installed =
                new Handler() {
                    @Override
                    public void publish(LogRecord record) {
                        if (record.getMessage() == null) return;
                        String text = record.getLoggerName() + ": " + record.getMessage();
                        if (record.getThrown() != null) text += " (" + record.getThrown() + ")";
                        emit(level(record.getLevel()), text);
                    }

                    @Override
                    public void flush() {}

                    @Override
                    public void close() {}
                };
        installed.setLevel(Level.ALL);
        root.setLevel(Level.ALL);
        root.addHandler(installed);
    }

    private static int level(Level level) {
        int v = level.intValue();
        if (v >= Level.SEVERE.intValue()) return ERROR;
        if (v >= Level.WARNING.intValue()) return WARN;
        if (v >= Level.INFO.intValue()) return INFO;
        return DEBUG;
    }

    static void emit(int level, String message) {
        long target = fn;
        if (target == 0) return;
        try (CCharPointerHolder text = CTypeConversion.toCString(message)) {
            Fn callback = WordFactory.pointer(target);
            callback.invoke(level, text.get(), WordFactory.pointer(payload));
        }
    }

    /** A failure's stack trace, when a host listens; nothing is formatted otherwise. */
    static void trace(Throwable t) {
        if (fn == 0) return;
        java.io.StringWriter out = new java.io.StringWriter();
        t.printStackTrace(new java.io.PrintWriter(out));
        emit(DEBUG, "jinfer.native: " + out);
    }

    static void debug(String message) {
        emit(DEBUG, "jinfer.native: " + message);
    }

    private Log() {}
}
