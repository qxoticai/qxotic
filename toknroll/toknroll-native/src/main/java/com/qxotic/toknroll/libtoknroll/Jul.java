package com.qxotic.toknroll.libtoknroll;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

/**
 * java.util.logging's root, taken over where the image carries it: libjinfer initializes its
 * classes at build time, so its loggers were made by the builder's JDK and log to java.util.logging
 * past the image's finder. One root handler sends every line to Log; at off, nothing is handled.
 * The lean image never calls this: Lean.IMAGE is a build-time constant, and this class is not in
 * it.
 */
final class Jul {
    private static final Level[] LEVELS = {
        Level.FINEST, Level.INFO, Level.WARNING, Level.SEVERE, Level.OFF
    };

    // held: java.util.logging forgets a level set on a logger it collects
    private static Logger processes;

    static void install(int level) {
        Logger root = Logger.getLogger("");
        for (Handler handler : root.getHandlers()) root.removeHandler(handler);
        root.setLevel(LEVELS[level]); // a line below it is never formatted
        // the JDK traces every process start at debug, stack included: one per curl download chunk,
        // not a line the libraries' debug promises
        processes = Logger.getLogger("java.lang.ProcessBuilder");
        processes.setLevel(Level.OFF);
        if (level == Log.OFF) return;
        Handler handler =
                new Handler() {
                    final SimpleFormatter format = new SimpleFormatter();

                    @Override
                    public void publish(LogRecord record) {
                        if (!isLoggable(record)) return;
                        int at = of(record.getLevel());
                        String text = record.getLoggerName() + ": " + format.formatMessage(record);
                        Throwable thrown = record.getThrown();
                        if (thrown != null && at == Log.DEBUG) {
                            StringWriter trace = new StringWriter();
                            thrown.printStackTrace(new PrintWriter(trace));
                            text += ": " + trace;
                        } else if (thrown != null) text += " (" + thrown + ")";
                        Log.emit(at, text);
                    }

                    @Override
                    public void flush() {}

                    @Override
                    public void close() {}
                };
        // the handler's own level too: a logger given a finer one passes the root's by
        handler.setLevel(LEVELS[level]);
        root.addHandler(handler);
    }

    private static int of(Level level) {
        int v = level.intValue();
        return v >= Level.SEVERE.intValue()
                ? Log.ERROR
                : v >= Level.WARNING.intValue()
                        ? Log.WARN
                        : v >= Level.INFO.intValue() ? Log.INFO : Log.DEBUG;
    }

    private Jul() {}
}
