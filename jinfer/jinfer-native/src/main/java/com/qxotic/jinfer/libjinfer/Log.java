package com.qxotic.jinfer.libjinfer;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
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
        void invoke(int level, CCharPointer message, VoidPointer user);
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

    /**
     * A callback and its user pointer, replaced as one; raw pointers, as word types may not live in
     * object fields. running counts the calls in it, which a replacement waits out.
     */
    private static final class Sink {
        final long fn;
        final long user;
        final AtomicInteger running = new AtomicInteger();

        Sink(long fn, long user) {
            this.fn = fn;
            this.user = user;
        }
    }

    private static volatile Sink sink; // null: silent

    /** The sinks this thread is calling, innermost last: a set from a callback waits not for it. */
    private static final ThreadLocal<ArrayList<Sink>> CALLING =
            ThreadLocal.withInitial(ArrayList::new);

    private static Handler installed; // guarded by Log.class

    /**
     * jinfer_log_set. Once it returns, the callback it replaced is not running and is never called
     * again (but for calls of this very thread, a set from inside the callback): a call that picked
     * the old one up either is waited out here or sees the new one and takes that instead. The wait
     * is outside the lock, so a callback on another thread may set too.
     */
    static void set(Fn callback, VoidPointer user) {
        Sink old;
        synchronized (Log.class) {
            old = sink;
            sink = callback.isNull() ? null : new Sink(callback.rawValue(), user.rawValue());
            route(sink != null);
        }
        if (old == null) return;
        int own = 0;
        for (Sink s : CALLING.get()) if (s == old) own++;
        for (int spins = 0; old.running.get() > own; spins++) {
            if (spins < 1000) Thread.onSpinWait();
            else LockSupport.parkNanos(50_000); // a callback that takes its time: a host's lock
        }
    }

    /** The root handler that forwards every line, there only while a host listens. */
    private static void route(boolean on) {
        Logger root = Logger.getLogger("");
        if (on == (installed != null)) return;
        if (!on) {
            root.removeHandler(installed);
            installed = null;
            return;
        }
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
        // with the root at ALL the JDK traces every process start, stack trace included; curl runs
        // as a process per download chunk, and that trace is the JDK's own, not a line jinfer's
        // debug level promises
        Logger.getLogger("java.lang.ProcessBuilder").setLevel(Level.INFO);
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
        Sink s;
        for (; ; ) { // counted in before it is used, then checked still current: see set
            s = sink;
            if (s == null) return;
            s.running.incrementAndGet();
            if (sink == s) break;
            s.running.decrementAndGet();
        }
        ArrayList<Sink> calling = CALLING.get();
        calling.add(s);
        try (CCharPointerHolder text = CTypeConversion.toCString(message)) {
            Fn callback = WordFactory.pointer(s.fn);
            callback.invoke(level, text.get(), WordFactory.pointer(s.user));
        } finally {
            calling.remove(calling.size() - 1);
            s.running.decrementAndGet();
        }
    }

    /** A failure's stack trace, when a host listens; nothing is formatted otherwise. */
    static void trace(Throwable t) {
        if (sink == null) return;
        java.io.StringWriter out = new java.io.StringWriter();
        t.printStackTrace(new java.io.PrintWriter(out));
        emit(DEBUG, "jinfer.native: " + out);
    }

    static void debug(String message) {
        emit(DEBUG, "jinfer.native: " + message);
    }

    private Log() {}
}
