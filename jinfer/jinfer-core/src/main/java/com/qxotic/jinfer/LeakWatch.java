package com.qxotic.jinfer;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Opt-in leak detector ({@code -Djinfer.leakDetection}), CloseGuard-style: owned native resources
 * capture their creation stack when armed, and reclamation by GC without a proper {@code close()}
 * logs that stack - the warning names the call site that forgot. WATCH-ONLY: it never frees
 * anything itself, so it is safe on resources whose reclamation cannot be (weights mid-kernel). Off
 * by default: zero capture cost, no cleaner thread.
 *
 * <p>Caveat, known and accepted: the trigger is GC, and a native-heavy JVM may never run one - a
 * quiet log is NOT a clean bill of health. The RSS-bounded leak-gate ITs remain the deterministic
 * detector; this is the debugging aid that turns "memory grows somewhere" into a stack trace.
 */
public final class LeakWatch {

    private LeakWatch() {}

    static volatile boolean enabled = Boolean.getBoolean("jinfer.leakDetection");

    /** Test seam; production sink is the {@code jinfer.leaks} platform logger. */
    static volatile java.util.function.Consumer<String> sink =
            msg -> System.getLogger("jinfer.leaks").log(System.Logger.Level.WARNING, msg);

    private static final Runnable NOOP = () -> {};

    // lazy: no cleaner thread unless detection is on and something arms
    private static final class Holder {
        static final Cleaner CLEANER = Cleaner.create();
    }

    /**
     * The creation site to embed in a resource's own cleanup action; null when detection is off.
     */
    static Throwable site(String what) {
        return enabled ? new Throwable(what + " was never closed - created here") : null;
    }

    /**
     * Watch {@code owner}: if it is reclaimed by GC before the returned disarm runnable runs (call
     * it first thing in {@code close()}), the creation site is reported. No-op when detection is
     * off.
     */
    public static Runnable arm(Object owner, String what) {
        if (!enabled) return NOOP;
        Throwable site = site(what);
        AtomicBoolean closed = new AtomicBoolean();
        // the action captures the flag and the site only - never the owner, or it would never die
        Holder.CLEANER.register(
                owner,
                () -> {
                    if (!closed.get()) report(site);
                });
        return () -> closed.set(true);
    }

    static void report(Throwable site) {
        StringWriter sw = new StringWriter();
        site.printStackTrace(new PrintWriter(sw));
        sink.accept(sw.toString());
    }
}
