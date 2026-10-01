package com.qxotic.jinfer.cli;

import java.io.PrintStream;

/** Append-only loading dots on stderr; redirected output gets one static line. */
final class LoadSpinner implements AutoCloseable {

    private final Thread ticker;
    private final PrintStream out;
    private final Thread shutdown;
    private boolean closed;

    private LoadSpinner(Thread ticker, PrintStream out) {
        this.ticker = ticker;
        this.out = out;
        this.shutdown = ticker == null ? null : new Thread(this::close, "jinfer-load-shutdown");
        if (shutdown != null) Runtime.getRuntime().addShutdownHook(shutdown);
    }

    static LoadSpinner start(String label, Main.IO io) {
        return start(label, io.err(), io.isTerminal(2) && !"dumb".equals(System.getenv("TERM")));
    }

    static LoadSpinner start(String label, PrintStream out, boolean animate) {
        if (!animate) {
            out.println(label + " ...");
            out.flush();
            return new LoadSpinner(null, out);
        }
        Thread ticker =
                new Thread(
                        () -> {
                            while (!Thread.currentThread().isInterrupted()) {
                                try {
                                    Thread.sleep(500);
                                } catch (InterruptedException done) {
                                    return;
                                }
                                synchronized (out) {
                                    if (Thread.currentThread().isInterrupted()) return;
                                    out.print('.');
                                    out.flush();
                                }
                            }
                        },
                        "jinfer-load-spinner");
        ticker.setDaemon(true);
        synchronized (out) {
            // Install cleanup before emitting a partial line; a signal may arrive at the first dot.
            var spinner = new LoadSpinner(ticker, out);
            out.print(label + " ...");
            out.flush();
            ticker.start();
            return spinner;
        }
    }

    /** Stops the dots and ends the line; idempotent, no-op off-terminal. */
    @Override
    public synchronized void close() {
        if (ticker == null || closed) {
            return;
        }
        closed = true;
        ticker.interrupt();
        // Finish after the last dot, even when the caller is interrupted.
        synchronized (out) {
            out.println();
            out.flush();
        }
        try {
            Runtime.getRuntime().removeShutdownHook(shutdown);
        } catch (IllegalStateException shuttingDown) {
            // The hook ends the same line when a signal bypasses try-with-resources.
        }
    }
}
