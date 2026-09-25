package com.qxotic.jinfer.cli;

import java.io.PrintStream;

/**
 * A stderr heartbeat for the silent seconds of a cold model load (mmap + parse + weight packing):
 * {@code - Loading model ... 12s}, redrawn in place, erased on stop. Rendered ONLY when stderr is a
 * terminal ({@link Terminal#isTerminal}; stdin may be piped audio): redirected runs see no bytes,
 * and embedders never reach this class (it is the CLI's, not the library's: jinfer-kernels keeps
 * its silent DEBUG {@code Timer}, and whoever owns the terminal owns the rendering).
 */
final class LoadSpinner implements AutoCloseable {

    private static final char[] FRAMES = {'|', '/', '-', '\\'};

    private final Thread ticker;
    private final int width; // of the widest line it may have drawn
    private final PrintStream out;
    private boolean closed;

    private LoadSpinner(Thread ticker, int width, PrintStream out) {
        this.ticker = ticker;
        this.width = width;
        this.out = out;
    }

    /** Starts the heartbeat; a no-op handle when stderr is not an interactive terminal. */
    static LoadSpinner start(String label, Main.IO io) {
        if (!io.isTerminal(2)) return new LoadSpinner(null, 0, io.err());
        long startNanos = System.nanoTime();
        Thread ticker =
                new Thread(
                        () -> {
                            for (int frame = 0; ; frame++) {
                                long s = (System.nanoTime() - startNanos) / 1_000_000_000L;
                                io.err().printf("\r%c %s ... %ds", FRAMES[frame & 3], label, s);
                                try {
                                    Thread.sleep(120);
                                } catch (InterruptedException done) {
                                    return;
                                }
                            }
                        },
                        "jinfer-load-spinner");
        ticker.setDaemon(true);
        ticker.start();
        return new LoadSpinner(
                ticker, label.length() + 16, io.err()); // "| " + label + " ... 99999s"
    }

    /** Stops the heartbeat and erases the line; idempotent, no-op off-terminal. */
    @Override
    public void close() {
        if (ticker == null || closed) {
            return;
        }
        closed = true;
        ticker.interrupt();
        try {
            ticker.join(500);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        // spaces, not an erase escape: a Windows console takes escapes only once the view is up
        out.print("\r" + " ".repeat(width) + "\r");
        out.flush();
    }
}
