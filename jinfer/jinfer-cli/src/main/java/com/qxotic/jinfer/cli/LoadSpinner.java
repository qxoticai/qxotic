package com.qxotic.jinfer.cli;

import java.io.Console;

/**
 * A stderr heartbeat for the silent seconds of a cold model load (mmap + parse + weight packing):
 * {@code - Loading model ... 12s}, redrawn in place, erased on stop. Rendered ONLY when stderr is
 * an interactive console ({@link System#console} attached and a TTY) - piped and scripted runs see
 * no bytes, and embedders never reach this class (it is the CLI's, not the library's:
 * jinfer-kernels keeps its silent DEBUG {@code Timer}, and whoever owns the terminal owns the
 * rendering).
 */
final class LoadSpinner {

    private static final char[] FRAMES = {'|', '/', '-', '\\'};

    private final Thread ticker;

    private LoadSpinner(Thread ticker) {
        this.ticker = ticker;
    }

    /** Starts the heartbeat; a no-op handle when stderr is not an interactive terminal. */
    static LoadSpinner start(String label) {
        Console console = System.console();
        if (console == null || !console.isTerminal()) {
            return new LoadSpinner(null);
        }
        long startNanos = System.nanoTime();
        Thread ticker =
                new Thread(
                        () -> {
                            for (int frame = 0; ; frame++) {
                                long s = (System.nanoTime() - startNanos) / 1_000_000_000L;
                                System.err.print(
                                        "\r" + FRAMES[frame & 3] + " " + label + " ... " + s + "s");
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
        return new LoadSpinner(ticker);
    }

    /** Stops the heartbeat and erases the line; idempotent, no-op off-terminal. */
    void stop() {
        if (ticker == null) {
            return;
        }
        ticker.interrupt();
        try {
            ticker.join(500);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        System.err.print("\r[2K"); // clear the spinner line
        System.err.flush();
    }
}
