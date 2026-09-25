package com.qxotic.jinfer.cli;

import java.io.PrintStream;

/** Append-only loading dots on stderr; redirected output gets one static line. */
final class LoadSpinner implements AutoCloseable {

    private final Thread ticker;
    private final PrintStream out;
    private boolean closed;

    private LoadSpinner(Thread ticker, PrintStream out) {
        this.ticker = ticker;
        this.out = out;
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
        out.print(label + " ...");
        out.flush();
        Thread ticker =
                new Thread(
                        () -> {
                            while (!Thread.currentThread().isInterrupted()) {
                                try {
                                    Thread.sleep(500);
                                } catch (InterruptedException done) {
                                    return;
                                }
                                out.print('.');
                                out.flush();
                            }
                        },
                        "jinfer-load-spinner");
        ticker.setDaemon(true);
        ticker.start();
        return new LoadSpinner(ticker, out);
    }

    /** Stops the dots and ends the line; idempotent, no-op off-terminal. */
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
        out.println();
        out.flush();
    }
}
