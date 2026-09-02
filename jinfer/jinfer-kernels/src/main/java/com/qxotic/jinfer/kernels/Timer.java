package com.qxotic.jinfer.kernels;

import java.util.concurrent.TimeUnit;

/**
 * Scoped timer for model-loading phases. Closing it logs the elapsed time at {@link
 * System.Logger.Level#DEBUG} on the {@code jinfer.load} logger.
 */
public interface Timer extends AutoCloseable {

    System.Logger LOG = System.getLogger("jinfer.load");

    @Override
    void close(); // no Exception

    /** A timer logging {@code label}'s elapsed time in milliseconds on {@link #close()}. */
    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    /** A timer logging {@code label}'s elapsed time in {@code timeUnit} on {@link #close()}. */
    static Timer log(String label, TimeUnit timeUnit) {
        long startNanos = System.nanoTime();
        return () ->
                LOG.log(
                        System.Logger.Level.DEBUG,
                        "{0}: {1} {2}",
                        label,
                        timeUnit.convert(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS),
                        timeUnit.toChronoUnit().name().toLowerCase());
    }
}
