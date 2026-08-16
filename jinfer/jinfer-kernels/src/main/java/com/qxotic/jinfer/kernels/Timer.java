// A scoped timer for model-loading phases; try-with-resources logs the elapsed time on close.
// DEBUG on the jinfer.load logger: silent by default, and an embedder or a curious user turns it
// on with their logging config rather than jinfer deciding their stderr needs a timing line.
package com.qxotic.jinfer.kernels;

import java.util.concurrent.TimeUnit;

public interface Timer extends AutoCloseable {

    System.Logger LOG = System.getLogger("jinfer.load");

    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

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
