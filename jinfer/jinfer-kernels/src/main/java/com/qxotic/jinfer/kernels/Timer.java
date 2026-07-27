// A scoped stderr timer for model-loading phases; try-with-resources logs the elapsed time on
// close.
package com.qxotic.jinfer.kernels;

import java.util.concurrent.TimeUnit;

public interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    static Timer log(String label, TimeUnit timeUnit) {
        long startNanos = System.nanoTime();
        return () ->
                System.err.println(
                        label
                                + ": "
                                + timeUnit.convert(
                                        System.nanoTime() - startNanos, TimeUnit.NANOSECONDS)
                                + " "
                                + timeUnit.toChronoUnit().name().toLowerCase());
    }
}
