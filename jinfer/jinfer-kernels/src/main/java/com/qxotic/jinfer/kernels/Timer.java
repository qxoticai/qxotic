// A scoped stderr timer for model-loading phases; try-with-resources logs the elapsed time on
// close.
package com.qxotic.jinfer.kernels;

import com.qxotic.jinfer.*;
import java.util.concurrent.TimeUnit;

public interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();

            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                System.err.println(
                        label
                                + ": "
                                + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS)
                                + " "
                                + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}
