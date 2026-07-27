// A scoped stderr timer for model-loading phases; try-with-resources logs the elapsed time on
// close.
package com.qxotic.jinfer.kernels;

public interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        long startNanos = System.nanoTime();
        return () ->
                System.err.println(
                        label + ": " + (System.nanoTime() - startNanos) / 1_000_000 + " millis");
    }
}
