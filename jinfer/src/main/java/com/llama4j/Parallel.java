// Parallel work runner. General concurrency utilities, independent of
// the tensor/kernel code that uses them.
package com.llama4j;

import java.util.function.IntConsumer;
import java.util.stream.IntStream;

final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    /** Runs {@code action} for each row index in {@code [0, rows)}, in parallel — but inline when
     *  there is a single row (decode), avoiding the fork/join overhead on the hot single-token path.
     *  The shared batched-prefill loop primitive for every model's per-row work. */
    public static void forRows(int rows, IntConsumer action) {
        if (rows == 1) {
            action.accept(0);
        } else {
            parallelFor(0, rows, action);
        }
    }
}
