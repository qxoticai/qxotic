// Parallel work runner. General concurrency utilities, independent of
// the tensor/kernel code that uses them.
package com.llama4j;

import java.util.function.IntConsumer;
import java.util.stream.IntStream;

final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}
