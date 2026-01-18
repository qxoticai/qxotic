package ai.qxotic.model.llm.llama;

import java.util.function.IntConsumer;
import java.util.function.LongConsumer;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        if (endExclusive - startInclusive == 1) {
            action.accept(startInclusive);
        } else {
            IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
        }
    }

    public static void parallelFor(
            int startInclusive, int endExclusive, int batchIndex, IntConsumer action) {
        if (batchIndex >= 0) {
            action.accept(batchIndex);
        } else if (endExclusive - startInclusive == 1) {
            action.accept(startInclusive);
        } else {
            IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
        }
    }

    public static void parallelForLong(
            long startInclusive, long endExclusive, LongConsumer action) {
        if (endExclusive - startInclusive == 1) {
            action.accept(startInclusive);
        } else {
            LongStream.range(startInclusive, endExclusive).parallel().forEach(action);
        }
    }
}
