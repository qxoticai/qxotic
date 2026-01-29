package ai.qxotic.jota.ir.interpreter;

import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;

final class EvaluatorUtils {

    private EvaluatorUtils() {}

    static long broadcastCoord(long coord, long inputDim, long outputDim) {
        if (inputDim == 1 && outputDim > 1) {
            return 0;
        }
        return coord;
    }

    static void forEachElement(Layout layout, ElementConsumer consumer) {
        Shape shape = layout.shape();
        long size = shape.size();
        for (long i = 0; i < size; i++) {
            long[] coord = Indexing.linearToCoord(shape, i);
            consumer.accept(i, coord);
        }
    }

    @FunctionalInterface
    interface ElementConsumer {
        void accept(long linearIndex, long[] coord);
    }
}
