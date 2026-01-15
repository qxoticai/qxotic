package ai.qxotic.model.llm;

import ai.qxotic.span.FloatSpan;

import java.util.function.Function;

public interface SpanLoader extends Function<Object, FloatSpan>, AutoCloseable {
    @Override
    default void close() throws Exception {
    }
}
