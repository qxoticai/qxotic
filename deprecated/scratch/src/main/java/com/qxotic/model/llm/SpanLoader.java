package com.qxotic.model.llm;

import com.qxotic.span.FloatSpan;
import java.util.function.Function;

public interface SpanLoader extends Function<Object, FloatSpan>, AutoCloseable {
    @Override
    default void close() throws Exception {}
}
