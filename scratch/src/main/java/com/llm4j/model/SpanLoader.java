package com.llm4j.model;

import com.llm4j.api.BaseTensorInfo;
import com.llm4j.span.FloatSpan;

import java.util.function.Function;

public interface SpanLoader extends Function<BaseTensorInfo, FloatSpan>, AutoCloseable {
    @Override
    default void close() throws Exception {
    }
}
