package com.qxotic.model.llm.llama;

import com.qxotic.span.FloatSpan;

public interface FloatSpanFactory<S extends FloatSpan> extends SpanFactory<Float, S> {
    //    @Override
    //    S allocateBatches(int batchSize, long... dims);
    //
    //    @Override
    //    S allocate(long... dims);
}
