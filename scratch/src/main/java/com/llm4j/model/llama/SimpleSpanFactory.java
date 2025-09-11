package com.llm4j.model.llama;

import java.lang.foreign.Arena;
import java.util.stream.LongStream;

public class SimpleSpanFactory implements FloatSpanFactory<ArraySpan> {
    @Override
    public ArraySpan allocate(long... dims) {
        int[] intDims = LongStream.of(dims).mapToInt(Math::toIntExact).toArray();
        return ArraySpan.allocate(intDims);
//        long totalNumberOfElements = LongStream.of(dims).reduce(1, Math::multiplyExact);
//        return new F32Span(Arena.ofAuto().allocate(totalNumberOfElements * Float.BYTES));
    }
}
