package com.llm4j.model.llama;

import com.llm4j.span.Span;

import java.util.stream.LongStream;

public interface SpanFactory<T, S extends Span<T>> {
    S allocate(long... dims);

    default S allocateBatches(int batchSize, long... dims) {
        assert batchSize > 0 && Integer.bitCount(batchSize) == 1 : "batchSize must be a power of 2";
        long[] longDims = LongStream.concat(LongStream.of(batchSize), LongStream.of(dims)).toArray();
        return allocate(longDims);
    }
}

