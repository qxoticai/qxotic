package com.llm4j.model.llama;

import com.llm4j.gguf.GGMLType;
import com.llm4j.span.FloatSpan;

import java.lang.foreign.MemorySegment;

public class Q4_0Span extends MemorySegmentSpan {
    public Q4_0Span(MemorySegment memorySegment) {
        super(memorySegment);
        assert memorySegment.byteSize() % GGMLType.Q4_0.getBlockByteSize() == 0;
    }

    @Override
    public long size() {
        return this.memorySegment.byteSize() / GGMLType.Q4_0.getBlockByteSize() * GGMLType.Q4_0.getElementsPerBlock();
    }

    @Override
    public FloatSpan slice(long spanIndex, long spanLength) {
        if (spanIndex == 0 && spanLength == size()) {
            return this;
        }
        if (spanIndex % GGMLType.Q4_0.getElementsPerBlock() != 0 || spanLength % GGMLType.Q4_0.getElementsPerBlock() != 0) {
            throw new IllegalArgumentException("Slices must be multiples of " + GGMLType.Q4_0.getElementsPerBlock());
        }
        long offsetInBytes = spanIndex / GGMLType.Q4_0.getElementsPerBlock() * GGMLType.Q4_0.getBlockByteSize();
        long lengthInBytes = spanLength / GGMLType.Q4_0.getElementsPerBlock() * GGMLType.Q4_0.getBlockByteSize();
        return new Q4_0Span(this.memorySegment.asSlice(offsetInBytes, lengthInBytes));
    }
}

