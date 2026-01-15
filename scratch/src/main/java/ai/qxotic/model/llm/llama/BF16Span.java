package ai.qxotic.model.llm.llama;

import ai.qxotic.span.FloatSpan;

import java.lang.foreign.MemorySegment;

public final class BF16Span extends MemorySegmentSpan {
    public BF16Span(MemorySegment memorySegment) {
        super(memorySegment);
        assert memorySegment.byteSize() % BFloat16.BYTES == 0;
    }

    @Override
    public long size() {
        return this.memorySegment.byteSize() / BFloat16.BYTES;
    }

    @Override
    public FloatSpan slice(long spanIndex, long spanLength) {
        if (spanIndex == 0 && spanLength == size()) {
            return this;
        }
        long offsetInBytes = spanIndex * BFloat16.BYTES;
        long lengthInBytes = spanLength * BFloat16.BYTES;
        return new BF16Span(this.memorySegment.asSlice(offsetInBytes, lengthInBytes));
    }
}
