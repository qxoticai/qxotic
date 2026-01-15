package ai.qxotic.model.llm.llama;

import ai.qxotic.format.gguf.GGMLType;
import ai.qxotic.span.FloatSpan;

import java.lang.foreign.MemorySegment;

public class Q8_0Span extends MemorySegmentSpan {
    public Q8_0Span(MemorySegment memorySegment) {
        super(memorySegment);
        assert memorySegment.byteSize() % GGMLType.Q8_0.getBlockByteSize() == 0;
    }

    @Override
    public long size() {
        return this.memorySegment.byteSize() / GGMLType.Q8_0.getBlockByteSize() * GGMLType.Q8_0.getElementsPerBlock();
    }

    @Override
    public FloatSpan slice(long spanIndex, long spanLength) {
        if (spanIndex % GGMLType.Q8_0.getElementsPerBlock() != 0 || spanLength % GGMLType.Q8_0.getElementsPerBlock() != 0) {
            throw new IllegalArgumentException("Slices must be aligned to " + GGMLType.Q8_0.getElementsPerBlock());
        }
        long offsetInBytes = spanIndex / GGMLType.Q8_0.getElementsPerBlock() * GGMLType.Q8_0.getBlockByteSize();
        long lengthInBytes = spanLength / GGMLType.Q8_0.getElementsPerBlock() * GGMLType.Q8_0.getBlockByteSize();
        return new Q8_0Span(this.memorySegment.asSlice(offsetInBytes, lengthInBytes));
    }
}
