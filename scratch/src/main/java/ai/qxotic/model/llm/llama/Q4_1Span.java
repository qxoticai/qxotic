package ai.qxotic.model.llm.llama;

import ai.qxotic.format.gguf.GGMLType;
import ai.qxotic.span.FloatSpan;
import java.lang.foreign.MemorySegment;

public class Q4_1Span extends MemorySegmentSpan {
    public Q4_1Span(MemorySegment memorySegment) {
        super(memorySegment);
        assert memorySegment.byteSize() % GGMLType.Q4_1.getBlockByteSize() == 0;
    }

    @Override
    public long size() {
        return this.memorySegment.byteSize()
                / GGMLType.Q4_1.getBlockByteSize()
                * GGMLType.Q4_1.getElementsPerBlock();
    }

    @Override
    public FloatSpan slice(long spanIndex, long spanLength) {
        if (spanIndex % GGMLType.Q4_1.getElementsPerBlock() != 0
                || spanLength % GGMLType.Q4_1.getElementsPerBlock() != 0) {
            throw new IllegalArgumentException(
                    "Slices must be aligned to " + GGMLType.Q4_1.getElementsPerBlock());
        }
        long offsetInBytes =
                spanIndex / GGMLType.Q4_1.getElementsPerBlock() * GGMLType.Q4_1.getBlockByteSize();
        long lengthInBytes =
                spanLength / GGMLType.Q4_1.getElementsPerBlock() * GGMLType.Q4_1.getBlockByteSize();
        return new Q4_1Span(this.memorySegment.asSlice(offsetInBytes, lengthInBytes));
    }
}
