package ai.qxotic.model.llm.llama;

import ai.qxotic.format.gguf.GGMLType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class Q8_0BBSpan extends ByteBufferSpan {
    public Q8_0BBSpan(ByteBuffer byteBuffer) {
        super(byteBuffer);
        assert byteBuffer.capacity() % GGMLType.Q8_0.getBlockByteSize() == 0;
    }

    @Override
    public long size() {
        return this.byteBuffer.capacity()
                / GGMLType.Q8_0.getBlockByteSize()
                * (long) GGMLType.Q8_0.getElementsPerBlock();
    }

    @Override
    public Q8_0BBSpan slice(long sliceIndex, long sliceLength) {
        if (sliceIndex % GGMLType.Q8_0.getElementsPerBlock() != 0
                || sliceLength % GGMLType.Q8_0.getElementsPerBlock() != 0) {
            throw new IllegalArgumentException(
                    "Slices must be aligned to " + GGMLType.Q8_0.getElementsPerBlock());
        }
        int offsetInBytes =
                Math.toIntExact(sliceIndex)
                        / GGMLType.Q8_0.getElementsPerBlock()
                        * GGMLType.Q8_0.getBlockByteSize();
        int lengthInBytes =
                Math.toIntExact(sliceLength)
                        / GGMLType.Q8_0.getElementsPerBlock()
                        * GGMLType.Q8_0.getBlockByteSize();
        return new Q8_0BBSpan(
                this.byteBuffer.slice(offsetInBytes, lengthInBytes).order(ByteOrder.nativeOrder()));
    }
}
