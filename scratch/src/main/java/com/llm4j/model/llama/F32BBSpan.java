package com.llm4j.model.llama;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class F32BBSpan extends ByteBufferSpan {

    public F32BBSpan(ByteBuffer byteBuffer) {
        super(byteBuffer);
        assert byteBuffer.capacity() % Float.BYTES == 0;
    }

    @Override
    public F32BBSpan slice(long sliceIndex, long sliceLength) {
        if (sliceIndex == 0 && sliceLength == size()) {
            return this;
        }
        Util.checkBounds(0 <= sliceIndex && sliceIndex <= size(), "invalid offset");
        Util.checkBounds(sliceLength <= size(), "invalid size");
        Util.checkBounds(sliceIndex <= size() - sliceLength, "slice ouf of bounds");
        return new F32BBSpan(this.byteBuffer.slice(Math.toIntExact(sliceIndex * Float.BYTES), Math.toIntExact(sliceLength * Float.BYTES)).order(ByteOrder.nativeOrder()));
    }

    @Override
    public long size() {
        return this.byteBuffer.capacity() / Float.BYTES;
    }
}
