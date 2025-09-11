package com.llm4j.model.llama;

import com.llm4j.span.FloatSpan;

import java.nio.ByteBuffer;

public abstract class ByteBufferSpan implements FloatSpan {

    final ByteBuffer byteBuffer;

    protected ByteBufferSpan(ByteBuffer byteBuffer) {
        this.byteBuffer = byteBuffer;
    }

    @Override
    public long sizeInBytes() {
        return this.byteBuffer.capacity();
    }

    @Override
    public abstract ByteBufferSpan slice(long spanIndex, long spanLength);
}
