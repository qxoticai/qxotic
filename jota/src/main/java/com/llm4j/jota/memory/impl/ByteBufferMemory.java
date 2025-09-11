package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.Memory;

import java.nio.ByteBuffer;
import java.util.Objects;

final class ByteBufferMemory implements Memory<ByteBuffer> {

    private final ByteBuffer byteBuffer;

    private ByteBufferMemory(ByteBuffer byteBuffer) {
        this.byteBuffer = Objects.requireNonNull(byteBuffer);
    }

    static ByteBufferMemory of(ByteBuffer byteBuffer) {
        return new ByteBufferMemory(byteBuffer);
    }

    @Override
    public long byteSize() {
        return byteBuffer.capacity();
    }

    @Override
    public boolean isReadOnly() {
        return byteBuffer.isReadOnly();
    }

    @Override
    public Device device() {
        return Device.CPU;
    }

    @Override
    public ByteBuffer base() {
        return this.byteBuffer;
    }

    public ByteBufferMemory asReadOnly() {
        if (isReadOnly()) {
            return this;
        } else {
            return of(this.byteBuffer.asReadOnlyBuffer());
        }
    }

    @Override
    public String toString() {
        return "ByteBufferMemory{" +
                "size=" + byteSize() +
                "readOnly=" + isReadOnly() +
                "device=" + device() +
                '}';
    }
}
