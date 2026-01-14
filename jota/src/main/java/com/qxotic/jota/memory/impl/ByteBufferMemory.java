package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;

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
        return byteBuffer.isDirect() ? Device.NATIVE : Device.JAVA;
    }

    @Override
    public ByteBuffer base() {
        return this.byteBuffer;
    }

    @Override
    public boolean supportsDataType(DataType dataType) {
        return true; // all data types supported
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
        StringBuilder sb = new StringBuilder("Memory{ByteBuffer, byteSize=")
                .append(byteSize())
                .append(", device=")
                .append(device());
        if (isReadOnly()) {
            sb.append(", readOnly=true");
        }
        sb.append('}');
        return sb.toString();
    }
}
