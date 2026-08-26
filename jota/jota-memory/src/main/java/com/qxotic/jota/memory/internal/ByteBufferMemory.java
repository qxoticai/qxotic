package com.qxotic.jota.memory.internal;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import java.nio.ByteBuffer;
import java.util.Objects;

final class ByteBufferMemory implements Memory<ByteBuffer> {

    private final ByteBuffer byteBuffer;

    private ByteBufferMemory(ByteBuffer byteBuffer) {
        this.byteBuffer = Objects.requireNonNull(byteBuffer);
    }

    /**
     * A cursor-free view of the same storage: absolute {@code get}/{@code put} honour the limit, so
     * the memory holds a duplicate with limit = capacity. Read-only is inherited; the byte order is
     * copied explicitly because {@code duplicate()} resets it to big-endian, like {@code slice()}.
     * The caller's buffer and its cursor are untouched.
     */
    static ByteBufferMemory of(ByteBuffer byteBuffer) {
        return new ByteBufferMemory(byteBuffer.duplicate().clear().order(byteBuffer.order()));
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
        return DeviceType.JAVA.deviceIndex(0);
    }

    @Override
    public ByteBuffer base() {
        return this.byteBuffer;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
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
        StringBuilder sb =
                new StringBuilder("Memory{ByteBuffer, byteSize=")
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
