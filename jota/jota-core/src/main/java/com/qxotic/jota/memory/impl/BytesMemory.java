package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import java.util.Objects;

final class BytesMemory implements Memory<byte[]> {

    final byte[] bytes;

    private BytesMemory(byte[] bytes) {
        this.bytes = Objects.requireNonNull(bytes);
    }

    static Memory<byte[]> of(byte[] bytes) {
        return new BytesMemory(bytes);
    }

    @Override
    public long byteSize() {
        return bytes.length;
    }

    @Override
    public boolean isReadOnly() {
        return false;
    }

    @Override
    public Device device() {
        return DeviceType.JAVA.deviceIndex(0);
    }

    @Override
    public byte[] base() {
        return bytes;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public String toString() {
        return new StringBuilder("Memory{byte[], byteSize=")
                .append(byteSize())
                .append(", device=")
                .append(device())
                .append('}')
                .toString();
    }
}
