package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;

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
        return Device.JAVA;
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
