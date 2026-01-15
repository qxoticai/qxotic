package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;

import java.util.Objects;

final class BooleansMemory implements Memory<boolean[]> {

    final boolean[] booleans;

    private BooleansMemory(boolean[] booleans) {
        this.booleans = Objects.requireNonNull(booleans);
    }

    static Memory<boolean[]> of(boolean[] booleans) {
        return new BooleansMemory(booleans);
    }

    @Override
    public long byteSize() {
        return booleans.length;  // 1 byte per boolean in array
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
    public boolean[] base() {
        return booleans;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;  // 1 byte per boolean
    }

    @Override
    public boolean supportsDataType(DataType dataType) {
        return dataType == DataType.BOOL;  // ONLY BOOL - override default behavior
    }

    @Override
    public String toString() {
        return new StringBuilder("Memory{boolean[], byteSize=")
                .append(byteSize())
                .append(", device=")
                .append(device())
                .append('}')
                .toString();
    }
}
