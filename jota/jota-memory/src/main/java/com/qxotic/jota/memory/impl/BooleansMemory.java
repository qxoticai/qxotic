package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
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
        return booleans.length; // 1 byte per boolean in array
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
    public boolean[] base() {
        return booleans;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES; // 1 byte per boolean
    }

    @Override
    public boolean supportsDataType(DataType dataType) {
        return dataType == DataType.BOOL; // ONLY BOOL - override default behavior
    }

    @Override
    public String toString() {
        return "Memory{boolean[], byteSize=" + byteSize() + ", device=" + device() + '}';
    }
}
