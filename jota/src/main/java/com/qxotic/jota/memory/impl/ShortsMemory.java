package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;

import java.util.Objects;

final class ShortsMemory implements Memory<short[]> {

    final short[] shorts;

    private ShortsMemory(short[] shorts) {
        this.shorts = Objects.requireNonNull(shorts);
    }

    static Memory<short[]> of(short[] shorts) {
        return new ShortsMemory(shorts);
    }

    @Override
    public long byteSize() {
        return shorts.length * (long) Short.BYTES;
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
    public short[] base() {
        return shorts;
    }

    @Override
    public boolean supportsDataType(DataType dataType) {
        return dataType == DataType.I16 || dataType == DataType.F16 || dataType == DataType.BF16;
    }

    @Override
    public String toString() {
        return new StringBuilder("Memory{short[], byteSize=")
                .append(byteSize())
                .append(", device=")
                .append(device())
                .append('}')
                .toString();
    }
}
