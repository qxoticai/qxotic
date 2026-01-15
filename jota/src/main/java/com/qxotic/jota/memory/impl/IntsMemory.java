package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;

import java.util.Objects;

final class IntsMemory implements Memory<int[]> {

    final int[] ints;

    private IntsMemory(int[] ints) {
        this.ints = Objects.requireNonNull(ints);
    }

    static Memory<int[]> of(int[] ints) {
        return new IntsMemory(ints);
    }

    @Override
    public long byteSize() {
        return ints.length * (long) Integer.BYTES;
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
    public int[] base() {
        return ints;
    }

    @Override
    public long memoryGranularity() {
        return Integer.BYTES;
    }

    @Override
    public String toString() {
        return new StringBuilder("Memory{int[], byteSize=")
                .append(byteSize())
                .append(", device=")
                .append(device())
                .append('}')
                .toString();
    }
}
