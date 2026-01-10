package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;

import java.util.Objects;

final class IntArrayMemory implements Memory<int[]> {

    final int[] ints;

    private IntArrayMemory(int[] ints) {
        this.ints = Objects.requireNonNull(ints);
    }

    static Memory<int[]> of(int[] ints) {
        return new IntArrayMemory(ints);
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
    public String toString() {
        return "IntArrayMemory{" +
                "size=" + byteSize() +
                ", readOnly=" + isReadOnly() +
                ", device=" + device() +
                '}';
    }
}
