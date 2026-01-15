package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;

import java.util.Objects;

final class DoublesMemory implements Memory<double[]> {

    final double[] doubles;

    private DoublesMemory(double[] doubles) {
        this.doubles = Objects.requireNonNull(doubles);
    }

    static Memory<double[]> of(double[] doubles) {
        return new DoublesMemory(doubles);
    }

    @Override
    public long byteSize() {
        return doubles.length * (long) Double.BYTES;
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
    public double[] base() {
        return doubles;
    }

    @Override
    public long memoryGranularity() {
        return Double.BYTES;
    }

    @Override
    public String toString() {
        return new StringBuilder("Memory{double[], byteSize=")
                .append(byteSize())
                .append(", device=")
                .append(device())
                .append('}')
                .toString();
    }
}
