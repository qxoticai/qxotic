package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;

import java.util.Objects;

final class FloatArrayMemory implements Memory<float[]> {

    final float[] floats;

    private FloatArrayMemory(float[] floats) {
        this.floats = Objects.requireNonNull(floats);
    }

    static Memory<float[]> of(float[] floats) {
        return new FloatArrayMemory(floats);
    }

    @Override
    public long byteSize() {
        return floats.length * (long) Float.BYTES;
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
    public float[] base() {
        return floats;
    }

    @Override
    public String toString() {
        return new StringBuilder("Memory{float[], byteSize=")
                .append(byteSize())
                .append(", device=")
                .append(device())
                .append('}')
                .toString();
    }
}
