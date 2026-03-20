package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import java.util.Objects;

final class FloatsMemory implements Memory<float[]> {

    final float[] floats;

    private FloatsMemory(float[] floats) {
        this.floats = Objects.requireNonNull(floats);
    }

    static Memory<float[]> of(float[] floats) {
        return new FloatsMemory(floats);
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
        return DeviceType.JAVA.deviceIndex(0);
    }

    @Override
    public float[] base() {
        return floats;
    }

    @Override
    public long memoryGranularity() {
        return Float.BYTES;
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
