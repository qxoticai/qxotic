package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.Memory;

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
        return Device.CPU;
    }

    @Override
    public float[] base() {
        return floats;
    }

    @Override
    public String toString() {
        return "FloatArrayMemory{" +
                "size=" + byteSize() +
                ", readOnly=" + isReadOnly() +
                ", device=" + device() +
                '}';
    }
}
