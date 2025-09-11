package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.Memory;

import java.util.Objects;

final class ByteArrayMemory implements Memory<byte[]> {

    final byte[] bytes;

    private ByteArrayMemory(byte[] bytes) {
        this.bytes = Objects.requireNonNull(bytes);
    }

    static Memory<byte[]> of(byte[] bytes) {
        return new ByteArrayMemory(bytes);
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
        return Device.CPU;
    }

    @Override
    public byte[] base() {
        return bytes;
    }

    @Override
    public String toString() {
        return "ByteArrayMemory{" +
                "size=" + byteSize() +
                ", readOnly=" + isReadOnly() +
                ", device=" + device() +
                '}';
    }
}
