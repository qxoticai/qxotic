package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;

import java.util.Objects;

final class LongsMemory implements Memory<long[]> {

    final long[] longs;

    private LongsMemory(long[] longs) {
        this.longs = Objects.requireNonNull(longs);
    }

    static Memory<long[]> of(long[] longs) {
        return new LongsMemory(longs);
    }

    @Override
    public long byteSize() {
        return longs.length * (long) Long.BYTES;
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
    public long[] base() {
        return longs;
    }

    @Override
    public long memoryGranularity() {
        return Long.BYTES;
    }

    @Override
    public String toString() {
        return new StringBuilder("Memory{long[], byteSize=")
                .append(byteSize())
                .append(", device=")
                .append(device())
                .append('}')
                .toString();
    }
}
