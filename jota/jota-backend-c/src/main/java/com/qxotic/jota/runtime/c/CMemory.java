package com.qxotic.jota.runtime.c;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class CMemory implements Memory<MemorySegment> {

    private final MemorySegment segment;
    private final long mallocAddress;

    CMemory(MemorySegment segment, long mallocAddress) {
        this.segment = Objects.requireNonNull(segment, "segment");
        this.mallocAddress = mallocAddress;
    }

    @Override
    public long byteSize() {
        return segment.byteSize();
    }

    @Override
    public boolean isReadOnly() {
        return segment.isReadOnly();
    }

    @Override
    public Device device() {
        return new Device(DeviceType.C, 0);
    }

    @Override
    public MemorySegment base() {
        return segment;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    long mallocAddress() {
        return mallocAddress;
    }
}
