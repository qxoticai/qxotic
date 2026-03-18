package com.qxotic.jota.runtime.nativeimpl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

record NativeMemorySegmentMemory(MemorySegment memorySegment) implements Memory<MemorySegment> {

    NativeMemorySegmentMemory(MemorySegment memorySegment) {
        this.memorySegment = Objects.requireNonNull(memorySegment);
    }

    static NativeMemorySegmentMemory of(MemorySegment memorySegment) {
        return new NativeMemorySegmentMemory(memorySegment);
    }

    @Override
    public long byteSize() {
        return memorySegment.byteSize();
    }

    @Override
    public boolean isReadOnly() {
        return memorySegment.isReadOnly();
    }

    @Override
    public Device device() {
        return new Device(DeviceType.PANAMA, 0);
    }

    @Override
    public MemorySegment base() {
        return memorySegment;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    public NativeMemorySegmentMemory asReadOnly() {
        if (isReadOnly()) {
            return this;
        } else {
            return new NativeMemorySegmentMemory(this.memorySegment.asReadOnly());
        }
    }

    @Override
    public String toString() {
        StringBuilder sb =
                new StringBuilder("Memory{MemorySegment, byteSize=")
                        .append(byteSize())
                        .append(", device=")
                        .append(device());
        if (isReadOnly()) {
            sb.append(", readOnly=true");
        }
        sb.append('}');
        return sb.toString();
    }
}
