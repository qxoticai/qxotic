package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class PanamaMemory implements Memory<MemorySegment> {

    final MemorySegment memorySegment;

    private PanamaMemory(MemorySegment memorySegment) {
        this.memorySegment = Objects.requireNonNull(memorySegment);
    }

    static PanamaMemory of(MemorySegment memorySegment) {
        return new PanamaMemory(memorySegment);
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
        return this.memorySegment.isNative() ? Device.PANAMA : Device.PANAMA;
    }

    @Override
    public MemorySegment base() {
        return memorySegment;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    public PanamaMemory asReadOnly() {
        if (isReadOnly()) {
            return this;
        } else {
            return new PanamaMemory(this.memorySegment.asReadOnly());
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
