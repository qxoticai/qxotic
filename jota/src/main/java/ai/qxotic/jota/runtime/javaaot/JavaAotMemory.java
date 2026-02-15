package ai.qxotic.jota.runtime.javaaot;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

record JavaAotMemory(MemorySegment base) implements Memory<MemorySegment> {

    JavaAotMemory {
        Objects.requireNonNull(base, "base");
    }

    @Override
    public long byteSize() {
        return base.byteSize();
    }

    @Override
    public boolean isReadOnly() {
        return base.isReadOnly();
    }

    @Override
    public Device device() {
        return Device.JAVA_AOT;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }
}
