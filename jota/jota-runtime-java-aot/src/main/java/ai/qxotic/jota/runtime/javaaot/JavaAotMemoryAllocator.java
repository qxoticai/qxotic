package ai.qxotic.jota.runtime.javaaot;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

final class JavaAotMemoryAllocator implements MemoryAllocator<MemorySegment> {

    @Override
    public Device device() {
        return Device.JAVA_AOT;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        MemorySegment segment = Arena.ofAuto().allocate(byteSize, byteAlignment);
        return new JavaAotMemory(segment);
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }
}
