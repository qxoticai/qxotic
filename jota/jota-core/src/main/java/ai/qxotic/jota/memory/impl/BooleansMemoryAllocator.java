package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;

final class BooleansMemoryAllocator implements MemoryAllocator<boolean[]> {

    private static final BooleansMemoryAllocator INSTANCE = new BooleansMemoryAllocator();

    public static MemoryAllocator<boolean[]> instance() {
        return INSTANCE;
    }

    private BooleansMemoryAllocator() {}

    @Override
    public Device device() {
        return Device.PANAMA;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public boolean supportsDataType(DataType dataType) {
        return dataType == DataType.BOOL; // ONLY BOOL - override default behavior
    }

    @Override
    public Memory<boolean[]> allocateMemory(long byteSize, long byteAlignment) {
        if (byteAlignment != 1) {
            throw new IllegalArgumentException("byteAlignment != 1");
        }
        int length = Math.toIntExact(byteSize); // 1 byte per boolean
        return BooleansMemory.of(new boolean[length]);
    }
}
