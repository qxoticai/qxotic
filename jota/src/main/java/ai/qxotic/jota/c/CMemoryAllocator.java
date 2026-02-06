package ai.qxotic.jota.c;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import sun.misc.Unsafe;

final class CMemoryAllocator implements MemoryAllocator<MemorySegment>, MemoryArena<MemorySegment> {

    private static final Unsafe UNSAFE = initializeUnsafe();

    private final Set<CMemory> allocations = Collections.newSetFromMap(new ConcurrentHashMap<>());

    @Override
    public Device device() {
        return Device.C;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        if (!isPowerOf2(byteAlignment)) {
            throw new IllegalArgumentException("invalid byteAlignment, not a power of 2");
        }
        long mallocAddress = UNSAFE.allocateMemory(byteSize + byteAlignment - 1);
        long alignedAddress = mallocAddress;
        if (alignedAddress % byteAlignment != 0) {
            alignedAddress += byteAlignment - (alignedAddress % byteAlignment);
        }
        MemorySegment segment = MemorySegment.ofAddress(alignedAddress).reinterpret(byteSize);
        CMemory memory = new CMemory(segment, mallocAddress);
        allocations.add(memory);
        return memory;
    }

    @Override
    public void close() {
        for (CMemory memory : allocations) {
            UNSAFE.freeMemory(memory.mallocAddress());
        }
        allocations.clear();
    }

    private static boolean isPowerOf2(long value) {
        return value > 0 && ((value & (value - 1)) == 0);
    }

    private static Unsafe initializeUnsafe() {
        try {
            Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
            theUnsafe.setAccessible(true);
            return (Unsafe) theUnsafe.get(null);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Unable to initialize Unsafe", e);
        }
    }
}
