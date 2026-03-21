package com.qxotic.jota.runtime.c;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import sun.misc.Unsafe;

final class CMemoryAllocator implements MemoryAllocator<MemorySegment>, MemoryArena<MemorySegment> {

    private static final Unsafe UNSAFE = initializeUnsafe();

    private final Device device;
    private final Set<CMemory> allocations = Collections.newSetFromMap(new ConcurrentHashMap<>());

    CMemoryAllocator(Device device) {
        this.device = device;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        if (byteSize < 0) {
            throw new IllegalArgumentException("invalid byteSize, must be >= 0");
        }
        if (!isPowerOf2(byteAlignment)) {
            throw new IllegalArgumentException("invalid byteAlignment, not a power of 2");
        }
        long allocationSize;
        try {
            allocationSize = Math.addExact(byteSize, byteAlignment - 1);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("invalid allocation size (overflow)", e);
        }
        long mallocAddress = UNSAFE.allocateMemory(allocationSize);
        long alignedAddress = mallocAddress;
        if (alignedAddress % byteAlignment != 0) {
            alignedAddress += byteAlignment - (alignedAddress % byteAlignment);
        }
        MemorySegment segment = MemorySegment.ofAddress(alignedAddress).reinterpret(byteSize);
        CMemory memory = new CMemory(device, segment, mallocAddress);
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
