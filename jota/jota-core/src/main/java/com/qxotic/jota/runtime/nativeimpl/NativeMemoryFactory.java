package com.qxotic.jota.runtime.nativeimpl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.ScopedMemoryAllocator;
import com.qxotic.jota.memory.ScopedMemoryAllocatorArena;
import java.lang.foreign.MemorySegment;

public final class NativeMemoryFactory {

    private NativeMemoryFactory() {}

    public static MemoryDomain<MemorySegment> createDomain() {
        return createDomain(createArena());
    }

    public static MemoryDomain<MemorySegment> createDomain(
            MemoryAllocator<MemorySegment> allocator) {
        return new NativeMemoryDomain(allocator);
    }

    public static ScopedMemoryAllocator<MemorySegment> scopedAllocator() {
        return NativeUnsafeAllocator.instance();
    }

    public static ScopedMemoryAllocatorArena<MemorySegment> createArena() {
        return NativeUnsafeAllocatorArena.create();
    }

    public static MemoryAllocator<MemorySegment> createManagedArena() {
        return new NativeAutoAllocator();
    }

    public static MemoryAllocator<MemorySegment> onHeapAllocator() {
        return NativeBytesAllocator.instance();
    }

    public static MemoryAccess<MemorySegment> memoryAccess() {
        return NativeMemoryAccess.instance();
    }

    public static MemoryOperations<MemorySegment> memoryOperations() {
        return NativeMemoryOperations.instance();
    }

    public static Memory<MemorySegment> memory(MemorySegment segment) {
        return NativeMemorySegmentMemory.of(segment);
    }
}
