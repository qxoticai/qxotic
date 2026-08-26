package com.qxotic.jota.memory.internal;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.ScopedArena;
import java.lang.foreign.MemorySegment;

/**
 * Internal bridge for the {@code memory} API; use the constructors in {@code
 * com.qxotic.jota.memory}.
 */
public final class NativeMemoryFactory {

    private NativeMemoryFactory() {}

    public static MemoryDomain<MemorySegment> createDomain(
            MemoryAllocator<MemorySegment> allocator) {
        return new NativeMemoryDomain(allocator);
    }

    public static ScopedArena<MemorySegment> newScopedArena() {
        return NativeUnsafeAllocatorArena.create();
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
