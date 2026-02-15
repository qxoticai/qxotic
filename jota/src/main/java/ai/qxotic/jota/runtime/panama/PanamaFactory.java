package ai.qxotic.jota.runtime.panama;

import ai.qxotic.jota.memory.*;
import ai.qxotic.jota.memory.MemoryDomain;
import java.lang.foreign.MemorySegment;

public final class PanamaFactory {

    private PanamaFactory() {}

    public static MemoryDomain<MemorySegment> createDomain() {
        return createDomain(createArena());
    }

    public static MemoryDomain<MemorySegment> createDomain(
            MemoryAllocator<MemorySegment> allocator) {
        return new PanamaDomain(allocator);
    }

    public static ScopedMemoryAllocator<MemorySegment> scopedAllocator() {
        return UnsafeAllocator.instance();
    }

    public static ScopedMemoryAllocatorArena<MemorySegment> createArena() {
        return UnsafeAllocatorArena.create();
    }

    public static MemoryAllocator<MemorySegment> createManagedArena() {
        return new PanamaAutoAllocator();
    }

    public static MemoryAllocator<MemorySegment> onHeapAllocator() {
        return PanamaBytesAllocator.instance();
    }

    public static MemoryAccess<MemorySegment> memoryAccess() {
        return PanamaMemoryAccess.instance();
    }

    public static MemoryOperations<MemorySegment> memoryOperations() {
        return PanamaMemoryOperations.instance();
    }

    public static Memory<MemorySegment> memory(MemorySegment segment) {
        return PanamaMemory.of(segment);
    }
}
