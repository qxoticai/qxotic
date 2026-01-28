package ai.qxotic.jota.panama;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryOperations;
import ai.qxotic.jota.memory.ScopedMemoryAllocator;
import ai.qxotic.jota.memory.ScopedMemoryAllocatorArena;
import java.lang.foreign.MemorySegment;

public final class PanamaFactory {

    private PanamaFactory() {}

    public static MemoryContext<MemorySegment> createContext() {
        return createContext(createArena());
    }

    public static MemoryContext<MemorySegment> createContext(
            MemoryAllocator<MemorySegment> allocator) {
        return new PanamaContext(allocator);
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

    public static MemoryOperations<MemorySegment> mempryOperations() {
        return PanamaMemoryOperations.instance();
    }

    public static Memory<MemorySegment> memory(MemorySegment segment) {
        return PanamaMemory.of(segment);
    }
}
