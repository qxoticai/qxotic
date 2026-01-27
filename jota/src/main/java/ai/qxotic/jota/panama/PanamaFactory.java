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

    public static MemoryContext<MemorySegment> context() {
        return new PanamaContext(UnsafeAllocatorArena.create());
    }

    public static MemoryContext<MemorySegment> context(MemoryAllocator<MemorySegment> allocator) {
        return new PanamaContext(allocator);
    }

    public static ScopedMemoryAllocator<MemorySegment> scopedAllocator() {
        return UnsafeAllocator.instance();
    }

    public static ScopedMemoryAllocatorArena<MemorySegment> newArena() {
        return UnsafeAllocatorArena.create();
    }

    public static MemoryAllocator<MemorySegment> autoAllocator() {
        return new PanamaAutoAllocator();
    }

    public static MemoryAllocator<MemorySegment> onHeapAllocator() {
        return PanamaBytesAllocator.instance();
    }

    public static MemoryAccess<MemorySegment> access() {
        return PanamaMemoryAccess.instance();
    }

    public static MemoryOperations<MemorySegment> operations() {
        return PanamaMemoryOperations.instance();
    }

    public static Memory<MemorySegment> memory(MemorySegment segment) {
        return PanamaMemory.of(segment);
    }
}
