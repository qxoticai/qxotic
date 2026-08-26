package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.internal.MemoryAllocatorFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class NativeAllocatorLifecycleTest {

    @Test
    void autoAllocatorRejectsAllocationAfterClose() {
        MemoryArena<MemorySegment> arena =
                (MemoryArena<MemorySegment>) MemoryAllocatorFactory.newPanamaAuto();
        assertTrue(arena.isAlive());
        arena.allocateMemory(1);

        arena.close();

        assertTrue(arena.isAlive());
        assertThrows(IllegalStateException.class, () -> arena.allocateMemory(1));
        assertThrows(IllegalStateException.class, arena::close);
    }

    @Test
    void onHeapAllocatorIsAlwaysAliveAndOnlyPromisesByteAlignment() {
        MemoryArena<MemorySegment> arena =
                (MemoryArena<MemorySegment>) MemoryAllocatorFactory.newPanamaOnHeap();

        assertDoesNotThrow(() -> arena.allocateMemory(4, 1));
        assertThrows(IllegalArgumentException.class, () -> arena.allocateMemory(4, 0));
        assertThrows(IllegalArgumentException.class, () -> arena.allocateMemory(4, -1));
        assertThrows(IllegalArgumentException.class, () -> arena.allocateMemory(4, 2));
        arena.close();
        assertTrue(arena.isAlive());
        assertDoesNotThrow(() -> arena.allocateMemory(1));
    }

    @Test
    void scopedArenaTracksExplicitAndBulkClose() {
        ScopedMemoryAllocatorArena<MemorySegment> arena = MemoryAllocatorFactory.newPanamaArena();
        ScopedMemory<MemorySegment> explicitlyClosed = arena.allocateMemory(8);
        ScopedMemory<MemorySegment> arenaClosed = arena.allocateMemory(8);

        explicitlyClosed.close();
        assertThrows(IllegalStateException.class, explicitlyClosed::close);

        arena.close();

        assertFalse(arena.isAlive());
        assertThrows(IllegalStateException.class, arenaClosed::close);
        assertThrows(IllegalStateException.class, () -> arena.allocateMemory(1));
        assertDoesNotThrow(arena::close);
    }
}
