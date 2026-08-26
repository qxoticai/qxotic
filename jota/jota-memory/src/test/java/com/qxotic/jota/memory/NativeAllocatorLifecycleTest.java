package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class NativeAllocatorLifecycleTest {

    @Test
    void scopedArenaTracksExplicitAndBulkClose() {
        ScopedArena<MemorySegment> arena = MemoryAllocators.newScopedArena();
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
