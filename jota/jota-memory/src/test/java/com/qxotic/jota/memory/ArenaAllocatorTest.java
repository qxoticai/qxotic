package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class ArenaAllocatorTest {

    @Test
    void sharedArenaOwnsItsAllocations() {
        Arena arena = Arena.ofShared();
        MemoryArena<MemorySegment> allocator = MemoryAllocators.ofArena(arena);
        MemoryDomain<MemorySegment> domain = MemoryDomains.of(allocator);
        MemoryView<MemorySegment> view = MemoryViews.zeros(domain, DataType.FP32, Shape.of(4));
        assertTrue(allocator.isAlive());
        assertEquals(0f, domain.directAccess().readFloat(view.memory(), 0));

        allocator.close();

        assertFalse(allocator.isAlive());
        // the JDK's scope check: a dead view throws instead of reading freed memory
        assertThrows(
                IllegalStateException.class,
                () -> domain.directAccess().readFloat(view.memory(), 0));
        assertThrows(IllegalStateException.class, () -> allocator.allocateMemory(4));
    }

    @Test
    void autoAndGlobalArenasCannotBeClosed() {
        MemoryArena<MemorySegment> auto = MemoryAllocators.ofArena(Arena.ofAuto());
        auto.allocateMemory(16);
        assertTrue(auto.isAlive());
        assertThrows(UnsupportedOperationException.class, auto::close);
        assertTrue(auto.isAlive());

        assertSame(
                MemoryAllocators.ofArena(Arena.global()), MemoryAllocators.ofArena(Arena.global()));
        assertThrows(
                UnsupportedOperationException.class,
                MemoryAllocators.ofArena(Arena.global())::close);
    }

    @Test
    void alignmentDefaultsToCachelineAndHonorsRequests() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> allocator = MemoryAllocators.ofArena(arena);
            assertEquals(64, allocator.defaultByteAlignment());
            assertEquals(0, allocator.allocateMemory(10).base().address() % 64);
            assertEquals(0, allocator.allocateMemory(10, 4096).base().address() % 4096);
            assertEquals(10, allocator.allocateMemory(10).byteSize());
            assertThrows(IllegalArgumentException.class, () -> allocator.allocateMemory(8, 3));
        }
    }
}
