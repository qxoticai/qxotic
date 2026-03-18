package com.qxotic.jota.memory.impl;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.ScopedMemory;
import com.qxotic.jota.memory.ScopedMemoryAllocator;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class UnsafeAllocatorTest {

    static final ScopedMemoryAllocator<MemorySegment> unsafeAllocator =
            MemoryAllocatorFactory.ofPanama();

    @Test
    void testAllocateMemory() {
        try (ScopedMemory<MemorySegment> memory = unsafeAllocator.allocateMemory(100)) {
            assertNotNull(memory);
            assertEquals(100, memory.byteSize());
            assertFalse(memory.isReadOnly());
            assertTrue(memory.device().belongsTo(Device.PANAMA));
            assertNotNull(memory.base());
        }
    }

    @Test
    void testBadAlignment() {
        for (long badAlignment : new long[] {-1, 0, 3, 17, 69}) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> unsafeAllocator.allocateMemory(128, badAlignment));
        }
    }

    @Test
    void testDevice() {
        assertTrue(unsafeAllocator.device().belongsTo(Device.PANAMA));
    }

    @Test
    void testCloseReleasesMemory() {
        ScopedMemory<MemorySegment> memory = unsafeAllocator.allocateMemory(64 * (1 << 10));
        MemorySegment segment = memory.base();
        long address = segment.address();

        memory.close();

        // Verify memory was freed by trying to allocate at same address (this is a bit
        // implementation-dependent)
        try (ScopedMemory<MemorySegment> newMemory = unsafeAllocator.allocateMemory(100)) {
            assertNotEquals(address, newMemory.base().address());
        }
    }
}
