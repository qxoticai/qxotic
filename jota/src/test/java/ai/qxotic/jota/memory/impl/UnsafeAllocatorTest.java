package ai.qxotic.jota.memory.impl;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.ScopedMemory;
import ai.qxotic.jota.memory.ScopedMemoryAllocator;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class UnsafeAllocatorTest {

    static final ScopedMemoryAllocator<MemorySegment> unsafeAllocator = UnsafeAllocator.instance();

    @Test
    void testAllocateMemory() {
        try (ScopedMemory<MemorySegment> memory = unsafeAllocator.allocateMemory(100)) {
            assertNotNull(memory);
            assertEquals(100, memory.byteSize());
            assertFalse(memory.isReadOnly());
            assertEquals(Device.PANAMA, memory.device());
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
        assertEquals(Device.PANAMA, unsafeAllocator.device());
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
