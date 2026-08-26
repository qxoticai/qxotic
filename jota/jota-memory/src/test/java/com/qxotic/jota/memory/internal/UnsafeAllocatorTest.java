package com.qxotic.jota.memory.internal;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.DeviceType;
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
            assertTrue(memory.device().belongsTo(DeviceType.PANAMA));
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
        assertTrue(unsafeAllocator.device().belongsTo(DeviceType.PANAMA));
    }

    @Test
    void testDoubleCloseFails() {
        ScopedMemory<MemorySegment> memory = unsafeAllocator.allocateMemory(64);
        memory.close();
        assertThrows(IllegalStateException.class, memory::close);
    }
}
