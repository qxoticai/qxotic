package com.qxotic.jota.memory.impl;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import org.junit.jupiter.api.Test;

class PanamaMemoryTest {

    @Test
    void testOfMemorySegment() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(100);
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(segment);
            assertSame(segment, memory.base());
            assertEquals(100, memory.byteSize());
            assertFalse(memory.isReadOnly());
            assertTrue(memory.device().belongsTo(DeviceType.PANAMA));
        }
    }

    @Test
    void testOfBuffer() {
        ByteBuffer buffer = ByteBuffer.allocate(40);
        Memory<MemorySegment> memory =
                MemoryFactory.ofMemorySegment(MemorySegment.ofBuffer(buffer));
        assertEquals(40, memory.byteSize());
        assertFalse(memory.isReadOnly());
    }

    @Test
    void testReadOnlySegment() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(40).asReadOnly();
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(segment);
            assertTrue(memory.isReadOnly());
        }
    }

    @Test
    void testNullMemorySegment() {
        assertThrows(
                NullPointerException.class,
                () -> MemoryFactory.ofMemorySegment((MemorySegment) null));
    }

    @Test
    void testToString() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(40);
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(segment);
            String str = memory.toString();
            assertTrue(str.contains("byteSize=40"));
            assertFalse(str.contains("readOnly=true")); // rw
            assertTrue(str.contains("device=panama:0"));
        }
    }
}
