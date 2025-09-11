package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.Memory;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.*;

class PanamaMemoryTest {

    @Test
    void testOfMemorySegment() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(100);
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(segment);
            assertSame(segment, memory.base());
            assertEquals(100, memory.byteSize());
            assertFalse(memory.isReadOnly());
            assertEquals(Device.CPU, memory.device());
        }
    }

    @Test
    void testOfBuffer() {
        ByteBuffer buffer = ByteBuffer.allocate(40);
        Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(MemorySegment.ofBuffer(buffer));
        assertEquals(40, memory.byteSize());
        assertFalse(memory.isReadOnly());
    }

    @Test
    void testAsReadOnly() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(40);
            PanamaMemory memory = PanamaMemory.of(segment);
            assertFalse(memory.isReadOnly());

            PanamaMemory readOnly = memory.asReadOnly();
            assertTrue(readOnly.isReadOnly());

            // Calling asReadOnly on already read-only memory should return same instance
            PanamaMemory readOnly2 = readOnly.asReadOnly();
            assertSame(readOnly, readOnly2);
        }
    }

    @Test
    void testNullMemorySegment() {
        assertThrows(NullPointerException.class, () -> MemoryFactory.ofMemorySegment((MemorySegment) null));
    }

    @Test
    void testToString() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(40);
            PanamaMemory memory = PanamaMemory.of(segment);
            String str = memory.toString();
            assertTrue(str.contains("size=40"));
            assertTrue(str.contains("readOnly=false"));
            assertTrue(str.contains("device=cpu"));
        }
    }
}
