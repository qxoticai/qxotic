package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.memory.impl.MemoryAccessFactory;
import com.llm4j.jota.memory.impl.MemoryFactory;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PanamaMemoryAccessTest {

    private final MemoryAccess<MemorySegment> memoryAccess = MemoryAccessFactory.ofMemorySegment();

    @Test
    void testReadWriteByte() {
        try (var arena = Arena.ofConfined()) {
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(arena.allocate(DataType.I8.byteSize()));
            memoryAccess.writeByte(memory, 0, (byte) 42);
            assertEquals(42, memoryAccess.readByte(memory, 0));
        }
    }

    @Test
    void testReadWriteShort() {
        try (var arena = Arena.ofConfined()) {
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(arena.allocate(DataType.I16.byteSize()));
            memoryAccess.writeShort(memory, 0, (short) 12345);
            assertEquals(12345, memoryAccess.readShort(memory, 0));
        }
    }

    @Test
    void testReadWriteInt() {
        try (var arena = Arena.ofConfined()) {
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(arena.allocate(DataType.I32.byteSize()));
            memoryAccess.writeInt(memory, 0, 123456789);
            assertEquals(123456789, memoryAccess.readInt(memory, 0));
        }
    }

    @Test
    void testReadWriteLong() {
        try (var arena = Arena.ofConfined()) {
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(arena.allocate(DataType.I64.byteSize()));
            memoryAccess.writeLong(memory, 0, 123456789012345L);
            assertEquals(123456789012345L, memoryAccess.readLong(memory, 0));
        }
    }

    @Test
    void testReadWriteFloat() {
        try (var arena = Arena.ofConfined()) {
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(arena.allocate(DataType.F32.byteSize()));
            memoryAccess.writeFloat(memory, 0, (float) Math.PI);
            assertEquals((float) Math.PI, memoryAccess.readFloat(memory, 0));
        }
    }

    @Test
    void testReadWriteDouble() {
        try (var arena = Arena.ofConfined()) {
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(arena.allocate(DataType.F64.byteSize()));
            memoryAccess.writeDouble(memory, 0, Math.PI);
            assertEquals(Math.PI, memoryAccess.readDouble(memory, 0));
        }
    }

    @Test
    void testWriteToReadOnlyMemory() {
        try (var arena = Arena.ofConfined()) {
            MemorySegment readOnlySegment = arena.allocate(64).asReadOnly();
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(readOnlySegment);
            assertThrows(IllegalArgumentException.class,
                    () -> memoryAccess.writeByte(memory, 0, Byte.MAX_VALUE));

            assertThrows(IllegalArgumentException.class,
                    () -> memoryAccess.writeShort(memory, 0, Short.MAX_VALUE));

            assertThrows(IllegalArgumentException.class,
                    () -> memoryAccess.writeInt(memory, 0, Integer.MAX_VALUE));

            assertThrows(IllegalArgumentException.class,
                    () -> memoryAccess.writeLong(memory, 0, Long.MAX_VALUE));

            assertThrows(IllegalArgumentException.class,
                    () -> memoryAccess.writeFloat(memory, 0, Float.MAX_VALUE));

            assertThrows(IllegalArgumentException.class,
                    () -> memoryAccess.writeDouble(memory, 0, Double.MAX_VALUE));
        }
    }
}