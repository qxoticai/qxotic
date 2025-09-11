package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;
import com.llm4j.jota.memory.impl.MemoryAllocatorFactory;
import com.llm4j.jota.memory.impl.MemoryFactory;
import com.llm4j.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.*;

class MemoryTest {

    @Test
    void testAllocate() {
        var allocator = MemoryAllocatorFactory.ofPanama();
        DataType dataType = DataType.F32;
        long totalBytes = dataType.byteSizeFor(3 * 5);
        try (var memory = allocator.allocateMemory(totalBytes)) {
            assertEquals(3 * 5 * dataType.byteSize(), memory.byteSize());
            assertFalse(memory.isReadOnly());
        }
    }

    @Test
    void testPanamaMemoryBase() {
        try (var arena = Arena.ofShared()) {
            MemorySegment memorySegment = arena.allocate(Float.BYTES * 16);
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
            assertEquals(memorySegment.byteSize(), memory.byteSize());
            assertSame(memorySegment, memory.base());
        }
    }

    @Test
    void testMemoryView() {
        try (var arena = Arena.ofShared()) {
            MemorySegment memorySegment = arena.allocate(Float.BYTES * 16);
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);


            var memoryView = MemoryViewFactory.of(Shape.of(16), DataType.F32, memory);
            assertTrue(memoryView.isContiguous());

            var it = OffsetIterator.create(memoryView);

            assertTrue(it.hasNext());
            for (int i = 0; i < 16; ++i) {
                assertEquals(4L * i, it.nextByteOffset());
            }
            assertFalse(it.hasNext());
        }
    }

    @Test
    void testMemoryViewMatrix() {
        try (var arena = Arena.ofShared()) {
            MemorySegment memorySegment = arena.allocate(Float.BYTES * (3 * 5));
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
            var memoryView = MemoryViewFactory.of(Shape.of(3, 5), DataType.F32, memory);

            assertTrue(memoryView.isContiguous());

            var it = OffsetIterator.create(memoryView);

            assertTrue(it.hasNext());
            long totalNumberOfElements = memoryView.shape().totalNumberOfElements();
            for (int i = 0; i < totalNumberOfElements; ++i) {
                assertEquals(4L * i, it.nextByteOffset());
            }
            assertFalse(it.hasNext());
        }
    }

    @Test
    void testReshape() {
        try (var arena = Arena.ofShared()) {
            MemorySegment memorySegment = arena.allocate(DataType.F32.byteSize() * Shape.of(3, 5).totalNumberOfElements());
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
            var memoryView = MemoryViewFactory.of(Shape.of(3, 5), DataType.F32, memory);

            assertTrue(memoryView.isContiguous());

            var it = OffsetIterator.create(memoryView);

            assertTrue(it.hasNext());
            long totalNumberOfElements = memoryView.shape().totalNumberOfElements();
            for (int i = 0; i < totalNumberOfElements; ++i) {
                assertEquals(4L * i, it.nextByteOffset());
            }
            assertFalse(it.hasNext());
        }
    }

}
