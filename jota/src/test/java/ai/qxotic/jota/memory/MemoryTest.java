package ai.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import ai.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class MemoryTest {

    @Test
    void testAllocate() {
        var allocator = MemoryAllocatorFactory.ofPanama();
        DataType dataType = DataType.FP32;
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

    //    @Test
    //    void testMemoryView() {
    //        try (var arena = Arena.ofShared()) {
    //            MemorySegment memorySegment = arena.allocate(Float.BYTES * 16);
    //            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
    //
    //
    //            var memoryView = MemoryViewFactory.of(DataType.F32, memory,
    // Layout.rowMajor(Shape.of(16)));
    //            assertTrue(memoryView.isContiguous());
    //
    //            var it = OffsetIterator.create(memoryView);
    //
    //            assertTrue(it.hasNext());
    //            for (int i = 0; i < 16; ++i) {
    //                assertEquals(4L * i, it.nextByteOffset());
    //            }
    //            assertFalse(it.hasNext());
    //        }
    //    }
    //
    //    @Test
    //    void testMemoryViewMatrix() {
    //        try (var arena = Arena.ofShared()) {
    //            MemorySegment memorySegment = arena.allocate(Float.BYTES * (3 * 5));
    //            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
    //            var memoryView = MemoryViewFactory.of(DataType.F32, memory,
    // Layout.rowMajor(Shape.of(3, 5)));
    //
    //            assertTrue(memoryView.isContiguous());
    //
    //            var it = OffsetIterator.create(memoryView);
    //
    //            assertTrue(it.hasNext());
    //            long totalNumberOfElements = memoryView.shape().size();
    //            for (int i = 0; i < totalNumberOfElements; ++i) {
    //                assertEquals(4L * i, it.nextByteOffset());
    //            }
    //            assertFalse(it.hasNext());
    //        }
    //    }
    //
    //    @Test
    //    void testReshape() {
    //        try (var arena = Arena.ofShared()) {
    //            MemorySegment memorySegment = arena.allocate(DataType.F32.byteSizeFor(Shape.of(3,
    // 5)));
    //            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
    //            var memoryView = MemoryViewFactory.of(DataType.F32, memory, Layout.rowMajor(3,
    // 5));
    //
    //            assertTrue(memoryView.isContiguous());
    //
    //            var it = OffsetIterator.create(memoryView);
    //
    //            assertTrue(it.hasNext());
    //            long totalNumberOfElements = memoryView.shape().size();
    //            for (int i = 0; i < totalNumberOfElements; ++i) {
    //                assertEquals(4L * i, it.nextByteOffset());
    //            }
    //            assertFalse(it.hasNext());
    //        }
    //    }

}
