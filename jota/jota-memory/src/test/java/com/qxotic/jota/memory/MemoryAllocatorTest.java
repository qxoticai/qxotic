package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.internal.MemoryAllocatorFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

public class MemoryAllocatorTest {

    static Stream<Supplier<MemoryAllocator<?>>> managedAllocatorProvider() {
        return Stream.of(
                MemoryAllocatorFactory::ofBooleans,
                MemoryAllocatorFactory::ofBytes,
                MemoryAllocatorFactory::ofShorts,
                MemoryAllocatorFactory::ofInts,
                MemoryAllocatorFactory::ofFloats,
                MemoryAllocatorFactory::ofDoubles,
                MemoryAllocatorFactory::ofLongs,
                MemoryAllocatorFactory::newPanamaAuto,
                () -> MemoryAllocatorFactory.ofByteBuffer(true, ByteOrder.nativeOrder()),
                () -> MemoryAllocatorFactory.ofByteBuffer(false, ByteOrder.nativeOrder()));
    }

    static Stream<Supplier<MemoryAllocator<?>>> javaArrayAllocatorProvider() {
        return Stream.of(
                MemoryAllocatorFactory::ofBooleans,
                MemoryAllocatorFactory::ofBytes,
                MemoryAllocatorFactory::ofShorts,
                MemoryAllocatorFactory::ofInts,
                MemoryAllocatorFactory::ofFloats,
                MemoryAllocatorFactory::ofDoubles,
                MemoryAllocatorFactory::ofLongs);
    }

    static Stream<Supplier<MemoryAllocator<?>>> byteBufferAllocatorProvider() {
        return Stream.of(
                () -> MemoryAllocatorFactory.ofByteBuffer(false, ByteOrder.nativeOrder()),
                () -> MemoryAllocatorFactory.ofByteBuffer(true, ByteOrder.nativeOrder()));
    }

    static Stream<Supplier<ScopedMemoryAllocator<?>>> scopedAllocatorProvider() {
        return Stream.of(MemoryAllocatorFactory::ofPanama, MemoryAllocatorFactory::newPanamaArena);
    }

    static Stream<Supplier<ScopedMemoryAllocatorArena<?>>> scopedArenaAllocatorProvider() {
        return Stream.of(MemoryAllocatorFactory::newPanamaArena);
    }

    private static final DataType[] DATA_TYPES = {
        DataType.BOOL,
        DataType.I8,
        DataType.I16,
        DataType.I32,
        DataType.I64,
        DataType.FP16,
        DataType.BF16,
        DataType.FP32,
        DataType.FP64
    };

    @ParameterizedTest
    @MethodSource("managedAllocatorProvider")
    <B> void testAllocateScalar(Supplier<MemoryAllocator<B>> memoryAllocatorSupplier) {
        var allocator = memoryAllocatorSupplier.get();
        for (DataType dataType : DATA_TYPES) {
            if (allocator.supportsDataType(dataType)) {
                Memory<B> memory = allocator.allocateMemory(dataType, Shape.scalar());
                assertEquals(dataType.byteSize(), memory.byteSize());
            }
        }
    }

    @ParameterizedTest
    @MethodSource("javaArrayAllocatorProvider")
    <B> void testJavaArrayAllocatorsUseJavaDevice(
            Supplier<MemoryAllocator<B>> memoryAllocatorSupplier) {
        assertTrue(memoryAllocatorSupplier.get().device().belongsTo(DeviceType.JAVA));
    }

    @ParameterizedTest
    @MethodSource("byteBufferAllocatorProvider")
    <B> void testByteBufferAllocatorsUseJavaDevice(
            Supplier<MemoryAllocator<B>> memoryAllocatorSupplier) {
        assertTrue(memoryAllocatorSupplier.get().device().belongsTo(DeviceType.JAVA));
    }

    @Test
    void arrayAllocatorsValidateSizeAndAlignment() {
        MemoryAllocator<int[]> allocator = MemoryAllocatorFactory.ofInts();

        assertDoesNotThrow(() -> allocator.allocateMemory(Integer.BYTES, Integer.BYTES));
        assertDoesNotThrow(() -> allocator.allocateMemory(Integer.BYTES, Short.BYTES));
        assertThrows(
                IllegalArgumentException.class, () -> allocator.allocateMemory(DataType.I8, 1));
        assertThrows(IllegalArgumentException.class, () -> allocator.allocateMemory(-1));
        assertThrows(IllegalArgumentException.class, () -> allocator.allocateMemory(1));
        assertThrows(
                IllegalArgumentException.class, () -> allocator.allocateMemory(Integer.BYTES, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> allocator.allocateMemory(Integer.BYTES, Long.BYTES));
    }

    @Test
    void booleanAllocatorOnlySupportsBooleanValues() {
        MemoryAllocator<boolean[]> allocator = MemoryAllocatorFactory.ofBooleans();
        assertTrue(allocator.supportsDataType(DataType.BOOL));
        assertFalse(allocator.supportsDataType(DataType.I8));
    }

    /** Regression: alignedSlice() rounded the limit down, so (100, 64) came back as 64 bytes. */
    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void byteBufferAllocatorHonorsSizeAndAlignment(boolean direct) {
        var allocator = MemoryAllocatorFactory.ofByteBuffer(direct);
        int maxAlign = direct ? 4096 : 1;
        for (int size : new int[] {1, 10, 100, 4097}) {
            for (int align = 1; align <= maxAlign; align *= 2) {
                Memory<ByteBuffer> memory = allocator.allocateMemory(size, align);
                assertEquals(size, memory.byteSize(), "size=" + size + " align=" + align);
                assertEquals(0, memory.base().alignmentOffset(0, align), "align=" + align);
            }
        }
        if (!direct) {
            assertThrows(IllegalArgumentException.class, () -> allocator.allocateMemory(100, 2));
        }
    }
}
