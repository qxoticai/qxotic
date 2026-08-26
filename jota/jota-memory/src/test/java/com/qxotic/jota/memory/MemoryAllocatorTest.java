package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Shape;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

public class MemoryAllocatorTest {

    static Stream<Supplier<MemoryAllocator<?>>> managedAllocatorProvider() {
        return Stream.of(
                MemoryDomains.booleans()::memoryAllocator,
                MemoryDomains.bytes()::memoryAllocator,
                MemoryDomains.shorts()::memoryAllocator,
                MemoryDomains.ints()::memoryAllocator,
                MemoryDomains.floats()::memoryAllocator,
                MemoryDomains.doubles()::memoryAllocator,
                MemoryDomains.longs()::memoryAllocator,
                () -> MemoryAllocators.ofArena(Arena.ofAuto()),
                () -> MemoryAllocators.newByteBuffer(true),
                () -> MemoryAllocators.newByteBuffer(false));
    }

    static Stream<Supplier<MemoryAllocator<?>>> javaArrayAllocatorProvider() {
        return Stream.of(
                MemoryDomains.booleans()::memoryAllocator,
                MemoryDomains.bytes()::memoryAllocator,
                MemoryDomains.shorts()::memoryAllocator,
                MemoryDomains.ints()::memoryAllocator,
                MemoryDomains.floats()::memoryAllocator,
                MemoryDomains.doubles()::memoryAllocator,
                MemoryDomains.longs()::memoryAllocator);
    }

    static Stream<Supplier<MemoryAllocator<?>>> byteBufferAllocatorProvider() {
        return Stream.of(
                () -> MemoryAllocators.newByteBuffer(false),
                () -> MemoryAllocators.newByteBuffer(true));
    }

    static Stream<Supplier<ScopedMemoryAllocator<?>>> scopedAllocatorProvider() {
        return Stream.of(MemoryAllocators::newScopedArena, MemoryAllocators::newScopedArena);
    }

    static Stream<Supplier<ScopedArena<?>>> scopedArenaAllocatorProvider() {
        return Stream.of(MemoryAllocators::newScopedArena);
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
        MemoryAllocator<int[]> allocator = MemoryDomains.ints().memoryAllocator();

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
        MemoryAllocator<boolean[]> allocator = MemoryDomains.booleans().memoryAllocator();
        assertTrue(allocator.supportsDataType(DataType.BOOL));
        assertFalse(allocator.supportsDataType(DataType.I8));
    }

    /** Regression: alignedSlice() rounded the limit down, so (100, 64) came back as 64 bytes. */
    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void byteBufferAllocatorHonorsSizeAndAlignment(boolean direct) {
        var allocator = MemoryAllocators.newByteBuffer(direct);
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
