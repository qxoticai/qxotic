package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.nio.ByteOrder;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
                () -> MemoryAllocatorFactory.ofByteBuffer(false, ByteOrder.nativeOrder())
        );
    }

    static Stream<Supplier<ScopedMemoryAllocator<?>>> scopedAllocatorProvider() {
        return Stream.of(
                MemoryAllocatorFactory::ofPanama,
                MemoryAllocatorFactory::newPanamaArena
        );
    }

    static Stream<Supplier<ScopedMemoryAllocatorArena<?>>> scopedArenaAllocatorProvider() {
        return Stream.of(
                MemoryAllocatorFactory::newPanamaArena
        );
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
}
