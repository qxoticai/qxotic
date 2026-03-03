package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Stream;

public abstract class AbstractMemoryTest {

    public static final List<DataType> PRIMITIVE_DATA_TYPES =
            List.of(
                    DataType.BOOL,
                    DataType.I8,
                    DataType.I16,
                    DataType.I32,
                    DataType.I64,
                    DataType.FP16,
                    DataType.BF16,
                    DataType.FP32,
                    DataType.FP64);

    static final List<DataType> INTEGRAL_DATA_TYPES =
            List.of(DataType.I8, DataType.I16, DataType.I32, DataType.I64);

    static final List<DataType> FLOATING_POINTS_DATA_TYPES =
            List.of(DataType.FP16, DataType.BF16, DataType.FP32, DataType.FP64);

    public static Stream<MemoryDomain<?>> onHeapDomains() {
        return suppliedBy(
                DomainFactory::ofBytes,
                DomainFactory::ofShorts,
                DomainFactory::ofInts,
                DomainFactory::ofLongs,
                DomainFactory::ofFloats,
                DomainFactory::ofDoubles);
    }

    public static Stream<MemoryDomain<?>> nativeDomains() {
        return suppliedBy(
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                DomainFactory::ofMemorySegment);
    }

    public static Stream<MemoryDomain<?>> allDomains() {
        return Stream.concat(onHeapDomains(), nativeDomains());
    }

    public static Stream<MemoryDomain<?>> domainsSupportingF32() {
        return suppliedBy(
                DomainFactory::ofBytes,
                DomainFactory::ofFloats,
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                DomainFactory::ofMemorySegment);
    }

    public static Stream<MemoryDomain<?>> domainsSupportingF64() {
        return suppliedBy(
                DomainFactory::ofBytes,
                DomainFactory::ofDoubles,
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                DomainFactory::ofMemorySegment);
    }

    public static Stream<MemoryDomain<?>> domainsSupportingI8() {
        return suppliedBy(
                DomainFactory::ofBytes,
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                DomainFactory::ofMemorySegment);
    }

    public static Stream<MemoryDomain<?>> domainsSupportingI16() {
        return suppliedBy(
                DomainFactory::ofBytes,
                DomainFactory::ofShorts,
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                DomainFactory::ofMemorySegment);
    }

    public static Stream<MemoryDomain<?>> domainsSupportingI32() {
        return suppliedBy(
                DomainFactory::ofInts,
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                DomainFactory::ofMemorySegment);
    }

    public static Stream<MemoryDomain<?>> domainsSupportingI64() {
        return suppliedBy(
                DomainFactory::ofLongs,
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                DomainFactory::ofMemorySegment);
    }

    public static Stream<MemoryDomain<?>> domainsSupportingBool() {
        return suppliedBy(
                DomainFactory::ofBooleans,
                DomainFactory::ofBytes,
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                DomainFactory::ofMemorySegment);
    }

    @SafeVarargs
    private static <T> Stream<T> suppliedBy(Supplier<T>... suppliers) {
        return Stream.of(suppliers).map(Supplier::get);
    }

    public static <B> float readFloat(
            MemoryAccess<B> memoryAccess, MemoryView<B> view, long... coords) {
        return memoryAccess.readFloat(view.memory(), Indexing.coordToOffset(view, coords));
    }

    public static <B> void writeFloat(
            MemoryAccess<B> memoryAccess, MemoryView<B> view, float floatValue, long... coords) {
        memoryAccess.writeFloat(view.memory(), Indexing.coordToOffset(view, coords), floatValue);
    }
}
