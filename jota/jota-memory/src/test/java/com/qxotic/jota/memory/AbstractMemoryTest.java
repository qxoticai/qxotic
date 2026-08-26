package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
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
                MemoryDomains::bytes,
                MemoryDomains::shorts,
                MemoryDomains::ints,
                MemoryDomains::longs,
                MemoryDomains::floats,
                MemoryDomains::doubles);
    }

    public static Stream<MemoryDomain<?>> nativeDomains() {
        return suppliedBy(
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true)),
                () -> MemoryDomains.of(MemoryAllocators.newScopedArena()));
    }

    public static Stream<MemoryDomain<?>> allDomains() {
        return Stream.concat(onHeapDomains(), nativeDomains());
    }

    public static Stream<MemoryDomain<?>> domainsSupportingF32() {
        return suppliedBy(
                MemoryDomains::bytes,
                MemoryDomains::floats,
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false)),
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true)),
                () -> MemoryDomains.of(MemoryAllocators.newScopedArena()));
    }

    public static Stream<MemoryDomain<?>> domainsSupportingF64() {
        return suppliedBy(
                MemoryDomains::bytes,
                MemoryDomains::doubles,
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false)),
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true)),
                () -> MemoryDomains.of(MemoryAllocators.newScopedArena()));
    }

    public static Stream<MemoryDomain<?>> domainsSupportingI8() {
        return suppliedBy(
                MemoryDomains::bytes,
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false)),
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true)),
                () -> MemoryDomains.of(MemoryAllocators.newScopedArena()));
    }

    public static Stream<MemoryDomain<?>> domainsSupportingI16() {
        return suppliedBy(
                MemoryDomains::bytes,
                MemoryDomains::shorts,
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false)),
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true)),
                () -> MemoryDomains.of(MemoryAllocators.newScopedArena()));
    }

    public static Stream<MemoryDomain<?>> domainsSupportingBF16() {
        return domainsSupportingI16();
    }

    public static Stream<MemoryDomain<?>> domainsSupportingI32() {
        return suppliedBy(
                MemoryDomains::ints,
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false)),
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true)),
                () -> MemoryDomains.of(MemoryAllocators.newScopedArena()));
    }

    public static Stream<MemoryDomain<?>> domainsSupportingI64() {
        return suppliedBy(
                MemoryDomains::longs,
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false)),
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true)),
                () -> MemoryDomains.of(MemoryAllocators.newScopedArena()));
    }

    public static Stream<MemoryDomain<?>> domainsSupportingBool() {
        return suppliedBy(
                MemoryDomains::booleans,
                MemoryDomains::bytes,
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false)),
                () -> MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true)),
                () -> MemoryDomains.of(MemoryAllocators.newScopedArena()));
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
