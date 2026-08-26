package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class MemoryCopyTest extends AbstractMemoryTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setupDomain() {
        domain = MemoryDomains.of(MemoryAllocators.newScopedArena());
    }

    @Test
    void copiesStridedViewsWithinDomainForAllTypes() {
        for (DataType dataType : PRIMITIVE_DATA_TYPES) {
            MemoryView<MemorySegment> base = range(dataType, Shape.of(2, 3));
            MemoryView<MemorySegment> src = base.transpose(0, 1);
            MemoryView<MemorySegment> dst =
                    MemoryView.of(
                            domain.memoryAllocator().allocateMemory(dataType, src.shape()),
                            dataType,
                            src.layout());

            domain.copy(src, dst);
            assertCopyMatches(src, dst, dataType);
        }
    }

    @Test
    void domainCopiesAcrossViewsForAllTypes() {
        for (DataType dataType : PRIMITIVE_DATA_TYPES) {
            MemoryView<MemorySegment> src = range(dataType, Shape.of(2, 2));
            MemoryView<MemorySegment> dst =
                    MemoryView.of(
                            domain.memoryAllocator().allocateMemory(dataType, src.shape()),
                            dataType,
                            src.layout());
            MemoryDomain.copy(domain, src, domain, dst);
            assertCopyMatches(src, dst, dataType);
        }
    }

    @ParameterizedTest(name = "{0}: {1} -> {2}", autoCloseArguments = false)
    @MethodSource("crossDomainCases")
    <S, D> void copiesContiguousViewsBetweenDomains(
            DataType dataType, MemoryDomain<S> srcDomain, MemoryDomain<D> dstDomain) {
        Shape shape = Shape.of(2, 3);
        MemoryView<S> src = range(srcDomain, dataType, shape);
        MemoryView<D> dst = allocate(dstDomain, dataType, shape);

        MemoryDomain.copy(srcDomain, src, dstDomain, dst);

        assertCopyMatches(srcDomain, src, dstDomain, dst, dataType);
    }

    @ParameterizedTest(name = "{0} strided: {1} -> {2}", autoCloseArguments = false)
    @MethodSource("crossDomainCases")
    <S, D> void copiesStridedViewsBetweenDomains(
            DataType dataType, MemoryDomain<S> srcDomain, MemoryDomain<D> dstDomain) {
        Shape baseShape = Shape.of(2, 3);
        MemoryView<S> src = range(srcDomain, dataType, baseShape).transpose(0, 1);
        MemoryView<D> dst = allocate(dstDomain, dataType, baseShape).transpose(0, 1);

        MemoryDomain.copy(srcDomain, src, dstDomain, dst);

        assertCopyMatches(srcDomain, src, dstDomain, dst, dataType);
    }

    @Test
    void usesOnlyTheSameOperationsInstanceForDirectCopies() {
        MemoryOperations<byte[]> sharedOperations = forwardingByteOperations();
        TestByteDomain left = new TestByteDomain(sharedOperations);
        TestByteDomain right = new TestByteDomain(sharedOperations);

        copyBytes(left, right);

        assertEquals(1, left.directCopies);
    }

    @Test
    void doesNotUseTheOperationsClassAsCompatibility() {
        MemoryOperations<byte[]> leftOperations = forwardingByteOperations();
        MemoryOperations<byte[]> rightOperations = forwardingByteOperations();
        assertSame(leftOperations.getClass(), rightOperations.getClass());

        TestByteDomain left = new TestByteDomain(leftOperations);
        TestByteDomain right = new TestByteDomain(rightOperations);

        copyBytes(left, right);

        assertEquals(0, left.directCopies);
    }

    static Stream<Arguments> crossDomainCases() {
        return Stream.of(
                        cases(DataType.BOOL, domainsSupportingBool().toList()),
                        cases(DataType.I8, domainsSupportingI8().toList()),
                        cases(DataType.I16, domainsSupportingI16().toList()),
                        cases(DataType.I32, domainsSupportingI32().toList()),
                        cases(DataType.I64, domainsSupportingI64().toList()),
                        cases(DataType.FP16, domainsSupportingI16().toList()),
                        cases(DataType.BF16, domainsSupportingI16().toList()),
                        cases(DataType.FP32, domainsSupportingF32().toList()),
                        cases(DataType.FP64, domainsSupportingF64().toList()))
                .flatMap(cases -> cases);
    }

    private static Stream<Arguments> cases(
            DataType dataType, java.util.List<MemoryDomain<?>> domains) {
        return domains.stream()
                .flatMap(src -> domains.stream().map(dst -> Arguments.of(dataType, src, dst)));
    }

    private static void copyBytes(TestByteDomain srcDomain, TestByteDomain dstDomain) {
        byte[] values = {1, 2, 3, 4};
        MemoryView<byte[]> src = MemoryView.rowMajor(Memories.of(values), DataType.I8, Shape.of(4));
        MemoryView<byte[]> dst = allocate(dstDomain, DataType.I8, Shape.of(4));

        MemoryDomain.copy(srcDomain, src, dstDomain, dst);

        assertArrayEquals(values, dst.memory().base());
    }

    @SuppressWarnings("unchecked")
    private static MemoryOperations<byte[]> forwardingByteOperations() {
        MemoryOperations<byte[]> delegate = MemoryDomains.bytes().memoryOperations();
        return (MemoryOperations<byte[]>)
                Proxy.newProxyInstance(
                        MemoryOperations.class.getClassLoader(),
                        new Class<?>[] {MemoryOperations.class},
                        (proxy, method, args) -> {
                            try {
                                return method.invoke(delegate, args);
                            } catch (InvocationTargetException failure) {
                                throw failure.getCause();
                            }
                        });
    }

    private static final class TestByteDomain implements MemoryDomain<byte[]> {

        private final MemoryOperations<byte[]> operations;
        private int directCopies;

        private TestByteDomain(MemoryOperations<byte[]> operations) {
            this.operations = operations;
        }

        @Override
        public Device device() {
            return MemoryDomains.bytes().device();
        }

        @Override
        public MemoryAllocator<byte[]> memoryAllocator() {
            return MemoryDomains.bytes().memoryAllocator();
        }

        @Override
        public MemoryAccess<byte[]> directAccess() {
            return MemoryDomains.bytes().directAccess();
        }

        @Override
        public MemoryOperations<byte[]> memoryOperations() {
            return operations;
        }

        @Override
        public void copy(MemoryView<byte[]> src, MemoryView<byte[]> dst) {
            directCopies++;
            MemoryDomain.super.copy(src, dst);
        }

        @Override
        public void close() {}
    }

    private MemoryView<MemorySegment> range(DataType dataType, Shape shape) {
        return range(domain, dataType, shape);
    }

    private static <B> MemoryView<B> range(MemoryDomain<B> domain, DataType dataType, Shape shape) {
        MemoryView<B> flat =
                dataType == DataType.BOOL
                        ? MemoryViews.full(domain, dataType, shape.size(), 1)
                        : MemoryViews.arange(domain, dataType, shape.size());
        return flat.view(shape);
    }

    private static <B> MemoryView<B> allocate(
            MemoryDomain<B> domain, DataType dataType, Shape shape) {
        return MemoryView.of(
                domain.memoryAllocator().allocateMemory(dataType, shape),
                dataType,
                Layout.rowMajor(shape));
    }

    private static <S, D> void assertCopyMatches(
            MemoryDomain<S> srcDomain,
            MemoryView<S> src,
            MemoryDomain<D> dstDomain,
            MemoryView<D> dst,
            DataType dataType) {
        long size = src.shape().size();
        for (long i = 0; i < size; i++) {
            Object expected =
                    readValue(
                            srcDomain.directAccess(),
                            src.memory(),
                            Indexing.linearToOffset(src, i),
                            dataType);
            Object actual =
                    readValue(
                            dstDomain.directAccess(),
                            dst.memory(),
                            Indexing.linearToOffset(dst, i),
                            dataType);
            assertEquals(expected, actual, "Mismatch at element " + i);
        }
    }

    private static <B> Object readValue(
            MemoryAccess<B> access, Memory<B> memory, long offset, DataType dataType) {
        if (dataType == DataType.BOOL || dataType == DataType.I8) {
            return access.readByte(memory, offset);
        }
        if (dataType == DataType.I16 || dataType == DataType.FP16 || dataType == DataType.BF16) {
            return access.readShort(memory, offset);
        }
        if (dataType == DataType.I32) {
            return access.readInt(memory, offset);
        }
        if (dataType == DataType.I64) {
            return access.readLong(memory, offset);
        }
        if (dataType == DataType.FP32) {
            return access.readFloat(memory, offset);
        }
        if (dataType == DataType.FP64) {
            return access.readDouble(memory, offset);
        }
        throw new IllegalStateException("Unsupported data type: " + dataType);
    }

    private void assertCopyMatches(MemoryView<?> src, MemoryView<?> dst, DataType dataType) {
        long size = src.shape().size();
        for (int i = 0; i < size; i++) {
            long srcOffset = Indexing.linearToOffset(src, i);
            long dstOffset = Indexing.linearToOffset(dst, i);
            Object srcValue = readValue((MemorySegment) src.memory().base(), srcOffset, dataType);
            Object dstValue = readValue((MemorySegment) dst.memory().base(), dstOffset, dataType);
            assertEquals(srcValue, dstValue, "Mismatch for dtype " + dataType + " at index " + i);
        }
    }

    private Object readValue(MemorySegment segment, long offset, DataType dataType) {
        if (dataType == DataType.BOOL || dataType == DataType.I8) {
            return segment.get(ValueLayout.JAVA_BYTE, offset);
        }
        if (dataType == DataType.I16 || dataType == DataType.FP16 || dataType == DataType.BF16) {
            return segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
        }
        if (dataType == DataType.I32) {
            return segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
        }
        if (dataType == DataType.I64) {
            return segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
        }
        if (dataType == DataType.FP32) {
            return segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
        }
        if (dataType == DataType.FP64) {
            return segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset);
        }
        throw new IllegalStateException("Unsupported data type: " + dataType);
    }
}
