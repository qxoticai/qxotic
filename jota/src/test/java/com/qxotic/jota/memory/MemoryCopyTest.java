package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.*;
import com.qxotic.jota.memory.impl.DomainFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MemoryCopyTest extends AbstractMemoryTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setupDomain() {
        domain = DomainFactory.ofMemorySegment();
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

    private MemoryView<MemorySegment> range(DataType dataType, Shape shape) {
        if (dataType == DataType.BOOL) {
            return MemoryHelpers.full(domain, dataType, shape.size(), 1).view(shape);
        }
        return MemoryHelpers.arange(domain, dataType, shape.size()).view(shape);
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
