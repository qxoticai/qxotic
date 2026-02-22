package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ContiguousOpsTest {

    private static final List<DataType> PRIMITIVE_TYPES =
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

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = DomainFactory.ofMemorySegment();
    }

    @Test
    void contiguousReturnsSameTensorForContiguousInput() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        Tensor result = TensorOpsContext.with(new EagerTensorOps(domain), input::contiguous);
        assertSame(input, result);
    }

    @Test
    void contiguousMaterializesStridedView() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 6).view(Shape.of(2, 3));
        MemoryView<MemorySegment> transposed = view.transpose(0, 1);
        Tensor input = Tensor.of(transposed);
        Tensor output = TensorOpsContext.with(new EagerTensorOps(domain), input::contiguous);
        MemoryView<?> materialized = output.materialize();
        assertTrue(materialized.isContiguous());
        Layout expected = Layout.rowMajor(transposed.layout().shape());
        assertEquals(expected, materialized.layout());
    }

    @Test
    void contiguousWorksAcrossPrimitiveTypes() {
        Shape shape = Shape.of(2, 2);
        for (DataType dataType : PRIMITIVE_TYPES) {
            MemoryView<MemorySegment> view =
                    dataType == DataType.BOOL
                            ? MemoryHelpers.full(domain, dataType, shape.size(), 1).view(shape)
                            : MemoryHelpers.arange(domain, dataType, shape.size()).view(shape);
            MemoryView<MemorySegment> transposed = view.transpose(0, 1);
            Tensor input = Tensor.of(transposed);
            Tensor output = TensorOpsContext.with(new EagerTensorOps(domain), input::contiguous);
            MemoryView<?> materialized = output.materialize();
            assertTrue(materialized.isContiguous(), "Expected contiguous for " + dataType);
            assertValuesEqual(transposed, materialized, dataType);
        }
    }

    private void assertValuesEqual(
            MemoryView<MemorySegment> source, MemoryView<?> target, DataType dataType) {
        MemoryAccess<MemorySegment> access = domain.directAccess();
        long size = source.shape().size();
        for (int i = 0; i < size; i++) {
            long srcOffset = Indexing.linearToOffset(source, i);
            long dstOffset = Indexing.linearToOffset(target, i);
            @SuppressWarnings("unchecked")
            Memory<MemorySegment> srcMemory = source.memory();
            Object srcValue = readValue(access, srcMemory, srcOffset, dataType);
            @SuppressWarnings("unchecked")
            Memory<MemorySegment> dstMemory = (Memory<MemorySegment>) target.memory();
            Object dstValue = readValue(access, dstMemory, dstOffset, dataType);
            assertEquals(srcValue, dstValue, "Mismatch for " + dataType + " at " + i);
        }
    }

    private Object readValue(
            MemoryAccess<MemorySegment> access,
            Memory<MemorySegment> memory,
            long offset,
            DataType dataType) {
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
        throw new IllegalStateException("Unsupported type: " + dataType);
    }
}
