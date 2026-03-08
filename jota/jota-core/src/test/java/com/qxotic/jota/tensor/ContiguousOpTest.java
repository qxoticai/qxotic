package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ContiguousOpTest {

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
        domain = Environment.current().nativeMemoryDomain();
    }

    @Test
    void contiguousReturnsSameTensorForContiguousInput() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        Tensor result = input.contiguous();
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
        Layout expected = Layout.rowMajor(view.layout().shape());
        assertEquals(expected, materialized.layout());
    }

    @Test
    void contiguousMaterializesStridedView() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 6).view(Shape.of(2, 3));
        MemoryView<MemorySegment> transposed = view.transpose(0, 1);
        Tensor input = Tensor.of(transposed);
        Tensor output = input.contiguous();
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
            Tensor output = input.contiguous();
            MemoryView<?> materialized = output.materialize();
            assertTrue(materialized.isContiguous(), "Expected contiguous for " + dataType);
            assertValuesEqual(Tensor.of(transposed), output, dataType);
        }
    }

    private void assertValuesEqual(Tensor source, Tensor target, DataType dataType) {
        long size = source.shape().size();
        for (int i = 0; i < size; i++) {
            Object srcValue = TensorTestReads.readValue(source, i, dataType);
            Object dstValue = TensorTestReads.readValue(target, i, dataType);
            assertEquals(srcValue, dstValue, "Mismatch for " + dataType + " at " + i);
        }
    }
}
