package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
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

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUpContext() {
        context = ContextFactory.ofMemorySegment();
    }

    @Test
    void contiguousReturnsSameTensorForContiguousInput() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(context, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        Tensor result = TensorOpsContext.with(new EagerTensorOps(context), input::contiguous);
        assertSame(input, result);
    }

    @Test
    void contiguousMaterializesStridedView() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(context, DataType.FP32, 6).view(Shape.of(2, 3));
        MemoryView<MemorySegment> transposed = view.transpose(0, 1);
        Tensor input = Tensor.of(transposed);
        Tensor output = TensorOpsContext.with(new EagerTensorOps(context), input::contiguous);
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
                            ? MemoryHelpers.full(context, dataType, shape.size(), 1).view(shape)
                            : MemoryHelpers.arange(context, dataType, shape.size()).view(shape);
            MemoryView<MemorySegment> transposed = view.transpose(0, 1);
            Tensor input = Tensor.of(transposed);
            Tensor output = TensorOpsContext.with(new EagerTensorOps(context), input::contiguous);
            MemoryView<?> materialized = output.materialize();
            assertTrue(materialized.isContiguous(), "Expected contiguous for " + dataType);
            assertValuesEqual(transposed, materialized, dataType);
        }
    }

    private void assertValuesEqual(
            MemoryView<MemorySegment> source, MemoryView<?> target, DataType dataType) {
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        long size = source.shape().size();
        for (int i = 0; i < size; i++) {
            long srcOffset = Indexing.linearToOffset(source, i);
            long dstOffset = Indexing.linearToOffset(target, i);
            @SuppressWarnings("unchecked")
            Memory<MemorySegment> srcMemory = source.memory();
            Object srcValue = readValue(access, srcMemory, srcOffset, dataType);
            @SuppressWarnings("unchecked")
            Memory<MemorySegment> dstMemory =
                    (Memory<MemorySegment>) target.memory();
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
