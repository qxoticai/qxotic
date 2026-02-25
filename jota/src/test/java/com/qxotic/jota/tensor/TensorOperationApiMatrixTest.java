package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.util.List;
import org.junit.jupiter.api.Test;

class TensorOperationApiMatrixTest {

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

    @Test
    void castSupportsAllPrimitiveTypePairs() {
        for (DataType source : PRIMITIVE_TYPES) {
            for (DataType target : PRIMITIVE_TYPES) {
                Tensor input = Tensor.scalar(1L, source);
                MemoryView<?> out = input.cast(target).materialize();
                assertEquals(target, out.dataType());
            }
        }
    }

    @Test
    void floatingUnaryOpsRejectNonFloatingTypes() {
        for (DataType type : PRIMITIVE_TYPES) {
            Tensor input = Tensor.scalar(2L, type);
            boolean floating = type.isFloatingPoint();
            if (floating) {
                assertEquals(type, input.sqrt().materialize().dataType());
                assertEquals(type, input.sin().materialize().dataType());
                assertEquals(type, input.cos().materialize().dataType());
                assertEquals(type, input.tanh().materialize().dataType());
                assertEquals(type, input.reciprocal().materialize().dataType());
            } else {
                assertThrows(IllegalArgumentException.class, input::sqrt);
                assertThrows(IllegalArgumentException.class, input::sin);
                assertThrows(IllegalArgumentException.class, input::cos);
                assertThrows(IllegalArgumentException.class, input::tanh);
                assertThrows(IllegalArgumentException.class, input::reciprocal);
            }
        }
    }

    @Test
    void reciprocalTargetTypeMustBeFloatingPoint() {
        Tensor base = Tensor.scalar(2L, DataType.I32);
        assertThrows(IllegalArgumentException.class, () -> base.reciprocal(DataType.I32));
        assertEquals(DataType.FP64, base.reciprocal(DataType.FP64).materialize().dataType());
    }

    @Test
    void logicalNotRequiresBool() {
        Tensor boolTensor = Tensor.scalar(1L, DataType.BOOL);
        assertEquals(DataType.BOOL, boolTensor.logicalNot().materialize().dataType());

        for (DataType type : PRIMITIVE_TYPES) {
            if (type == DataType.BOOL) {
                continue;
            }
            Tensor input = Tensor.scalar(1L, type);
            assertThrows(IllegalArgumentException.class, input::logicalNot);
        }
    }

    @Test
    void matmulValidatesRankShapeAndDtype() {
        Tensor left = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor right = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4));
        MemoryView<?> out = left.matmul(right).materialize();
        assertEquals(Shape.of(2, 4), out.shape());
        assertEquals(DataType.FP32, out.dataType());

        Tensor wrongRank = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 2, 3));
        assertThrows(IllegalArgumentException.class, () -> wrongRank.matmul(right));

        Tensor badInner = Tensor.iota(8, DataType.FP32).view(Shape.of(2, 4));
        assertThrows(IllegalArgumentException.class, () -> badInner.matmul(right));

        Tensor intRight = Tensor.iota(12, DataType.I32).view(Shape.of(3, 4));
        assertThrows(IllegalArgumentException.class, () -> left.matmul(intRight));
    }

    @Test
    void batchedMatmulValidatesRankShapeAndDtype() {
        Tensor left = Tensor.iota(24, DataType.FP32).view(Shape.of(2, 3, 4));
        Tensor right = Tensor.iota(40, DataType.FP32).view(Shape.of(2, 4, 5));
        MemoryView<?> out = left.batchedMatmul(right).materialize();
        assertEquals(Shape.of(2, 3, 5), out.shape());
        assertEquals(DataType.FP32, out.dataType());

        Tensor wrongRank = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4));
        assertThrows(IllegalArgumentException.class, () -> wrongRank.batchedMatmul(right));

        Tensor badBatch = Tensor.iota(20, DataType.FP32).view(Shape.of(1, 4, 5));
        assertThrows(IllegalArgumentException.class, () -> left.batchedMatmul(badBatch));

        Tensor intRight = Tensor.iota(40, DataType.I32).view(Shape.of(2, 4, 5));
        assertThrows(IllegalArgumentException.class, () -> left.batchedMatmul(intRight));
    }

    @Test
    void gatherValidatesIndexTypeAndAxis() {
        Tensor table = Tensor.iota(15, DataType.FP32).view(Shape.of(5, 3));
        Tensor indices = Tensor.iota(4, DataType.I32).view(Shape.of(2, 2));
        MemoryView<?> out = table.gather(indices, 0).materialize();
        assertEquals(Shape.of(2, 2, 3), out.shape());

        Tensor badIndices = Tensor.iota(4, DataType.FP32).view(Shape.of(2, 2));
        assertThrows(IllegalArgumentException.class, () -> table.gather(badIndices, 0));
        assertThrows(IllegalArgumentException.class, () -> table.gather(indices, 2));
    }

    @Test
    void viewAndReshapeValidateLayoutContracts() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        MemoryView<?> reshaped = input.reshape(Shape.of(3, 2)).materialize();
        assertEquals(Shape.of(3, 2), reshaped.shape());

        assertThrows(IllegalArgumentException.class, () -> input.reshape(Shape.of(4, 2)));

        MemoryView<?> transposed = input.transpose(0, 1).materialize();
        assertEquals(Shape.of(3, 2), transposed.shape());

        assertThrows(IllegalArgumentException.class, () -> input.permute(0));
    }
}
