package com.qxotic.jota.tensor.contracts;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class WhereSemanticsContractTest {

    @Test
    void lazyWhereSupportsTrueScalarSemantics() {
        Tensor values = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor condition = values.greaterThan(Tensor.scalar(2.0f));
        Tensor scalarTrue = Tensor.scalar(10.0f);

        Tensor result = condition.where(scalarTrue, values);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(2, 3), output.shape());
        assertEquals(0.0f, readFloat(result, 0), 1e-4f);
        assertEquals(1.0f, readFloat(result, 1), 1e-4f);
        assertEquals(2.0f, readFloat(result, 2), 1e-4f);
        assertEquals(10.0f, readFloat(result, 3), 1e-4f);
        assertEquals(10.0f, readFloat(result, 4), 1e-4f);
        assertEquals(10.0f, readFloat(result, 5), 1e-4f);
    }

    @Test
    void whereRejectsMismatchedBranchTypes() {
        Tensor condition = Tensor.iota(3, DataType.I32).lessThan(Tensor.scalar(2L));
        Tensor fp = Tensor.iota(3, DataType.FP32);
        Tensor ints = Tensor.iota(3, DataType.I32);
        assertThrows(IllegalArgumentException.class, () -> condition.where(fp, ints));
    }

    @Test
    void anyAndAllReduceBoolGlobally() {
        Tensor condition =
                Tensor.iota(4, DataType.I32)
                        .view(Shape.of(2, 2))
                        .lessThan(Tensor.full(3L, DataType.I32, Shape.scalar()));

        Tensor any = condition.any().cast(DataType.I32);
        Tensor all = condition.all().cast(DataType.I32);
        MemoryView<?> anyView = any.materialize();
        MemoryView<?> allView = all.materialize();

        assertEquals(Shape.scalar(), anyView.shape());
        assertEquals(Shape.scalar(), allView.shape());
        assertEquals(1, readInt(any, 0));
        assertEquals(0, readInt(all, 0));
    }

    @Test
    void numericOpsRejectBoolOperands() {
        Tensor left = Tensor.full(1L, DataType.BOOL, Shape.of(2));
        Tensor right = Tensor.full(0L, DataType.BOOL, Shape.of(2));

        assertThrows(IllegalArgumentException.class, () -> left.add(right).materialize());

        Tensor compare = left.equal(right);
        assertEquals(DataType.BOOL, compare.dataType());
    }

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }

    private static int readInt(Tensor tensor, long linearIndex) {
        return ((Number) TensorTestReads.readValue(tensor, linearIndex, DataType.I32)).intValue();
    }
}
