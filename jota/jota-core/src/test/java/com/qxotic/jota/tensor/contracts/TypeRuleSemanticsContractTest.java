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
class TypeRuleSemanticsContractTest {

    @Test
    void scalarOverloadArithmeticProducesExpectedValues() {
        Tensor intBase = Tensor.iota(3, DataType.I32);
        intBase.add(2).materialize();
        intBase.add(2L).materialize();
        assertEquals(2, readInt(intBase.add(2), 0));
        assertEquals(3, readInt(intBase.add(2), 1));
        assertEquals(2, readInt(intBase.add(2L), 0));

        intBase.subtract(1).materialize();
        intBase.multiply(2).materialize();
        assertEquals(-1, readInt(intBase.subtract(1), 0));
        assertEquals(0, readInt(intBase.subtract(1), 1));
        assertEquals(4, readInt(intBase.multiply(2), 2));

        Tensor fpBase = Tensor.iota(3, DataType.FP32);
        fpBase.add(2.0f).materialize();
        assertEquals(2.0f, readFloat(fpBase.add(2.0f), 0), 1e-4f);
        fpBase.divide(2.0f).materialize();
        assertEquals(0.5f, readFloat(fpBase.divide(2.0f), 1), 1e-4f);

        Tensor fp64Base = Tensor.iota(3, DataType.FP64);
        fp64Base.add(2.0d).materialize();
        assertEquals(2.0d, readDouble(fp64Base.add(2.0d), 0), 1e-8);

        assertThrows(IllegalArgumentException.class, () -> fpBase.add(2).materialize());
    }

    @Test
    void lazyArithmeticPromotesWithStrictTypeRules() {
        Tensor left = Tensor.full(2L, DataType.I16, Shape.of(2));
        Tensor right = Tensor.full(1.5d, DataType.FP32, Shape.of(2));

        Tensor result = left.add(right);
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(3.5f, readFloat(result, 0), 1e-4f);
        assertEquals(3.5f, readFloat(result, 1), 1e-4f);

        Tensor i64 = Tensor.full(1L, DataType.I64, Shape.of(2));
        Tensor fp32 = Tensor.full(1.0f, DataType.FP32, Shape.of(2));
        assertThrows(IllegalArgumentException.class, () -> i64.add(fp32).materialize());
    }

    @Test
    void bitwiseRequiresSameDtype() {
        Tensor left = Tensor.full(3L, DataType.I16, Shape.of(2));
        Tensor right = Tensor.full(1L, DataType.I32, Shape.of(2));

        assertThrows(IllegalArgumentException.class, () -> left.bitwiseAnd(right));
    }

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }

    private static int readInt(Tensor tensor, long linearIndex) {
        return ((Number) TensorTestReads.readValue(tensor, linearIndex, DataType.I32)).intValue();
    }

    private static double readDouble(Tensor tensor, long linearIndex) {
        return ((Number) TensorTestReads.readValue(tensor, linearIndex, DataType.FP64))
                .doubleValue();
    }
}
