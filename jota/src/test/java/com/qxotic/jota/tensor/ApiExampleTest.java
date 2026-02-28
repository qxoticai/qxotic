package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ApiExampleTest {

    @Test
    void basicFunctionRunsAndMaterializes() {
        MemoryView<?> view =
                MemoryHelpers.arange(DomainFactory.ofMemorySegment(), DataType.FP32, 6)
                        .view(Shape.of(2, 3));
        Tensor x = Tensor.of(view);
        Tensor y = x.add(x).sqrt();

        assertTrue(TensorTestInternals.isLazy(y));
        assertFalse(TensorTestInternals.isMaterialized(y));

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(2, 3), output.shape());
        assertEquals(0.0f, TensorTestReads.readFloat(y, 0), 0.0001f);
        assertEquals((float) Math.sqrt(10.0), TensorTestReads.readFloat(y, 5), 0.0001f);
    }
}
