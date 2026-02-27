package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class BooleanTensorTest {

    @Test
    void createsBoolTensorFromArray() {
        Tensor tensor = Tensor.of(new boolean[] {true, false, true});
        MemoryView<?> output = tensor.materialize();

        assertEquals(DataType.BOOL, output.dataType());
        assertEquals(Shape.flat(3), output.shape());
        assertEquals(1, TensorTestReads.readByte(tensor, 0));
        assertEquals(0, TensorTestReads.readByte(tensor, 1));
        assertEquals(1, TensorTestReads.readByte(tensor, 2));
    }

    @Test
    void rejectsShapeMismatch() {
        boolean[] data = new boolean[] {true, false};
        assertThrows(IllegalArgumentException.class, () -> Tensor.of(data, Shape.flat(3)));
    }
}
