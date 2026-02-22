package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class AddTest {

    @Test
    void canary() {
        Shape shape = Shape.of(2, 3);
        Tensor right = Tensor.iota(shape.size(), DataType.FP32).view(shape);
        Tensor left = Tensor.of(new float[] {2f}).view(Shape.scalar());

        Tensor result = left.add(right);
        MemoryView<MemorySegment> output = (MemoryView<MemorySegment>) result.materialize();
        assertEquals(shape, output.shape());

        assertEquals(2.0f, readFloat(output, 0), 0.0001f);

        //                Tensor input = Tensor.of(new float[]{1.0f, 2.0f, 4.0f, 8.0f}, shape);
        //                MemoryView<?> output = result.materialize();
        //
        //                assertEquals(DataType.FP32, output.dataType());
        //                assertEquals(shape, output.shape());

    }

    private static float readFloat(MemoryView<MemorySegment> view, long linearIndex) {
        MemoryAccess<MemorySegment> access =
                Environment.current().nativeMemoryDomain().directAccess();
        long offset = Indexing.linearToOffset(view, linearIndex);
        return access.readFloat(view.memory(), offset);
    }
}
