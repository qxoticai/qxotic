package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class ApiExampleTest {

    @SuppressWarnings("unchecked")
    private static final MemoryDomain<MemorySegment> CONTEXT =
            (MemoryDomain<MemorySegment>) Environment.current().nativeRuntime().memoryDomain();

    @Test
    void basicFunctionRunsAndMaterializes() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(CONTEXT, DataType.FP32, 6).view(Shape.of(2, 3));
        Tensor x = Tensor.of(view);
        Tensor y = x.add(x).sqrt();

        assertTrue(y.isLazy());
        assertFalse(y.isMaterialized());

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(2, 3), output.shape());
        assertEquals(0.0f, readFloat(CONTEXT, output, 0), 0.0001f);
        assertEquals((float) Math.sqrt(10.0), readFloat(CONTEXT, output, 5), 0.0001f);
    }

    private static float readFloat(
            MemoryDomain<MemorySegment> domain, MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return domain.directAccess().readFloat(typedView.memory(), offset);
    }
}
