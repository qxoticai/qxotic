package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class GatherTraceLazyTest {

    @Test
    void tracedGatherMaterializesThroughLowering() {
        Assumptions.assumeTrue(
                ConfiguredTestDevice.resolve().belongsTo(DeviceType.PANAMA),
                "Gather traced lowering currently panama-only in runtime-agnostic lane");
        Tensor table = Tensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, Shape.of(3, 3));
        Tensor indices = Tensor.of(new int[] {2, 0}, Shape.of(2));

        Tensor traced = Tracer.trace(List.of(table, indices), ts -> ts.get(0).gather(ts.get(1), 0));
        assertTrue(TensorTestInternals.isLazy(traced));

        MemoryView<?> view = traced.materialize();
        assertEquals(DataType.FP32, view.dataType());
        assertEquals(Shape.of(2, 3), view.shape());

        float[] expected = {6f, 7f, 8f, 0f, 1f, 2f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], TensorTestReads.readFloat(traced, i), 0.0001f);
        }
    }
}
