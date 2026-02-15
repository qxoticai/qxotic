package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.Test;

class GatherTraceLazyTest {

    @SuppressWarnings("unchecked")
    private static final MemoryDomain<MemorySegment> DOMAIN =
            (MemoryDomain<MemorySegment>) Environment.current().nativeRuntime().memoryDomain();

    @Test
    void tracedGatherMaterializesThroughLowering() {
        Tensor table = Tensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, Shape.of(3, 3));
        Tensor indices = Tensor.of(new int[] {2, 0}, Shape.of(2));

        Tensor traced = Tracer.trace(List.of(table, indices), ts -> ts.get(0).gather(ts.get(1), 0));
        assertTrue(traced.isLazy());

        MemoryView<?> view = traced.materialize();
        assertEquals(DataType.FP32, view.dataType());
        assertEquals(Shape.of(2, 3), view.shape());

        float[] expected = {6f, 7f, 8f, 0f, 1f, 2f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return DOMAIN.directAccess().readFloat(typedView.memory(), offset);
    }
}
