package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class ArangeTensorTest {

    @SuppressWarnings("unchecked")
    private static final MemoryContext<MemorySegment> CONTEXT =
            (MemoryContext<MemorySegment>) Environment.current().nativeBackend().memoryContext();

    @Test
    void arangeDefaultsToI64() {
        Tensor range = Tensor.arange(6);

        assertTrue(range.isLazy());
        assertFalse(range.isMaterialized());

        MemoryView<?> output = range.materialize();

        assertEquals(DataType.I64, output.dataType());
        assertEquals(Shape.flat(6), output.shape());
        assertEquals(0L, readLong(CONTEXT, output, 0));
        assertEquals(5L, readLong(CONTEXT, output, 5));
    }

    @Test
    void arangeCastsToFp32() {
        Tensor range = Tensor.arange(6, DataType.FP32);
        MemoryView<?> output = range.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.flat(6), output.shape());
        assertEquals(0.0f, readFloat(CONTEXT, output, 0), 0.0001f);
        assertEquals(5.0f, readFloat(CONTEXT, output, 5), 0.0001f);
    }

    @Test
    void arangeRejectsBool() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.arange(4, DataType.BOOL));
    }

    @Test
    void arangeRejectsNegativeCounts() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.arange(-1));
    }

    private static long readLong(
            MemoryContext<MemorySegment> context, MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return context.memoryAccess().readLong(typedView.memory(), offset);
    }

    private static float readFloat(
            MemoryContext<MemorySegment> context, MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return context.memoryAccess().readFloat(typedView.memory(), offset);
    }
}
