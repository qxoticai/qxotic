package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class ArangeTensorTest {

    @SuppressWarnings("unchecked")
    private static final MemoryDomain<MemorySegment> CONTEXT =
            (MemoryDomain<MemorySegment>) Environment.current().nativeRuntime().memoryDomain();

    @Test
    void iotaDefaultsToI64() {
        Tensor range = Tensor.iota(6);

        assertTrue(range.isLazy());
        assertFalse(range.isMaterialized());

        MemoryView<?> output = range.materialize();

        assertEquals(DataType.I64, output.dataType());
        assertEquals(Shape.flat(6), output.shape());
        assertEquals(0L, readLong(CONTEXT, output, 0));
        assertEquals(5L, readLong(CONTEXT, output, 5));
    }

    @Test
    void iotaCastsToFp32() {
        Tensor range = Tensor.iota(6, DataType.FP32);
        MemoryView<?> output = range.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.flat(6), output.shape());
        assertEquals(0.0f, readFloat(CONTEXT, output, 0), 0.0001f);
        assertEquals(5.0f, readFloat(CONTEXT, output, 5), 0.0001f);
    }

    @Test
    void iotaRejectsBool() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.iota(4, DataType.BOOL));
    }

    @Test
    void iotaRejectsNegativeCounts() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.iota(-1));
    }

    private static long readLong(
            MemoryDomain<MemorySegment> domain, MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return domain.directAccess().readLong(typedView.memory(), offset);
    }

    private static float readFloat(
            MemoryDomain<MemorySegment> domain, MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return domain.directAccess().readFloat(typedView.memory(), offset);
    }
}
