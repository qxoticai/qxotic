package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class BooleanTensorTest {

    @SuppressWarnings("unchecked")
    private static final MemoryContext<MemorySegment> CONTEXT =
            (MemoryContext<MemorySegment>) Environment.current().nativeBackend().memoryContext();

    @Test
    void createsBoolTensorFromArray() {
        Tensor tensor = Tensor.of(new boolean[] {true, false, true});
        MemoryView<?> output = tensor.materialize();

        assertEquals(DataType.BOOL, output.dataType());
        assertEquals(Shape.flat(3), output.shape());
        assertEquals(1, readByte(output, 0));
        assertEquals(0, readByte(output, 1));
        assertEquals(1, readByte(output, 2));
    }

    @Test
    void rejectsShapeMismatch() {
        boolean[] data = new boolean[] {true, false};
        assertThrows(IllegalArgumentException.class, () -> Tensor.of(data, Shape.flat(3)));
    }

    private static byte readByte(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return CONTEXT.memoryAccess().readByte(typedView.memory(), offset);
    }
}
