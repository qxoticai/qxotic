package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ReshapeOpTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = Environment.nativeMemoryDomain();
    }

    @Test
    void reshapeContiguousInput() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        Tensor result = input.reshape(Shape.of(4, 3));

        assertEquals(Shape.of(4, 3), result.shape());
    }

    @Test
    void reshapeWithHoles() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);
        Layout layout = Layout.of(Shape.flat(4), Stride.flat(2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        Tensor result = input.reshape(Shape.of(2, 2));

        assertEquals(Shape.of(2, 2), result.shape());
        assertTrue(result.materialize().isContiguous());
    }

    @Test
    void reshapeTransposedTensor() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        MemoryView<MemorySegment> transposed = view.transpose(0, 1);
        Tensor input = Tensor.of(transposed);

        Tensor flatView = input.view(Shape.of(12));
        assertTrue(flatView.materialize().isContiguous());

        Tensor flatReshape = input.reshape(Shape.of(12));
        assertEquals(Shape.of(12), flatReshape.shape());
        assertTrue(flatReshape.materialize().isContiguous());
    }

    @Test
    void reshapeSlicedTensorWithGaps() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(4, 3));

        MemoryView<MemorySegment> sliced = view.slice(0, 0, 4, 2);
        Tensor input = Tensor.of(sliced);

        Tensor result = input.reshape(Shape.of(6));

        assertEquals(Shape.of(6), result.shape());
        assertTrue(result.materialize().isContiguous());
    }

    @Test
    void reshapeThrowsOnSizeMismatch() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        assertThrows(IllegalArgumentException.class, () -> input.reshape(Shape.of(5, 5)));
    }

    @Test
    void reshapeThrowsWhenNewShapeTooLarge() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.reshape(Shape.of(4, 6)));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void reshapeThrowsWhenNewShapeTooSmall() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.reshape(Shape.of(2, 2)));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void reshapeThrowsWhenFlatteningToWrongSize() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.reshape(Shape.of(15)));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void reshapeThrowsWithNestedShapeSizeMismatch() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 8).view(Shape.of(2, 4));
        Tensor input = Tensor.of(view);

        Shape nestedShape = Shape.of(Shape.of(3L, 1L), 2L);
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.reshape(nestedShape));
        assertTrue(ex.getMessage().contains("mismatch"));
    }
}
