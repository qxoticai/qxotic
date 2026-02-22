package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

class ExpandTest {

    @Test
    void testExpandFlatSingletonToLarger() {
        float[] data = new float[4];
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.of(Shape.flat(4, 1), Stride.flat(1, 0)));

        // Expand (4, 1) -> (4, 5)
        MemoryView<float[]> expanded = view.expand(Shape.flat(4, 5));

        assertEquals(Shape.flat(4, 5), expanded.shape());
        assertTrue(expanded.isBroadcasted()); // Has zero stride
        assertArrayEquals(new long[] {1, 0}, expanded.stride().toArray());
    }

    @Test
    void testExpandNestedSingletonToLarger() {
        float[] data = new float[8]; // Need 4 * 1 * 2 = 8 elements
        // Shape: (4, (1, 2)) where nested mode is (1, 2)
        Shape nestedShape = Shape.of(4, Shape.of(1L, 2L));
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32, MemoryFactory.ofFloats(data), Layout.rowMajor(nestedShape));

        // Expand to: (4, (3, 2))
        Shape targetShape = Shape.of(4, Shape.of(3L, 2L));
        MemoryView<float[]> expanded = view.expand(targetShape);

        assertEquals(targetShape, expanded.shape());
        assertTrue(expanded.shape().isCongruentWith(targetShape));
        assertTrue(expanded.isBroadcasted());
    }

    @Test
    void testExpandPreservesCongruence() {
        float[] data = new float[6];
        // Shape: (2, (1, 3))
        Shape nestedShape = Shape.of(2, Shape.of(1L, 3L));
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32, MemoryFactory.ofFloats(data), Layout.rowMajor(nestedShape));

        // Expand to: (2, (5, 3))
        Shape targetShape = Shape.of(2, Shape.of(5L, 3L));
        MemoryView<float[]> expanded = view.expand(targetShape);

        assertTrue(expanded.shape().isCongruentWith(targetShape));
        assertFalse(expanded.shape().isFlat());
    }

    @Test
    void testExpandNonSingletonDimensionThrows() {
        float[] data = new float[12];
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.of(Shape.flat(3, 4), Stride.flat(4, 1)));

        // Cannot expand non-singleton dimension
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    view.expand(Shape.flat(5, 4));
                });
    }

    @Test
    void testExpandNonCongruentShapeThrows() {
        float[] data = new float[6];
        // Nested shape: (2, (1, 3))
        Shape nestedShape = Shape.of(2, Shape.of(1L, 3L));
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32, MemoryFactory.ofFloats(data), Layout.rowMajor(nestedShape));

        // Try to expand to flat shape (not congruent)
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    view.expand(Shape.flat(2, 5, 3));
                });
    }

    @Test
    void testExpandMatchingDimensionsPreservesStrides() {
        float[] data = new float[12];
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.of(Shape.flat(3, 4), Stride.flat(4, 1)));

        // Expand with no changes (all dimensions match)
        MemoryView<float[]> expanded = view.expand(Shape.flat(3, 4));

        assertEquals(Shape.flat(3, 4), expanded.shape());
        assertArrayEquals(new long[] {4, 1}, expanded.stride().toArray());
        assertFalse(expanded.isBroadcasted()); // No zero strides
    }

    @Test
    void testExpandScalarToShape() {
        float[] data = {42.0f};
        MemoryView<float[]> scalar =
                MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(data), Layout.scalar());

        // Cannot expand scalar directly - need to add dimensions first via view()
        // This should fail since ranks don't match
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    scalar.expand(Shape.flat(3, 4));
                });
    }

    @Test
    void testExpandMultipleSingletons() {
        float[] data = new float[2];
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.of(Shape.flat(1, 2, 1), Stride.flat(0, 1, 0)));

        // Expand both singleton dimensions
        MemoryView<float[]> expanded = view.expand(Shape.flat(5, 2, 3));

        assertEquals(Shape.flat(5, 2, 3), expanded.shape());
        assertArrayEquals(new long[] {0, 1, 0}, expanded.stride().toArray());
        assertTrue(expanded.isBroadcasted());
    }

    @Test
    void testExpandWithNestedMultipleSingletons() {
        float[] data = new float[4];
        // Shape: ((1, 2), (1, 2))
        Shape nestedShape = Shape.of(Shape.of(1L, 2L), Shape.of(1L, 2L));
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32, MemoryFactory.ofFloats(data), Layout.rowMajor(nestedShape));

        // Expand to: ((3, 2), (5, 2))
        Shape targetShape = Shape.of(Shape.of(3L, 2L), Shape.of(5L, 2L));
        MemoryView<float[]> expanded = view.expand(targetShape);

        assertTrue(expanded.shape().isCongruentWith(targetShape));
        assertTrue(expanded.isBroadcasted());
    }
}
