package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

class BroadcastNestedTest {

    @Test
    void testBroadcastFlatToFlatPreservesStructure() {
        float[] data = {1.0f, 2.0f, 3.0f};
        MemoryView<float[]> vec =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.rowMajor(Shape.flat(3)));

        // Broadcast (3) -> (4, 3)
        MemoryView<float[]> result = vec.broadcast(Shape.flat(4, 3));

        assertEquals(Shape.flat(4, 3), result.shape());
        assertTrue(result.shape().isFlat());
        assertTrue(result.isBroadcasted());
    }

    @Test
    void testBroadcastNestedToNestedPreservesCongruence() {
        float[] data = new float[6];
        // Shape: (2, (1, 3))
        Shape nestedShape = Shape.of(2, Shape.of(1L, 3L));
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32, MemoryFactory.ofFloats(data), Layout.rowMajor(nestedShape));

        // Broadcast to: (2, (5, 3))
        Shape targetShape = Shape.of(2, Shape.of(5L, 3L));
        MemoryView<float[]> result = view.broadcast(targetShape);

        assertTrue(result.shape().isCongruentWith(targetShape));
        assertFalse(result.shape().isFlat());
        assertTrue(result.isBroadcasted());
    }

    @Test
    void testBroadcastAddsLeadingDimensionsFlat() {
        float[] data = {1.0f, 2.0f};
        MemoryView<float[]> vec =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.rowMajor(Shape.flat(2)));

        // Broadcast (2) -> (3, 4, 2)
        MemoryView<float[]> result = vec.broadcast(Shape.flat(3, 4, 2));

        assertEquals(Shape.flat(3, 4, 2), result.shape());
        assertTrue(result.isBroadcasted());
    }

    @Test
    void testBroadcastSameRankUsesExpandDirectly() {
        float[] data = new float[4];
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.of(Shape.flat(4, 1), Stride.flat(1, 0)));

        // Broadcast (4, 1) -> (4, 5) - same rank, just expands
        MemoryView<float[]> result = view.broadcast(Shape.flat(4, 5));

        assertEquals(Shape.flat(4, 5), result.shape());
        assertTrue(result.isBroadcasted());
    }

    @Test
    void testBroadcastScalarToVector() {
        float[] data = {42.0f};
        MemoryView<float[]> scalar =
                MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(data), Layout.scalar());

        // Broadcast () -> (5)
        MemoryView<float[]> result = scalar.broadcast(Shape.flat(5));

        assertEquals(Shape.flat(5), result.shape());
        assertTrue(result.isBroadcasted());
    }

    @Test
    void testBroadcastScalarToMatrix() {
        float[] data = {7.0f};
        MemoryView<float[]> scalar =
                MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(data), Layout.scalar());

        // Broadcast () -> (3, 4)
        MemoryView<float[]> result = scalar.broadcast(Shape.flat(3, 4));

        assertEquals(Shape.flat(3, 4), result.shape());
        assertTrue(result.isBroadcasted());
    }

    @Test
    void testBroadcastIncompatibleDimensionThrows() {
        float[] data = {1.0f, 2.0f, 3.0f};
        MemoryView<float[]> vec =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.rowMajor(Shape.flat(3)));

        // Cannot broadcast (3) -> (4) - incompatible
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    vec.broadcast(Shape.flat(4));
                });
    }

    @Test
    void testBroadcastFewerRanksThrows() {
        float[] data = new float[12];
        MemoryView<float[]> mat =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.rowMajor(Shape.flat(3, 4)));

        // Cannot broadcast to fewer dimensions
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    mat.broadcast(Shape.flat(3));
                });
    }

    @Test
    void testBroadcastWithExistingSingletonDimension() {
        float[] data = {5.0f, 10.0f};
        MemoryView<float[]> vec =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.rowMajor(Shape.flat(2, 1)));

        // Broadcast (2, 1) -> (2, 4)
        MemoryView<float[]> result = vec.broadcast(Shape.flat(2, 4));

        assertEquals(Shape.flat(2, 4), result.shape());
        assertTrue(result.isBroadcasted());
    }

    @Test
    void testBroadcastChainedWithOtherOps() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        MemoryView<float[]> vec =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(data),
                        Layout.rowMajor(Shape.flat(4)));

        // Chain: view -> broadcast -> transpose
        MemoryView<float[]> result =
                vec.view(Shape.flat(1, 4)).broadcast(Shape.flat(3, 4)).transpose(0, 1);

        assertEquals(Shape.flat(4, 3), result.shape());
    }

    @Test
    void testBroadcastNestedWithMultipleModes() {
        float[] data = new float[8];
        // Shape: ((2, 1), (2, 2))
        Shape nestedShape = Shape.of(Shape.of(2L, 1L), Shape.of(2L, 2L));
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32, MemoryFactory.ofFloats(data), Layout.rowMajor(nestedShape));

        // Broadcast to: ((2, 5), (2, 2))
        Shape targetShape = Shape.of(Shape.of(2L, 5L), Shape.of(2L, 2L));
        MemoryView<float[]> result = view.broadcast(targetShape);

        assertTrue(result.shape().isCongruentWith(targetShape));
        assertTrue(result.isBroadcasted());
    }
}
