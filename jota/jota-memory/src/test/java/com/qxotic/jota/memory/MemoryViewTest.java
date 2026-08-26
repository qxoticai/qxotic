package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import org.junit.jupiter.api.Test;

class MemoryViewTest {

    @Test
    void testSlice1() {
        float[] floats = new float[2 * 3 * 5];
        for (int i = 0; i < floats.length; ++i) {
            floats[i] = i;
        }
        MemoryView<float[]> view =
                MemoryViews.of(
                        Memories.of(floats), DataType.FP32, Layout.rowMajor(Shape.of(2, 3, 5)));
        MemoryView<float[]> view0 = view.slice(0, 0, 1).view(Shape.of(3, 5));
        MemoryView<float[]> view1 = view.slice(0, 1, 2).view(Shape.of(3, 5));
        MemoryAccess<float[]> memoryAccess = MemoryDomains.floats().directAccess();
    }

    @Test
    void testSliceLast() {
        float[] floats = new float[2 * 3 * 5];
        for (int i = 0; i < floats.length; ++i) {
            floats[i] = i;
        }
        MemoryView<float[]> view =
                MemoryViews.of(
                        Memories.of(floats), DataType.FP32, Layout.rowMajor(Shape.of(2, 3, 5)));
        MemoryView<float[]> view0 = view.slice(-1, 0, 1); // .view(Shape.of(2, 3));
        MemoryView<float[]> view1 = view.slice(-1, 1, 2); // .view(Shape.of(2, 3));
        MemoryAccess<float[]> memoryAccess = MemoryDomains.floats().directAccess();
    }

    @Test
    void testToStringMetadata() {
        float[] floats = new float[4];
        MemoryView<float[]> view =
                MemoryViews.of(Memories.of(floats), DataType.FP32, Layout.rowMajor(Shape.of(2, 2)));

        String text = view.toString();
        assertTrue(text.startsWith("MemoryView{"));
        assertTrue(text.contains("layout="));
        assertTrue(text.contains("dataType=fp32"));
    }

    @Test
    void testToStringValuesElision() {
        float[] floats = new float[100];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = i;
        }
        MemoryView<float[]> view =
                MemoryViews.of(
                        Memories.of(floats), DataType.FP32, Layout.rowMajor(Shape.of(10, 10)));
        MemoryAccess<float[]> memoryAccess = MemoryDomains.floats().directAccess();

        String text = view.toString(memoryAccess);
        assertTrue(text.contains("..."));
    }

    @Test
    void testToStringCompactFloats() {
        float[] floats = new float[] {4.0f, 4.5f, Float.POSITIVE_INFINITY, Float.NaN};
        MemoryView<float[]> view =
                MemoryViews.of(Memories.of(floats), DataType.FP32, Layout.rowMajor(Shape.of(4)));
        MemoryAccess<float[]> memoryAccess = MemoryDomains.floats().directAccess();

        String text = view.toString(memoryAccess, ViewPrintOptions.valuesOnly());
        assertTrue(text.contains("4."));
        assertTrue(text.contains("4.5"));
        assertTrue(text.contains("+INF"));
        assertTrue(text.contains("NaN"));
        assertFalse(text.contains("4.0000"));
    }

    @Test
    void testNonContiguousBounds() {
        float[] small = new float[4];
        Layout outOfBoundsLayout = Layout.of(Shape.flat(2, 2), Stride.flat(3, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> MemoryViews.of(Memories.of(small), DataType.FP32, outOfBoundsLayout));

        float[] larger = new float[10];
        assertDoesNotThrow(
                () -> MemoryViews.of(Memories.of(larger), DataType.FP32, outOfBoundsLayout));
    }

    @Test
    void testNegativeStrideBounds() {
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(-3, 1));
        Memory<float[]> memory = Memories.of(new float[5]);

        assertThrows(
                IllegalArgumentException.class,
                () -> MemoryViews.of(memory, 8L, DataType.FP32, layout));
        assertDoesNotThrow(() -> MemoryViews.of(memory, 12L, DataType.FP32, layout));
    }

    @Test
    void testBroadcastStrideBounds() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(0, 1));

        Memory<float[]> small = Memories.of(new float[2]);
        assertThrows(
                IllegalArgumentException.class,
                () -> MemoryViews.of(small, 0L, DataType.FP32, layout));

        Memory<float[]> exact = Memories.of(new float[3]);
        assertDoesNotThrow(() -> MemoryViews.of(exact, 0L, DataType.FP32, layout));
    }

    @Test
    void testZeroSizedViewAllowsAnyOffset() {
        Layout layout = Layout.of(Shape.flat(0, 3), Stride.flat(1, 1));
        Memory<float[]> memory = Memories.of(new float[1]);

        assertDoesNotThrow(() -> MemoryViews.of(memory, 0L, DataType.FP32, layout));
        assertDoesNotThrow(
                () -> MemoryViews.of(memory, memory.byteSize() + 16L, DataType.FP32, layout));
    }

    @Test
    void viewRejectsOverflowingSpan() {
        // (dim - 1) * stride wraps negative and used to land on the "min" side of the span
        Memory<byte[]> memory = Memories.of(new byte[16]);
        long stride = Long.MAX_VALUE / 2 + 2;
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        MemoryView.of(
                                memory,
                                4,
                                DataType.I8,
                                Layout.of(Shape.flat(3), Stride.flat(stride))));
    }
}
