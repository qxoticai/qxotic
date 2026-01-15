package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import ai.qxotic.jota.memory.impl.MemoryAccessFactory;
import ai.qxotic.jota.memory.impl.MemoryFactory;
import ai.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MemoryViewTest {

    @Test
    void testSlice1() {
        float[] floats = new float[2 * 3 * 5];
        for (int i = 0; i < floats.length; ++i) {
            floats[i] = i;
        }
        MemoryView<float[]> view = MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(floats), Layout.rowMajor(Shape.of(2, 3, 5)));
        MemoryView<float[]> view0 = view.slice(0, 0, 1).view(Shape.of(3, 5));
        MemoryView<float[]> view1 = view.slice(0, 1, 2).view(Shape.of(3, 5));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        System.out.println(MemoryViewPrinter.toString(view, memoryAccess));
        System.out.println(MemoryViewPrinter.toString(view0, memoryAccess));
        System.out.println(MemoryViewPrinter.toString(view1, memoryAccess));
    }

    @Test
    void testSliceLast() {
        float[] floats = new float[2 * 3 * 5];
        for (int i = 0; i < floats.length; ++i) {
            floats[i] = i;
        }
        MemoryView<float[]> view = MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(floats), Layout.rowMajor(Shape.of(2, 3, 5)));
        MemoryView<float[]> view0 = view.slice(-1, 0, 1); // .view(Shape.of(2, 3));
        MemoryView<float[]> view1 = view.slice(-1, 1, 2); // .view(Shape.of(2, 3));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        System.out.println(MemoryViewPrinter.toString(view, memoryAccess));
        System.out.println(MemoryViewPrinter.toString(view0, memoryAccess));
        System.out.println(MemoryViewPrinter.toString(view1, memoryAccess));
    }

    @Test
    void testToStringMetadata() {
        float[] floats = new float[4];
        MemoryView<float[]> view = MemoryViewFactory.of(
                DataType.FP32,
                MemoryFactory.ofFloats(floats),
                Layout.rowMajor(Shape.of(2, 2))
        );

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
        MemoryView<float[]> view = MemoryViewFactory.of(
                DataType.FP32,
                MemoryFactory.ofFloats(floats),
                Layout.rowMajor(Shape.of(10, 10))
        );
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        String text = view.toString(memoryAccess);
        assertTrue(text.contains("..."));
    }

    @Test
    void testToStringCompactFloats() {
        float[] floats = new float[]{4.0f, 4.5f, Float.POSITIVE_INFINITY, Float.NaN};
        MemoryView<float[]> view = MemoryViewFactory.of(
                DataType.FP32,
                MemoryFactory.ofFloats(floats),
                Layout.rowMajor(Shape.of(4))
        );
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

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
        assertThrows(IllegalArgumentException.class, () ->
                MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(small), outOfBoundsLayout));

        float[] larger = new float[10];
        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(larger), outOfBoundsLayout));
    }

    @Test
    void testNegativeStrideBounds() {
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(-3, 1));
        Memory<float[]> memory = MemoryFactory.ofFloats(new float[5]);

        assertThrows(IllegalArgumentException.class, () ->
                MemoryViewFactory.of(DataType.FP32, memory, 8L, layout));
        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.FP32, memory, 12L, layout));
    }

    @Test
    void testBroadcastStrideBounds() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(0, 1));

        Memory<float[]> small = MemoryFactory.ofFloats(new float[2]);
        assertThrows(IllegalArgumentException.class, () ->
                MemoryViewFactory.of(DataType.FP32, small, 0L, layout));

        Memory<float[]> exact = MemoryFactory.ofFloats(new float[3]);
        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.FP32, exact, 0L, layout));
    }

    @Test
    void testZeroSizedViewAllowsAnyOffset() {
        Layout layout = Layout.of(Shape.flat(0, 3), Stride.flat(1, 1));
        Memory<float[]> memory = MemoryFactory.ofFloats(new float[1]);

        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.FP32, memory, 0L, layout));
        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.FP32, memory, memory.byteSize() + 16L, layout));
    }
}
