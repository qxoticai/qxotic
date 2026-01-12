package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.impl.MemoryAccessFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MemoryViewTest {

    @Test
    void testSlice1() {
        float[] floats = new float[2 * 3 * 5];
        for (int i = 0; i < floats.length; ++i) {
            floats[i] = i;
        }
        MemoryView<float[]> view = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(floats), Layout.rowMajor(Shape.of(2, 3, 5)));
        MemoryView<float[]> view0 = view.slice(0, 0, 1).view(Shape.of(3, 5));
        MemoryView<float[]> view1 = view.slice(0, 1, 2).view(Shape.of(3, 5));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        System.out.println(AbstractMemoryTest.toString(memoryAccess, view));
        System.out.println(AbstractMemoryTest.toString(memoryAccess, view0));
        System.out.println(AbstractMemoryTest.toString(memoryAccess, view1));
    }

    @Test
    void testSliceLast() {
        float[] floats = new float[2 * 3 * 5];
        for (int i = 0; i < floats.length; ++i) {
            floats[i] = i;
        }
        MemoryView<float[]> view = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(floats), Layout.rowMajor(Shape.of(2, 3, 5)));
        MemoryView<float[]> view0 = view.slice(-1, 0, 1); // .view(Shape.of(2, 3));
        MemoryView<float[]> view1 = view.slice(-1, 1, 2); // .view(Shape.of(2, 3));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        System.out.println(AbstractMemoryTest.toString(memoryAccess, view));
        System.out.println(AbstractMemoryTest.toString(memoryAccess, view0));
        System.out.println(AbstractMemoryTest.toString(memoryAccess, view1));
    }

    @Test
    void testNonContiguousBounds() {
        float[] small = new float[4];
        Layout outOfBoundsLayout = Layout.of(Shape.flat(2, 2), Stride.flat(3, 1));
        assertThrows(IllegalArgumentException.class, () ->
                MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(small), outOfBoundsLayout));

        float[] larger = new float[10];
        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(larger), outOfBoundsLayout));
    }

    @Test
    void testNegativeStrideBounds() {
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(-3, 1));
        Memory<float[]> memory = MemoryFactory.ofFloats(new float[5]);

        assertThrows(IllegalArgumentException.class, () ->
                MemoryViewFactory.of(DataType.F32, memory, 8L, layout));
        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.F32, memory, 12L, layout));
    }

    @Test
    void testBroadcastStrideBounds() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(0, 1));

        Memory<float[]> small = MemoryFactory.ofFloats(new float[2]);
        assertThrows(IllegalArgumentException.class, () ->
                MemoryViewFactory.of(DataType.F32, small, 0L, layout));

        Memory<float[]> exact = MemoryFactory.ofFloats(new float[3]);
        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.F32, exact, 0L, layout));
    }

    @Test
    void testZeroSizedViewAllowsAnyOffset() {
        Layout layout = Layout.of(Shape.flat(0, 3), Stride.flat(1, 1));
        Memory<float[]> memory = MemoryFactory.ofFloats(new float[1]);

        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.F32, memory, 0L, layout));
        assertDoesNotThrow(() ->
                MemoryViewFactory.of(DataType.F32, memory, memory.byteSize() + 16L, layout));
    }
}
