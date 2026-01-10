package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.impl.MemoryAccessFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

class MemoryViewTest {

    @Test
    void testSlice1() {
        float[] floats = new float[2 * 3 * 5];
        for (int i = 0; i < floats.length; ++i) {
            floats[i] = i;
        }
        MemoryView<float[]> view = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(floats), Layout.rowMajor(Shape.of(2, 3, 5)));
        MemoryView<float[]> view0 = view.slice(0, 0, 1).reshape(Shape.of(3, 5));
        MemoryView<float[]> view1 = view.slice(0, 1, 2).reshape(Shape.of(3, 5));
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
        MemoryView<float[]> view0 = view.slice(-1, 0, 1); // .reshape(Shape.of(2, 3));
        MemoryView<float[]> view1 = view.slice(-1, 1, 2); // .reshape(Shape.of(2, 3));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        System.out.println(AbstractMemoryTest.toString(memoryAccess, view));
        System.out.println(AbstractMemoryTest.toString(memoryAccess, view0));
        System.out.println(AbstractMemoryTest.toString(memoryAccess, view1));
    }
}
