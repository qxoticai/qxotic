package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;
import com.llm4j.jota.memory.impl.MemoryAccessFactory;
import com.llm4j.jota.memory.impl.MemoryFactory;
import com.llm4j.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MemoryViewTest {

    @Test
    void testSlice1() {
        float[] floats = new float[2 * 3 * 5];
        for (int i = 0; i < floats.length; ++i) {
            floats[i] = i;
        }
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(2, 3, 5), DataType.F32, MemoryFactory.ofFloats(floats));
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
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(2, 3, 5), DataType.F32, MemoryFactory.ofFloats(floats));
        MemoryView<float[]> view0 = view.slice(-1, 0, 1); // .reshape(Shape.of(2, 3));
        MemoryView<float[]> view1 = view.slice(-1, 1, 2); // .reshape(Shape.of(2, 3));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        System.out.println(AbstractMemoryTest.toString(memoryAccess, view));
        System.out.println(AbstractMemoryTest.toString(memoryAccess, view0));
        System.out.println(AbstractMemoryTest.toString(memoryAccess, view1));
    }
}
