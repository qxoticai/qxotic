package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;
import com.llm4j.jota.memory.impl.MemoryFactory;
import com.llm4j.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class IsContiguousTest extends MemoryTest {

    @Test
    void testScalarIsContiguous() {
        MemoryView<float[]> scalar = MemoryViewFactory.of(Shape.scalar(), DataType.F32, MemoryFactory.ofFloats(42));
        assertTrue(scalar.isContiguous());
    }

    @Test
    void test1DContiguous() {
        MemoryView<float[]> v = MemoryViewFactory.of(Shape.of(10), DataType.F32, MemoryFactory.ofFloats(new float[10]));
        assertTrue(v.isContiguous(), "1D fresh view should be contiguous");
    }

    @Test
    void test2DContiguous() {
        MemoryView<float[]> v = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(new float[12]));
        assertTrue(v.isContiguous(), "Fresh 2D row-major view should be contiguous");
    }

    @Test
    void testFlattenIsContiguous() {
        MemoryView<float[]> v = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(new float[12]));
        MemoryView<float[]> flat = v.reshape(Shape.of(12));
        assertTrue(flat.isContiguous(), "Reshape to 1D should remain contiguous");
    }

    @Test
    void testTransposeIsNotContiguous() {
        MemoryView<float[]> v = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(new float[12]));
        MemoryView<float[]> t = v.transpose(0, 1);
        assertFalse(t.isContiguous(), "Transposed 2D view should not be contiguous (row-major)");
    }

    @Test
    void testDoubleTransposeBackIsContiguous() {
        MemoryView<float[]> v = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(new float[12]));
        MemoryView<float[]> t = v.transpose(0, 1).transpose(0, 1);
        assertTrue(t.isContiguous(), "Double transpose gets back to contiguous layout");
    }

    @Test
    void testSliceSingleRowIsContiguous() {
        MemoryView<float[]> v = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(new float[12]));
        // Take the second row; shape becomes (1, 4). This should remain contiguous.
        MemoryView<float[]> row = v.slice(0, 1, 2);
        assertTrue(row.isContiguous(), "Slicing a single full row should be contiguous");
        // Optional: also check after reshape to 1D (length 4)
        assertTrue(row.reshape(Shape.of(4)).isContiguous(), "Reshaped row should be contiguous");
    }

    @Test
    void testSliceWithIndexStrideIsNotContiguous() {
        MemoryView<float[]> v = MemoryViewFactory.of(Shape.of(3, 6), DataType.F32, MemoryFactory.ofFloats(new float[18]));
        // Take every other column: stride 2 along the last axis -> non-contiguous
        MemoryView<float[]> colStride2 = v.slice(-1, 0, 6, 2);
        assertFalse(colStride2.isContiguous(), "Slicing with indexStride > 1 breaks contiguity");
    }

    @Test
    void testSliceMiddleAxisToSingletonIsNotContiguous() {
        MemoryView<float[]> v = MemoryViewFactory.of(Shape.of(2, 3, 4), DataType.F32, MemoryFactory.ofFloats(new float[24]));
        // Take the middle axis index 1..2 -> shape becomes (2, 1, 4); but strides reflect original layout -> non-contiguous
        MemoryView<float[]> midSingleton = v.slice(1, 1, 2);
        assertFalse(midSingleton.isContiguous(), "Slicing a middle axis to size 1 does not repack; should be non-contiguous");
    }

    @Test
    void testConstructedWithSingletonMiddleAxisIsContiguous() {
        // Contrasts with the slice case: building a fresh view (2,1,4) should be compact and contiguous
        MemoryView<float[]> fresh = MemoryViewFactory.of(Shape.of(2, 1, 4), DataType.F32, MemoryFactory.ofFloats(new float[8]));
        assertTrue(fresh.isContiguous(), "Fresh compact (2,1,4) view should be contiguous");
    }

    @Test
    void testBroadcastExpandIsNotContiguous() {
        MemoryView<float[]> scalar = MemoryViewFactory.of(Shape.scalar(), DataType.F32, MemoryFactory.ofFloats(42f));
        assertTrue(scalar.isContiguous(), "Scalar should be contiguous");
        MemoryView<float[]> broadcast = scalar.broadcast(Shape.of(2, 3));
        assertFalse(broadcast.isContiguous(), "Broadcasted (expanded) view uses stride 0; should not be contiguous");
    }
}
