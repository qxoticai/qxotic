package ai.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.impl.MemoryFactory;
import ai.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

public class IsContiguousTest extends MemoryTest {

    @Test
    void testScalarIsContiguous() {
        MemoryView<float[]> scalar =
                MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(42), Layout.scalar());
        assertTrue(scalar.isRowMajorContiguous());
        assertTrue(scalar.isSpanContiguous());
        assertTrue(scalar.isNonOverlapping());
    }

    @Test
    void test1DContiguous() {
        MemoryView<float[]> v =
                MemoryViewFactory.of(
                        DataType.FP32, MemoryFactory.ofFloats(new float[10]), Layout.rowMajor(10));
        assertTrue(v.isRowMajorContiguous(), "1D fresh view should be row-major contiguous");
        assertTrue(v.isSpanContiguous(), "1D fresh view should be span-contiguous");
        assertTrue(v.isNonOverlapping(), "1D fresh view should be non-overlapping");
    }

    @Test
    void test2DContiguous() {
        MemoryView<float[]> v =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(new float[12]),
                        Layout.rowMajor(3, 4));
        assertTrue(
                v.isRowMajorContiguous(), "Fresh 2D row-major view should be row-major contiguous");
        assertTrue(v.isSpanContiguous(), "Fresh 2D row-major view should be span-contiguous");
        assertTrue(v.isNonOverlapping(), "Fresh 2D row-major view should be non-overlapping");
    }

    @Test
    void testFlattenIsContiguous() {
        MemoryView<float[]> v =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(new float[12]),
                        Layout.rowMajor(3, 4));
        MemoryView<float[]> flat = v.view(Shape.of(12));
        assertTrue(flat.isRowMajorContiguous(), "Reshape to 1D should remain row-major contiguous");
        assertTrue(flat.isSpanContiguous(), "Reshape to 1D should remain span-contiguous");
        assertTrue(flat.isNonOverlapping(), "Reshape to 1D should remain non-overlapping");
    }

    @Test
    void testTransposeIsContiguous() {
        MemoryView<float[]> v =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(new float[12]),
                        Layout.rowMajor(3, 4));
        MemoryView<float[]> t = v.transpose(0, 1);
        // Transpose is span-contiguous and non-overlapping, but not row-major contiguous
        assertFalse(t.isRowMajorContiguous(), "Transpose should not be row-major contiguous");
        assertTrue(t.isSpanContiguous(), "Transposed 2D view spans a contiguous range");
        assertTrue(t.isNonOverlapping(), "Transpose should be non-overlapping");
    }

    @Test
    void testDoubleTransposeBackIsContiguous() {
        MemoryView<float[]> v =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(new float[12]),
                        Layout.rowMajor(3, 4));
        MemoryView<float[]> t = v.transpose(0, 1).transpose(0, 1);
        assertTrue(t.isRowMajorContiguous(), "Double transpose gets back to row-major layout");
        assertTrue(t.isSpanContiguous(), "Double transpose gets back to span-contiguous layout");
        assertTrue(t.isNonOverlapping(), "Double transpose remains non-overlapping");
    }

    @Test
    void testSliceSingleRowIsContiguous() {
        MemoryView<float[]> v =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(new float[12]),
                        Layout.rowMajor(3, 4));
        // Take the second row; shape becomes (1, 4). This should remain contiguous.
        MemoryView<float[]> row = v.slice(0, 1, 2);
        assertTrue(row.isSpanContiguous(), "Slicing a single full row should be span-contiguous");
        assertTrue(row.isNonOverlapping(), "Slicing a single full row should be non-overlapping");
        // Optional: also check after reshape to 1D (length 4)
        assertTrue(
                row.view(Shape.of(4)).isRowMajorContiguous(),
                "Reshaped row should be row-major contiguous");
    }

    @Test
    void testSliceWithIndexStrideIsNotContiguous() {
        MemoryView<float[]> v =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(new float[18]),
                        Layout.rowMajor(3, 6));
        // Take every other column: stride 2 along the last axis -> non-contiguous
        MemoryView<float[]> colStride2 = v.slice(-1, 0, 6, 2);
        assertFalse(
                colStride2.isSpanContiguous(),
                "Slicing with indexStride > 1 breaks span contiguity");
        assertTrue(
                colStride2.isNonOverlapping(), "Slicing with indexStride > 1 is non-overlapping");
    }

    @Test
    void testSliceMiddleAxisToSingletonIsNotContiguous() {
        MemoryView<float[]> v =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(new float[24]),
                        Layout.rowMajor(2, 3, 4));
        // Take the middle axis index 1..2 -> shape becomes (2, 1, 4); but strides reflect original
        // layout -> non-contiguous
        MemoryView<float[]> midSingleton = v.slice(1, 1, 2);
        assertFalse(
                midSingleton.isSpanContiguous(),
                "Slicing a middle axis to size 1 does not repack; should be non-span-contiguous");
        assertTrue(
                midSingleton.isNonOverlapping(),
                "Slicing a middle axis to size 1 is non-overlapping");
    }

    @Test
    void testConstructedWithSingletonMiddleAxisIsContiguous() {
        // Contrasts with the slice case: building a fresh view (2,1,4) should be compact and
        // contiguous
        MemoryView<float[]> fresh =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(new float[8]),
                        Layout.rowMajor(2, 1, 4));
        assertTrue(
                fresh.isRowMajorContiguous(),
                "Fresh compact (2,1,4) view should be row-major contiguous");
        assertTrue(
                fresh.isSpanContiguous(), "Fresh compact (2,1,4) view should be span-contiguous");
        assertTrue(
                fresh.isNonOverlapping(), "Fresh compact (2,1,4) view should be non-overlapping");
    }

    @Test
    void testBroadcastExpandIsNotContiguous() {
        MemoryView<float[]> scalar =
                MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(42f), Layout.scalar());
        assertTrue(scalar.isRowMajorContiguous(), "Scalar should be row-major contiguous");
        assertTrue(scalar.isSpanContiguous(), "Scalar should be span-contiguous");
        assertTrue(scalar.isNonOverlapping(), "Scalar should be non-overlapping");
        MemoryView<float[]> broadcast = scalar.broadcast(Shape.of(2, 3));
        assertFalse(
                broadcast.isNonOverlapping(),
                "Broadcasted (expanded) view uses stride 0; should overlap");
        assertFalse(broadcast.isSpanContiguous(), "Broadcasted view should not be span-contiguous");
        assertFalse(
                broadcast.isRowMajorContiguous(),
                "Broadcasted view should not be row-major contiguous");
    }
}
