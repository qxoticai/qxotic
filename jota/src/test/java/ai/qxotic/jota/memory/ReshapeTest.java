package ai.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.impl.MemoryAccessFactory;
import ai.qxotic.jota.memory.impl.MemoryFactory;
import ai.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

class ReshapeTest extends AbstractMemoryTest {

    static float[] arange(int n) {
        float[] result = new float[n];
        for (int i = 0; i < n; i++) {
            result[i] = i;
        }
        return result;
    }

    @Test
    void testReshapeAfterSlice() {
        // Test the slice + reshape pattern from the reference
        float[] floats = arange(2 * 3 * 5);
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(floats), Shape.of(2, 3, 5));

        // Slice last dimension and reshape (squeeze)
        MemoryView<float[]> sliced = view.slice(-1, 0, 1); // Shape: (2, 3, 1)
        assertEquals(Shape.flat(2, 3, 1), sliced.shape());

        MemoryView<float[]> reshaped = sliced.view(Shape.of(2, 3)); // Remove singleton
        assertEquals(Shape.of(2, 3), reshaped.shape());

        // Verify data integrity using helper method
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
        // First column should be [0, 5, 10, 15, 20, 25]
        assertEquals(0.0f, readFloat(memoryAccess, reshaped, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, reshaped, 0, 1));
        assertEquals(10.0f, readFloat(memoryAccess, reshaped, 0, 2));
        assertEquals(15.0f, readFloat(memoryAccess, reshaped, 1, 0));
        assertEquals(20.0f, readFloat(memoryAccess, reshaped, 1, 1));
        assertEquals(25.0f, readFloat(memoryAccess, reshaped, 1, 2));
    }

    @Test
    void testReshapeContiguous() {
        // Test basic reshape on contiguous view
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(floats),
                        Layout.rowMajor(Shape.of(2, 3, 4)));

        // Reshape to different dimensions with same total elements
        MemoryView<float[]> reshaped = view.view(Shape.of(6, 4));
        assertEquals(Shape.of(6, 4), reshaped.shape());
        assertTrue(reshaped.isContiguous());

        // Reshape to 1D
        MemoryView<float[]> flat = view.view(Shape.of(24));
        assertEquals(Shape.of(24), flat.shape());
        assertTrue(flat.isContiguous());

        // Reshape to higher dimensions
        MemoryView<float[]> higher = view.view(Shape.of(2, 2, 3, 2));
        assertEquals(Shape.of(2, 2, 3, 2), higher.shape());
        assertTrue(higher.isContiguous());
    }

    @Test
    void testReshapeWithSingletons() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(floats),
                        Layout.rowMajor(Shape.of(2, 3, 4)));

        // Add singleton dimensions (unsqueeze)
        MemoryView<float[]> unsqueezed = view.view(Shape.of(1, 2, 3, 4, 1));
        assertEquals(Shape.of(1, 2, 3, 4, 1), unsqueezed.shape());

        // Remove singleton dimensions (squeeze) - start with view that has singletons
        MemoryView<float[]> withSingletons =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(floats),
                        Layout.rowMajor(Shape.of(1, 2, 3, 4, 1)));
        MemoryView<float[]> squeezed = withSingletons.view(Shape.of(2, 3, 4));
        assertEquals(Shape.of(2, 3, 4), squeezed.shape());
    }

    @Test
    void testReshapeDimensionCollapsing() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(floats),
                        Layout.rowMajor(Shape.of(2, 3, 4)));

        // Collapse first two dimensions
        MemoryView<float[]> collapsed = view.view(Shape.of(6, 4));
        assertEquals(Shape.of(6, 4), collapsed.shape());

        // Collapse last two dimensions
        MemoryView<float[]> collapsed2 = view.view(Shape.of(2, 12));
        assertEquals(Shape.of(2, 12), collapsed2.shape());

        // Collapse all dimensions
        MemoryView<float[]> flat = view.view(Shape.of(24));
        assertEquals(Shape.of(24), flat.shape());
    }

    @Test
    void testReshapeDimensionSplitting() {
        float[] floats = arange(24);
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(floats),
                        Layout.rowMajor(Shape.of(6, 4)));

        // Split first dimension
        MemoryView<float[]> split = view.view(Shape.of(2, 3, 4));
        assertEquals(Shape.of(2, 3, 4), split.shape());

        // Split second dimension
        MemoryView<float[]> split2 = view.view(Shape.of(6, 2, 2));
        assertEquals(Shape.of(6, 2, 2), split2.shape());

        // Split into more dimensions
        MemoryView<float[]> split3 = view.view(Shape.of(2, 3, 2, 2));
        assertEquals(Shape.of(2, 3, 2, 2), split3.shape());
    }

    @Test
    void testReshapeErrorCases() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(floats),
                        Layout.rowMajor(Shape.of(2, 3, 4)));

        // Test element count mismatch
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    view.view(Shape.of(2, 3, 5)); // 30 elements vs 24
                });

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    view.view(Shape.of(2, 3)); // 6 elements vs 24
                });

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    view.view(Shape.of(25)); // 25 elements vs 24
                });
    }

    /**
     * Tests that permuted views spanning contiguous memory can be reshaped.
     *
     * <p>With CuTe semantics, a permuted view (2,4,3):(12,1,4) spans [0-23] contiguously, so any
     * reshape to size 24 is valid - the new shape gets row-major strides.
     */
    @Test
    void testReshapePermutedViewSpanningContiguousRange() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(floats),
                        Layout.rowMajor(Shape.of(2, 3, 4)));

        // Permuted view (2, 4, 3):(12, 1, 4) spans [0-23] contiguously
        // Condition: (2-1)*12 + (4-1)*1 + (3-1)*4 = 12+3+8 = 23 = 24-1 ✓
        MemoryView<float[]> permuted = view.permute(0, 2, 1); // Shape: (2, 4, 3)
        assertFalse(permuted.isContiguous());

        // CuTe semantics: reshape works, new shape gets row-major strides
        MemoryView<float[]> reshaped = permuted.view(Shape.of(8, 3));
        assertEquals(Shape.of(8, 3), reshaped.shape());
        assertTrue(reshaped.isContiguous());
        assertSame(view.memory(), reshaped.memory());
    }

    @Test
    void testReshapePreservesMemory() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view =
                MemoryViewFactory.of(
                        DataType.FP32,
                        MemoryFactory.ofFloats(floats),
                        Layout.rowMajor(Shape.of(2, 3, 4)));

        MemoryView<float[]> reshaped = view.view(Shape.of(6, 4));

        // Should reference the same memory
        assertSame(view.memory(), reshaped.memory());
        assertEquals(view.byteOffset(), reshaped.byteOffset());
        assertEquals(view.dataType(), reshaped.dataType());
    }

    @Test
    void testReshapeZeroDimensions() {
        float[] floats = arange(1);
        MemoryView<float[]> scalar =
                MemoryViewFactory.of(
                        DataType.FP32, MemoryFactory.ofFloats(floats), Layout.scalar());

        // Reshape scalar to 1D with single element
        MemoryView<float[]> vector = scalar.view(Shape.of(1));
        assertEquals(Shape.of(1), vector.shape());

        // Reshape back to scalar
        MemoryView<float[]> backToScalar = vector.view(Shape.of());
        assertEquals(Shape.of(), backToScalar.shape());
    }

    @Test
    void testReshapeDataIntegrity() {
        // Verify that reshape doesn't change the logical data
        float[] floats = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(floats), Shape.of(3, 4));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        // Original: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
        assertEquals(0.0f, readFloat(memoryAccess, view, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, view, 1, 1));
        assertEquals(11.0f, readFloat(memoryAccess, view, 2, 3));

        // Reshape to (2, 6)
        MemoryView<float[]> reshaped = view.view(Shape.of(2, 6));
        // Should be: [[0,1,2,3,4,5], [6,7,8,9,10,11]]
        assertEquals(0.0f, readFloat(memoryAccess, reshaped, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, reshaped, 0, 5));
        assertEquals(6.0f, readFloat(memoryAccess, reshaped, 1, 0));
        assertEquals(11.0f, readFloat(memoryAccess, reshaped, 1, 5));

        // Reshape to (12,) - flat
        MemoryView<float[]> flat = view.view(Shape.of(12));
        for (int i = 0; i < 12; i++) {
            assertEquals((float) i, readFloat(memoryAccess, flat, i));
        }
    }

    @Test
    void testValidReshapeSameTotalElements() {
        float[] data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(data), Shape.of(3, 4));

        // Reshape to different shapes with same total elements (12)
        assertDoesNotThrow(() -> view.view(Shape.of(12)));
        assertDoesNotThrow(() -> view.view(Shape.of(6, 2)));
        assertDoesNotThrow(() -> view.view(Shape.of(2, 3, 2)));
        assertDoesNotThrow(() -> view.view(Shape.of(1, 12, 1)));
    }

    @Test
    void testInvalidReshapeDifferentTotalElements() {
        float[] data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(data), Shape.of(3, 4));

        // Attempt to reshape to shapes with different total elements
        assertThrows(IllegalArgumentException.class, () -> view.view(Shape.of(5)));
        assertThrows(IllegalArgumentException.class, () -> view.view(Shape.of(2, 5)));
        assertThrows(IllegalArgumentException.class, () -> view.view(Shape.of(3, 3)));
    }

    @Test
    void testReshapePreservesData() {
        float[] data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(data), Shape.of(3, 4));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        // Original 3x4 view
        assertEquals(0.0f, readFloat(memoryAccess, view, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, view, 1, 1));
        assertEquals(11.0f, readFloat(memoryAccess, view, 2, 3));

        // Reshape to 2x6
        MemoryView<float[]> reshaped2x6 = view.view(Shape.of(2, 6));
        assertEquals(0.0f, readFloat(memoryAccess, reshaped2x6, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, reshaped2x6, 0, 5));
        assertEquals(6.0f, readFloat(memoryAccess, reshaped2x6, 1, 0));
        assertEquals(11.0f, readFloat(memoryAccess, reshaped2x6, 1, 5));

        // Reshape to 4x3
        MemoryView<float[]> reshaped4x3 = view.view(Shape.of(4, 3));
        assertEquals(0.0f, readFloat(memoryAccess, reshaped4x3, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, reshaped4x3, 1, 2));
        assertEquals(6.0f, readFloat(memoryAccess, reshaped4x3, 2, 0));
        assertEquals(11.0f, readFloat(memoryAccess, reshaped4x3, 3, 2));
    }

    @Test
    void testReshapeEmptyView() {
        float[] emptyData = {};
        MemoryView<float[]> emptyView =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(emptyData), Shape.of(0));

        // Valid reshapes of empty view
        assertDoesNotThrow(() -> emptyView.view(Shape.of(0, 0)));
        assertDoesNotThrow(() -> emptyView.view(Shape.of(0, 1, 0)));

        // Invalid reshapes
        assertThrows(IllegalArgumentException.class, () -> emptyView.view(Shape.of(1)));
    }

    @Test
    void testReshapeNonContiguousView() {
        float[] data = arange(12);
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(data), Shape.of(3, 4));

        // Create a non-contiguous view by taking every other row: rows 0 and 2
        MemoryView<float[]> sliced = view.slice(0, 0, 3, 2);
        assertFalse(sliced.isContiguous());
    }

    @Test
    void testReshapeNonContiguousView2() {
        float[] data = arange(12);
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(data), Shape.of(3, 4));

        // Create a slice by taking the first two rows - this SHOULD be contiguous
        MemoryView<float[]> sliced = view.slice(0, 0, 2); // First two rows: (2, 4)

        // This slice should be contiguous because it takes consecutive memory
        assertTrue(sliced.isContiguous(), "First two rows should be contiguous");

        // Should be able to reshape freely since it's contiguous
        assertDoesNotThrow(() -> sliced.view(Shape.of(2, 4)));
        assertDoesNotThrow(() -> sliced.view(Shape.of(8)));
        assertDoesNotThrow(() -> sliced.view(Shape.of(2, 2, 2)));
    }

    @Test
    void testReshapeNonContiguousView3() {
        float[] data = arange(12);
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(data), Shape.of(3, 4));

        // Create a non-contiguous view by taking every other element in the last dimension
        MemoryView<float[]> sliced = view.slice(1, 0, 4, 2); // Every other element in dim 1
        // This creates strides like (16, 8) instead of (16, 4)

        assertFalse(sliced.isContiguous());

        // Should throw when trying to reshape non-contiguous data
        assertThrows(IllegalArgumentException.class, () -> sliced.view(Shape.of(6)));
    }

    /**
     * Tests that strided slices with gaps cannot be reshaped (except to same shape).
     *
     * <p>Taking every other row (rows 0 and 2) creates shape (2, 4) with stride (8, 1). This layout
     * accesses offsets {0,1,2,3, 8,9,10,11} - there's a gap at offsets 4-7. Span: (2-1)*8 + (4-1)*1
     * = 11 ≠ 8-1 = 7, so not contiguous.
     *
     * <p>With CuTe semantics, reshapes that change dimensions fail because the memory has gaps.
     * Same-shape "reshape" succeeds as it's effectively a no-op.
     */
    @Test
    void testReshapeNonContiguousView4() {
        float[] data = arange(12);
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(data), Shape.of(3, 4));

        // Create a non-contiguous view by taking every other row (rows 0 and 2)
        MemoryView<float[]> sliced = view.slice(0, 0, 3, 2); // shape (2, 4), stride (8, 1)
        assertFalse(sliced.isContiguous());

        // Same-shape "reshape" is a no-op - succeeds trivially
        assertDoesNotThrow(() -> sliced.view(Shape.of(2, 4)));

        // Reshapes that change dimensions fail - memory has gaps
        assertThrows(IllegalArgumentException.class, () -> sliced.view(Shape.of(2, 2, 2)));
        assertThrows(IllegalArgumentException.class, () -> sliced.view(Shape.of(8)));
    }

    @Test
    void testViewContiguous() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(floats), Shape.of(2, 3, 4));

        MemoryView<float[]> viewed = view.view(Shape.of(6, 4));
        assertEquals(Shape.of(6, 4), viewed.shape());
        assertTrue(viewed.isContiguous());
        assertSame(view.memory(), viewed.memory());
        assertEquals(view.byteOffset(), viewed.byteOffset());
    }

    @Test
    void testViewNonContiguousRejectsCopy() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view =
                MemoryViewFactory.rowMajor(
                        DataType.FP32, MemoryFactory.ofFloats(floats), Shape.of(2, 3, 4));
        MemoryView<float[]> sliced = view.slice(2, 0, 4, 2);

        assertThrows(IllegalArgumentException.class, () -> sliced.view(Shape.of(6)));
    }
}
