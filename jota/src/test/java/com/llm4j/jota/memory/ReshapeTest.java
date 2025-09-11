package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;
import com.llm4j.jota.memory.impl.MemoryAccessFactory;
import com.llm4j.jota.memory.impl.MemoryFactory;
import com.llm4j.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

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
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(2, 3, 5), DataType.F32, MemoryFactory.ofFloats(floats));

        // Slice last dimension and reshape (squeeze)
        MemoryView<float[]> sliced = view.slice(-1, 0, 1); // Shape: (2, 3, 1)
        assertEquals(Shape.of(2, 3, 1), sliced.shape());

        MemoryView<float[]> reshaped = sliced.reshape(Shape.of(2, 3)); // Remove singleton
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
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(2, 3, 4), DataType.F32, MemoryFactory.ofFloats(floats));

        // Reshape to different dimensions with same total elements
        MemoryView<float[]> reshaped = view.reshape(Shape.of(6, 4));
        assertEquals(Shape.of(6, 4), reshaped.shape());
        assertTrue(reshaped.isContiguous());

        // Reshape to 1D
        MemoryView<float[]> flat = view.reshape(Shape.of(24));
        assertEquals(Shape.of(24), flat.shape());
        assertTrue(flat.isContiguous());

        // Reshape to higher dimensions
        MemoryView<float[]> higher = view.reshape(Shape.of(2, 2, 3, 2));
        assertEquals(Shape.of(2, 2, 3, 2), higher.shape());
        assertTrue(higher.isContiguous());
    }

    @Test
    void testReshapeWithSingletons() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(2, 3, 4), DataType.F32, MemoryFactory.ofFloats(floats));

        // Add singleton dimensions (unsqueeze)
        MemoryView<float[]> unsqueezed = view.reshape(Shape.of(1, 2, 3, 4, 1));
        assertEquals(Shape.of(1, 2, 3, 4, 1), unsqueezed.shape());

        // Remove singleton dimensions (squeeze) - start with view that has singletons
        MemoryView<float[]> withSingletons = MemoryViewFactory.of(Shape.of(1, 2, 3, 4, 1), DataType.F32, MemoryFactory.ofFloats(floats));
        MemoryView<float[]> squeezed = withSingletons.reshape(Shape.of(2, 3, 4));
        assertEquals(Shape.of(2, 3, 4), squeezed.shape());
    }

    @Test
    void testReshapeDimensionCollapsing() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(2, 3, 4), DataType.F32, MemoryFactory.ofFloats(floats));

        // Collapse first two dimensions
        MemoryView<float[]> collapsed = view.reshape(Shape.of(6, 4));
        assertEquals(Shape.of(6, 4), collapsed.shape());

        // Collapse last two dimensions
        MemoryView<float[]> collapsed2 = view.reshape(Shape.of(2, 12));
        assertEquals(Shape.of(2, 12), collapsed2.shape());

        // Collapse all dimensions
        MemoryView<float[]> flat = view.reshape(Shape.of(24));
        assertEquals(Shape.of(24), flat.shape());
    }

    @Test
    void testReshapeDimensionSplitting() {
        float[] floats = arange(24);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(6, 4), DataType.F32, MemoryFactory.ofFloats(floats));

        // Split first dimension
        MemoryView<float[]> split = view.reshape(Shape.of(2, 3, 4));
        assertEquals(Shape.of(2, 3, 4), split.shape());

        // Split second dimension
        MemoryView<float[]> split2 = view.reshape(Shape.of(6, 2, 2));
        assertEquals(Shape.of(6, 2, 2), split2.shape());

        // Split into more dimensions
        MemoryView<float[]> split3 = view.reshape(Shape.of(2, 3, 2, 2));
        assertEquals(Shape.of(2, 3, 2, 2), split3.shape());
    }

    @Test
    void testReshapeErrorCases() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(2, 3, 4), DataType.F32, MemoryFactory.ofFloats(floats));

        // Test element count mismatch
        assertThrows(IllegalArgumentException.class, () -> {
            view.reshape(Shape.of(2, 3, 5)); // 30 elements vs 24
        });

        assertThrows(IllegalArgumentException.class, () -> {
            view.reshape(Shape.of(2, 3)); // 6 elements vs 24
        });

        assertThrows(IllegalArgumentException.class, () -> {
            view.reshape(Shape.of(25)); // 25 elements vs 24
        });
    }

    @Test
    void testReshapeNonContiguousErrorCases() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(2, 3, 4), DataType.F32, MemoryFactory.ofFloats(floats));

        // Create non-contiguous view by permuting
        MemoryView<float[]> permuted = view.permute(0, 2, 1); // Shape: (2, 4, 3)
        assertFalse(permuted.isContiguous());

        // Some reshapes should fail on non-contiguous views
        assertThrows(IllegalArgumentException.class, () -> {
            permuted.reshape(Shape.of(8, 3)); // This would require copying
        });
    }

    @Test
    void testReshapePreservesMemory() {
        float[] floats = arange(2 * 3 * 4);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(2, 3, 4), DataType.F32, MemoryFactory.ofFloats(floats));

        MemoryView<float[]> reshaped = view.reshape(Shape.of(6, 4));

        // Should reference the same memory
        assertSame(view.memory(), reshaped.memory());
        assertEquals(view.byteOffset(), reshaped.byteOffset());
        assertEquals(view.dataType(), reshaped.dataType());
    }

    @Test
    void testReshapeZeroDimensions() {
        float[] floats = arange(1);
        MemoryView<float[]> scalar = MemoryViewFactory.of(Shape.of(), DataType.F32, MemoryFactory.ofFloats(floats));

        // Reshape scalar to 1D with single element
        MemoryView<float[]> vector = scalar.reshape(Shape.of(1));
        assertEquals(Shape.of(1), vector.shape());

        // Reshape back to scalar
        MemoryView<float[]> backToScalar = vector.reshape(Shape.of());
        assertEquals(Shape.of(), backToScalar.shape());
    }

    @Test
    void testReshapeDataIntegrity() {
        // Verify that reshape doesn't change the logical data
        float[] floats = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(floats));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        // Original: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
        assertEquals(0.0f, readFloat(memoryAccess, view, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, view, 1, 1));
        assertEquals(11.0f, readFloat(memoryAccess, view, 2, 3));

        // Reshape to (2, 6)
        MemoryView<float[]> reshaped = view.reshape(Shape.of(2, 6));
        // Should be: [[0,1,2,3,4,5], [6,7,8,9,10,11]]
        assertEquals(0.0f, readFloat(memoryAccess, reshaped, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, reshaped, 0, 5));
        assertEquals(6.0f, readFloat(memoryAccess, reshaped, 1, 0));
        assertEquals(11.0f, readFloat(memoryAccess, reshaped, 1, 5));

        // Reshape to (12,) - flat
        MemoryView<float[]> flat = view.reshape(Shape.of(12));
        for (int i = 0; i < 12; i++) {
            assertEquals((float) i, readFloat(memoryAccess, flat, i));
        }
    }

    @Test
    void testValidReshapeSameTotalElements() {
        float[] data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(data));

        // Reshape to different shapes with same total elements (12)
        assertDoesNotThrow(() -> view.reshape(Shape.of(12)));
        assertDoesNotThrow(() -> view.reshape(Shape.of(6, 2)));
        assertDoesNotThrow(() -> view.reshape(Shape.of(2, 3, 2)));
        assertDoesNotThrow(() -> view.reshape(Shape.of(1, 12, 1)));
    }

    @Test
    void testInvalidReshapeDifferentTotalElements() {
        float[] data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(data));

        // Attempt to reshape to shapes with different total elements
        assertThrows(IllegalArgumentException.class, () -> view.reshape(Shape.of(5)));
        assertThrows(IllegalArgumentException.class, () -> view.reshape(Shape.of(2, 5)));
        assertThrows(IllegalArgumentException.class, () -> view.reshape(Shape.of(3, 3)));
    }

    @Test
    void testReshapePreservesData() {
        float[] data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(data));
        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();

        // Original 3x4 view
        assertEquals(0.0f, readFloat(memoryAccess, view, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, view, 1, 1));
        assertEquals(11.0f, readFloat(memoryAccess, view, 2, 3));

        // Reshape to 2x6
        MemoryView<float[]> reshaped2x6 = view.reshape(Shape.of(2, 6));
        assertEquals(0.0f, readFloat(memoryAccess, reshaped2x6, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, reshaped2x6, 0, 5));
        assertEquals(6.0f, readFloat(memoryAccess, reshaped2x6, 1, 0));
        assertEquals(11.0f, readFloat(memoryAccess, reshaped2x6, 1, 5));

        // Reshape to 4x3
        MemoryView<float[]> reshaped4x3 = view.reshape(Shape.of(4, 3));
        assertEquals(0.0f, readFloat(memoryAccess, reshaped4x3, 0, 0));
        assertEquals(5.0f, readFloat(memoryAccess, reshaped4x3, 1, 2));
        assertEquals(6.0f, readFloat(memoryAccess, reshaped4x3, 2, 0));
        assertEquals(11.0f, readFloat(memoryAccess, reshaped4x3, 3, 2));
    }

    @Test
    void testReshapeEmptyView() {
        float[] emptyData = {};
        MemoryView<float[]> emptyView = MemoryViewFactory.of(Shape.of(0), DataType.F32, MemoryFactory.ofFloats(emptyData));

        // Valid reshapes of empty view
        assertDoesNotThrow(() -> emptyView.reshape(Shape.of(0, 0)));
        assertDoesNotThrow(() -> emptyView.reshape(Shape.of(0, 1, 0)));

        // Invalid reshapes
        assertThrows(IllegalArgumentException.class, () -> emptyView.reshape(Shape.of(1)));
    }

    @Test
    void testReshapeNonContiguousView() {
        float[] data = arange(12);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(data));

        // Create a non-contiguous view by taking every other row: rows 0 and 2
        MemoryView<float[]> sliced = view.slice(0, 0, 3, 2);
        assertFalse(sliced.isContiguous());
    }

    @Test
    void testReshapeNonContiguousView2() {
        float[] data = arange(12);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(data));

        // Create a slice by taking the first two rows - this SHOULD be contiguous
        MemoryView<float[]> sliced = view.slice(0, 0, 2); // First two rows: (2, 4)

        // This slice should be contiguous because it takes consecutive memory
        assertTrue(sliced.isContiguous(), "First two rows should be contiguous");

        // Should be able to reshape freely since it's contiguous
        assertDoesNotThrow(() -> sliced.reshape(Shape.of(2, 4)));
        assertDoesNotThrow(() -> sliced.reshape(Shape.of(8)));
        assertDoesNotThrow(() -> sliced.reshape(Shape.of(2, 2, 2)));
    }

    @Test
    void testReshapeNonContiguousView3() {
        float[] data = arange(12);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(data));

        // Create a non-contiguous view by taking every other element in the last dimension
        MemoryView<float[]> sliced = view.slice(1, 0, 4, 2); // Every other element in dim 1
        // This creates strides like [16, 8] instead of [16, 4]

        assertFalse(sliced.isContiguous());

        // Should throw when trying to reshape non-contiguous data
        assertThrows(IllegalArgumentException.class, () -> sliced.reshape(Shape.of(6)));
    }

    @Test
    void testReshapeNonContiguousView4() {
        float[] data = arange(12);
        MemoryView<float[]> view = MemoryViewFactory.of(Shape.of(3, 4), DataType.F32, MemoryFactory.ofFloats(data));

        // Create a non-contiguous view by taking every other row
        MemoryView<float[]> sliced = view.slice(0, 0, 3, 2); // First two rows
        assertFalse(sliced.isContiguous());

        // Should be able to reshape as long as we don't change non-singleton dimensions
        assertDoesNotThrow(() -> sliced.reshape(Shape.of(2, 4)));
        assertDoesNotThrow(() -> sliced.reshape(Shape.of(2, 2, 2)));

        // Should throw when trying to change non-singleton dimensions
        assertThrows(IllegalArgumentException.class, () -> sliced.reshape(Shape.of(8)));
    }
}
