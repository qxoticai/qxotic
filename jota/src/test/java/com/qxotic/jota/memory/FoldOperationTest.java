package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.impl.ContextFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.AutoClose;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class FoldOperationTest {

    @AutoClose
    MemoryContext<float[]> context = ContextFactory.ofFloats();

    // Helper methods from the sketch
    public static MemoryView<float[]> of(float... floats) {
        return MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(floats), Layout.rowMajor(floats.length));
    }

    public static MemoryView<float[]> full(float value, Shape shape) {
        var floats = new float[Math.toIntExact(shape.size())];
        Arrays.fill(floats, value);
        return of(floats).view(shape);
    }

    public static MemoryView<float[]> zeros(Shape shape) {
        return full(0f, shape);
    }

    public static MemoryView<float[]> range(int fromInclusive, int toExclusive) {
        int n = toExclusive - fromInclusive;
        var floats = new float[n];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = (float) (fromInclusive + i);
        }
        return of(floats);
    }

    public static MemoryView<float[]> range(int toExclusive) {
        return range(0, toExclusive);
    }

    // Helper to extract values from a MemoryView for testing
    private float[] extractValues(MemoryView<float[]> view) {
        long totalElements = view.shape().size();
        float[] result = new float[Math.toIntExact(totalElements)];
        MemoryAccess<float[]> memoryAccess = context.memoryAccess();

        for (int i = 0; i < result.length; i++) {
            result[i] = memoryAccess.readFloat(view.memory(), view.byteOffset() + i * Float.BYTES);
        }
        return result;
    }

    @Test
    void testFold1D_Sum() {
        // Test: [1, 2, 3, 4] with sum and initial value 10
        var input = range(1, 5); // [1, 2, 3, 4]
        var output = zeros(Shape.scalar()); // scalar output

        context.floatOperations().fold(input, Float::sum, 10f, output, 0);

        float[] result = extractValues(output);
        assertEquals(20f, result[0], 1e-6f); // 10 + 1 + 2 + 3 + 4 = 20
    }

    @Test
    void testFold1D_Product() {
        // Test: [2, 3, 4] with product and initial value 1
        var input = range(2, 5); // [2, 3, 4]
        var output = zeros(Shape.of()); // scalar output

        context.floatOperations().fold(input, (a, b) -> a * b, 1f, output, 0);

        float[] result = extractValues(output);
        assertEquals(24f, result[0], 1e-6f); // 1 * 2 * 3 * 4 = 24
    }

    @Test
    void testFold1D_Max() {
        // Test: [5, 2, 8, 1] with max and initial value -infinity
        var input = of(5f, 2f, 8f, 1f);
        var output = zeros(Shape.of()); // scalar output

        context.floatOperations().fold(input, Float::max, Float.NEGATIVE_INFINITY, output, 0);

        float[] result = extractValues(output);
        assertEquals(8f, result[0], 1e-6f);
    }

    @Test
    void testFold1D_Min() {
        // Test: [5, 2, 8, 1] with min and initial value +infinity
        var input = of(5f, 2f, 8f, 1f);
        var output = zeros(Shape.of()); // scalar output

        context.floatOperations().fold(input, Float::min, Float.POSITIVE_INFINITY, output, 0);

        float[] result = extractValues(output);
        assertEquals(1f, result[0], 1e-6f);
    }

    @Test
    void testFold2D_Axis0() {
        // Test 2x3 matrix folding along axis 0
        // Input: [[0, 1, 2],
        //         [3, 4, 5]]
        // Expected output: [0+3, 1+4, 2+5] = [3, 5, 7] (with initial value 0)
        var input = range(6).view(Shape.of(2, 3));
        var output = zeros(Shape.of(3));

        context.floatOperations().fold(input, Float::sum, 0f, output, 0);

        float[] result = extractValues(output);
        assertArrayEquals(new float[]{3f, 5f, 7f}, result, 1e-6f);
    }

    @Test
    void testFold2D_Axis1() {
        // Test 2x3 matrix folding along axis 1
        // Input: [[0, 1, 2],
        //         [3, 4, 5]]
        // Expected output: [0+1+2, 3+4+5] = [3, 12] (with initial value 0)
        var input = range(6).view(Shape.of(2, 3));
        var output = zeros(Shape.of(2));

        context.floatOperations().fold(input, Float::sum, 0f, output, 1);

        float[] result = extractValues(output);
        assertArrayEquals(new float[]{3f, 12f}, result, 1e-6f);
    }

    @Test
    void testFold2D_ProductAxis0() {
        // Test 2x3 matrix folding along axis 0 with product
        // Input: [[1, 2, 3],
        //         [4, 5, 6]]
        // Expected output: [1*4, 2*5, 3*6] = [4, 10, 18] (with initial value 1)
        var input = range(1, 7).view(Shape.of(2, 3));
        var output = zeros(Shape.of(3));

        context.floatOperations().fold(input, (a, b) -> a * b, 1f, output, 0);

        float[] result = extractValues(output);
        assertArrayEquals(new float[]{4f, 10f, 18f}, result, 1e-6f);
    }

    @Test
    void testFold3D_Axis1() {
        // Test 2x3x2 tensor folding along axis 1
        // Input shape: [2, 3, 2], Output shape: [2, 2]
        var input = range(12).view(Shape.of(2, 3, 2));
        var output = zeros(Shape.of(2, 2));

        context.floatOperations().fold(input, Float::sum, 0f, output, 1);

        float[] result = extractValues(output);
        // For each [i, j] in output: sum over input[i, :, j]
        // output[0,0] = 0 + 2 + 4 = 6
        // output[0,1] = 1 + 3 + 5 = 9
        // output[1,0] = 6 + 8 + 10 = 24
        // output[1,1] = 7 + 9 + 11 = 27
        assertArrayEquals(new float[]{6f, 9f, 24f, 27f}, result, 1e-6f);
    }

    @Test
    void testFold3D_Axis2() {
        // Test 2x3x2 tensor folding along axis 2
        // Input shape: [2, 3, 2], Output shape: [2, 3]
        var input = range(12).view(Shape.of(2, 3, 2));
        var output = zeros(Shape.of(2, 3));

        context.floatOperations().fold(input, Float::sum, 0f, output, 2);

        float[] result = extractValues(output);
        // For each [i, j] in output: sum over input[i, j, :]
        // output[0,0] = 0 + 1 = 1
        // output[0,1] = 2 + 3 = 5
        // output[0,2] = 4 + 5 = 9
        // output[1,0] = 6 + 7 = 13
        // output[1,1] = 8 + 9 = 17
        // output[1,2] = 10 + 11 = 21
        assertArrayEquals(new float[]{1f, 5f, 9f, 13f, 17f, 21f}, result, 1e-6f);
    }

    @Test
    void testFoldWithTranspose() {
        // Test folding on a transposed view (different strides)
        var input = range(6).view(Shape.of(2, 3));
        var transposed = input.permute(1, 0); // Shape becomes [3, 2]
        var output = zeros(Shape.of(3));

        context.floatOperations().fold(transposed, Float::sum, 0f, output, 1);

        float[] result = extractValues(output);
        // Transposed matrix: [[0, 3],
        //                     [1, 4],
        //                     [2, 5]]
        // Folding along axis 1: [0+3, 1+4, 2+5] = [3, 5, 7]
        assertArrayEquals(new float[]{3f, 5f, 7f}, result, 1e-6f);
    }

//    @Test
//    void testFoldWithSlice() {
//        // Test folding on a sliced view
//        var input = range(12).view(Shape.of(3, 4));
//        var sliced = input.slice(0, 1, 3); // Take rows 1-2, shape becomes [2, 4]
//        var output = zeros(Shape.of(4));
//
//        context.floatOperations().fold(sliced, Float::sum, 0f, output, 0);
//
//        float[] result = extractValues(output);
//        // Sliced matrix: [[4, 5, 6, 7],
//        //                 [8, 9, 10, 11]]
//        // Folding along axis 0: [4+8, 5+9, 6+10, 7+11] = [12, 14, 16, 18]
//        assertArrayEquals(new float[]{12f, 14f, 16f, 18f}, result, 1e-6f);
//    }

    @Test
    void testFoldWithNonZeroInitialValue() {
        // Test that initial value is properly used
        var input = range(1, 4); // [1, 2, 3]
        var output = zeros(Shape.of());

        context.floatOperations().fold(input, Float::sum, 100f, output, 0);

        float[] result = extractValues(output);
        assertEquals(106f, result[0], 1e-6f); // 100 + 1 + 2 + 3 = 106
    }

    @Test
    void testFoldSingleElement() {
        // Test folding a single element
        var input = of(42f);
        var output = zeros(Shape.of());

        context.floatOperations().fold(input, Float::sum, 10f, output, 0);

        float[] result = extractValues(output);
        assertEquals(52f, result[0], 1e-6f); // 10 + 42 = 52
    }

    @Test
    void testFoldErrorCases() {
        var input = range(6).view(Shape.of(2, 3));
        var output = zeros(Shape.of(3));

        // Test invalid axis
        assertThrows(IllegalArgumentException.class, () -> {
            context.floatOperations().fold(input, Float::sum, 0f, output, -1);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            context.floatOperations().fold(input, Float::sum, 0f, output, 2);
        });

        // Test wrong output shape
        var wrongOutput = zeros(Shape.of(2, 3)); // Should be [3] when folding [2,3] along axis 0
        assertThrows(IllegalArgumentException.class, () -> {
            context.floatOperations().fold(input, Float::sum, 0f, wrongOutput, 0);
        });
    }

    @Test
    void testFoldLargeAxis() {
        // Test folding along a large axis to check performance and correctness
        var input = range(1000).view(Shape.of(10, 100));
        var output = zeros(Shape.of(10));

        context.floatOperations().fold(input, Float::sum, 0f, output, 1);

        float[] result = extractValues(output);
        // Each row sums to: row_start + (row_start+1) + ... + (row_start+99)
        // For row i: sum from (i*100) to (i*100+99) = 100*i*100 + (0+1+...+99) = 10000*i + 4950
        for (int i = 0; i < 10; i++) {
            float expected = 10000f * i + 4950f;
            assertEquals(expected, result[i], 1e-4f);
        }
    }

    @Test
    void testFoldPreservesNaN() {
        // Test that NaN values are properly handled
        var input = of(1f, Float.NaN, 3f);
        var output = zeros(Shape.of());

        context.floatOperations().fold(input, Float::sum, 0f, output, 0);

        float[] result = extractValues(output);
        assertTrue(Float.isNaN(result[0]));
    }

    @Test
    void testFoldPreservesInfinity() {
        // Test that infinity values are properly handled
        var input = of(1f, Float.POSITIVE_INFINITY, 3f);
        var output = zeros(Shape.of());

        context.floatOperations().fold(input, Float::sum, 0f, output, 0);

        float[] result = extractValues(output);
        assertEquals(Float.POSITIVE_INFINITY, result[0]);
    }
}
