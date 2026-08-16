package com.qxotic.jota.memory;

class SliceTest {

    //    private final MemoryView<float[]> baseView = MemoryViewFactory.of(
    //            DataType.F32, MemoryFactory.ofFloats(createTestData()), Shape.of(2, 3, 5)
    //    );
    //
    //    private final MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //
    //    private float[] createTestData() {
    //        float[] arr = new float[2 * 3 * 5];
    //        for (int i = 0; i < arr.length; i++) {
    //            arr[i] = i;
    //        }
    //        return arr;
    //    }
    //
    //    private float readFloatAt(MemoryView<float[]> view, long... indices) {
    //        long offset = view.byteOffset();
    //        for (int i = 0; i < indices.length; i++) {
    //            offset += indices[i] * view.byteStride()[i];
    //        }
    //        return memoryAccess.readFloat(view.memory(), offset);
    //    }
    //
    //    @Test
    //    void testBasicSliceWithStep() {
    //        // Slice middle dimension with step=2
    //        MemoryView<float[]> sliced = baseView.slice(1, 0, 3, 2); // Shape: (2, 2, 5)
    //        assertEquals(Shape.of(2, 2, 5), sliced.shape());
    //
    //        // Verify selected elements
    //        assertEquals(0.0f, readFloatAt(sliced, 0, 0, 0));
    //        assertEquals(1.0f, readFloatAt(sliced, 0, 0, 1));
    //        assertEquals(10.0f, readFloatAt(sliced, 0, 1, 0));
    //        assertEquals(15.0f, readFloatAt(sliced, 1, 0, 0));
    //        assertEquals(25.0f, readFloatAt(sliced, 1, 1, 0));
    //    }
    //
    //    @Test
    //    void testNegativeStep() {
    //        // Reverse slice on last dimension
    //        MemoryView<float[]> reversed = baseView.slice(-1, 4, -1, -1); // Shape: (2, 3, 5)
    //        assertEquals(Shape.of(2, 3, 5), reversed.shape());
    //
    //        // Verify elements are reversed in the last dimension
    //        assertEquals(4.0f, readFloatAt(reversed, 0, 0, 0));
    //        assertEquals(3.0f, readFloatAt(reversed, 0, 0, 1));
    //        assertEquals(0.0f, readFloatAt(reversed, 0, 0, 4));
    //        assertEquals(29.0f, readFloatAt(reversed, 1, 2, 0));
    //        assertEquals(28.0f, readFloatAt(reversed, 1, 2, 1));
    //    }
    //
    //    @Test
    //    void testPartialSliceWithStep() {
    //        // Take every element from first dimension (step=1)
    //        MemoryView<float[]> sliced = baseView.slice(0, 0, 2, 1); // Shape: (2, 3, 5)
    //        assertEquals(Shape.of(2, 3, 5), sliced.shape());
    //
    //        // Verify we get the original elements
    //        assertEquals(0.0f, readFloatAt(sliced, 0, 0, 0));
    //        assertEquals(14.0f, readFloatAt(sliced, 0, 2, 4));
    //        assertEquals(15.0f, readFloatAt(sliced, 1, 0, 0));
    //        assertEquals(29.0f, readFloatAt(sliced, 1, 2, 4));
    //    }
    //
    //    @Test
    //    void testEdgeCaseSteps() {
    //        // Step larger than dimension size
    //        MemoryView<float[]> largeStep = baseView.slice(1, 0, 3, 5); // Shape: (2, 1, 5)
    //        assertEquals(Shape.of(2, 1, 5), largeStep.shape());
    //        assertEquals(0.0f, readFloatAt(largeStep, 0, 0, 0));
    //        assertEquals(15.0f, readFloatAt(largeStep, 1, 0, 0));
    //
    //        // Negative step with bounds
    //        MemoryView<float[]> negStep = baseView.slice(2, 4, 0, -2); // Shape: (2, 3, 2)
    //        assertEquals(Shape.of(2, 3, 2), negStep.shape());
    //        assertEquals(4.0f, readFloatAt(negStep, 0, 0, 0));
    //        assertEquals(2.0f, readFloatAt(negStep, 0, 0, 1));
    //        assertEquals(9.0f, readFloatAt(negStep, 0, 1, 0));
    //    }
    //
    //    @Test
    //    void testMemoryViewIntegrity() {
    //        // Get original value at position
    //        float originalValue = readFloatAt(baseView, 0, 1, 0);
    //
    //        // Create slice and modify through it
    //        MemoryView<float[]> sliced = baseView.slice(1, 1, 2, 1); // Shape: (2, 1, 5)
    //        memoryAccess.writeFloat(sliced.memory(),
    //                sliced.byteOffset(),
    //                -1.0f);
    //
    //        // Original should be modified
    //        assertEquals(-1.0f, readFloatAt(baseView, 0, 1, 0));
    //
    //        // Other elements unchanged
    //        assertEquals(0.0f, readFloatAt(baseView, 0, 0, 0));
    //        assertEquals(originalValue + 1, readFloatAt(baseView, 0, 1, 1));
    //    }
    //
    //    @Test
    //    void testStridedSliceOffsetCalculation() {
    //        // Verify byte offset and stride calculations
    //        MemoryView<float[]> sliced = baseView.slice(1, 1, 3, 2); // Shape: (2, 1, 5)
    //
    //        // Original stride for dim1 was 5*4=20 bytes (assuming float=4 bytes)
    //        // New stride should be 2*20=40 bytes
    //        assertEquals(40, sliced.byteStride()[1]);
    //
    //        // Offset should point to second row (index 1) of dim1
    //        assertEquals(5.0f, readFloatAt(sliced, 0, 0, 0));
    //        assertEquals(20.0f, readFloatAt(sliced, 1, 0, 0));
    //    }
    //
    //    @Test
    //    void testInvalidSteps() {
    //        // Zero step
    //        assertThrows(IllegalArgumentException.class, () -> baseView.slice(0, 0, 2, 0));
    //
    //        // Step incompatible with bounds
    //        assertThrows(IllegalArgumentException.class, () -> baseView.slice(0, 0, 2, -1));
    //        assertThrows(IllegalArgumentException.class, () -> baseView.slice(0, 1, 0, 1));
    //        assertThrows(IllegalArgumentException.class, () -> baseView.slice(0, 2, 0, -1));
    //    }

    //    private static final float EPSILON = 1e-6f;
    //    private final MemoryView<float[]> baseView = MemoryViewFactory.of(
    //            Shape.of(2, 3, 5),
    //            DataType.F32,
    //            MemoryFactory.ofFloats(arange(2 * 3 * 5))
    //    );
    //
    //    // Helper to create test array [0, 1, 2, ..., n-1]
    //    private static float[] arange(int n) {
    //        float[] arr = new float[n];
    //        for (int i = 0; i < n; i++) arr[i] = i;
    //        return arr;
    //    }
    //
    //    // Helper to assert array equality with epsilon tolerance
    //    private void assertFloatArrayEquals(float[] expected, float[] actual) {
    //        assertArrayEquals(expected, actual);
    //    }
    //
    //    @Test
    //    void testBasicSliceWithStep() {
    //        // Slice middle dimension with step=2
    //        MemoryView<float[]> sliced = baseView.slice(1, 0, 3, 2); // Shape: (2, 2, 5)
    //        assertEquals(Shape.of(2, 2, 5), sliced.shape());
    //
    //        // Verify selected elements
    //        float[] expectedRow0 = {0, 1, 2, 3, 4, 10, 11, 12, 13, 14};
    //        float[] expectedRow1 = {15, 16, 17, 18, 19, 25, 26, 27, 28, 29};
    //        assertFloatArrayEquals(expectedRow0, sliced.toArray(0));
    //        assertFloatArrayEquals(expectedRow1, sliced.toArray(1));
    //    }
    //
    //    @Test
    //    void testNegativeStep() {
    //        // Reverse slice on last dimension
    //        MemoryView<float[]> reversed = baseView.slice(-1, 4, -1, -1); // Shape: (2, 3, 5)
    //        assertEquals(Shape.of(2, 3, 5), reversed.shape());
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //
    //        // First element should be from end of array
    //        assertEquals(4.0f, readFloatAt(memoryAccess, reversed, 0, 0, 0));
    //        assertEquals(3.0f, readFloatAt(memoryAccess, reversed,0, 0, 1));
    //        assertEquals(29.0f, readFloatAt(memoryAccess, reversed,1, 2, 4));
    //    }
    //
    //    @Test
    //    void testPartialSliceWithStep() {
    //        // Take every 3rd element from first dimension
    //        MemoryView<float[]> sliced = baseView.slice(0, 0, 2, 1); // Shape: (2, 3, 5)
    //        assertEquals(Shape.of(2, 3, 5), sliced.shape());
    //
    //        // Verify we get the full original array (since step=1)
    //        assertFloatArrayEquals(baseView.toArray(), sliced.toArray());
    //    }
    //
    //    @Test
    //    void testEdgeCaseSteps() {
    //        // Step larger than dimension size
    //        MemoryView<float[]> largeStep = baseView.slice(1, 0, 3, 5); // Shape: (2, 1, 5)
    //        assertEquals(Shape.of(2, 1, 5), largeStep.shape());
    //        assertEquals(0.0f, largeStep.get(0, 0, 0));
    //        assertEquals(15.0f, largeStep.get(1, 0, 0));
    //
    //        // Negative step with bounds
    //        MemoryView<float[]> negStep = baseView.slice(2, 4, 0, -2); // Shape: (2, 3, 2)
    //        assertEquals(Shape.of(2, 3, 2), negStep.shape());
    //        assertEquals(4.0f, negStep.get(0, 0, 0));
    //        assertEquals(2.0f, negStep.get(0, 0, 1));
    //    }
    //
    //    @Test
    //    void testInvalidSteps() {
    //        // Zero step
    //        assertThrows(IllegalArgumentException.class, () -> baseView.slice(0, 0, 2, 0));
    //
    //        // Step incompatible with bounds
    //        assertThrows(IllegalArgumentException.class, () -> baseView.slice(0, 0, 2, -1)); //
    // Negative step with increasing bounds
    //        assertThrows(IllegalArgumentException.class, () -> baseView.slice(0, 1, 0, 1)); //
    // Positive step with decreasing bounds
    //    }
    //
    //    @Test
    //    void testMemoryViewIntegrity() {
    //        // Verify underlying memory isn't corrupted
    //        MemoryView<float[]> sliced = baseView.slice(1, 1, 3, 2); // Shape: (2, 1, 5)
    //        float[] originalCopy = baseView.toArray().clone();
    //
    //        // Modify through slice
    //        sliced.set(0, 0, 0, -1.0f);
    //
    //        // Original should be modified (shared memory)
    //        assertEquals(-1.0f, baseView.get(0, 1, 0));
    //
    //        // Other elements unchanged
    //        for (int i = 0; i < originalCopy.length; i++) {
    //            if (i != 5) { // Position we modified
    //                assertEquals(originalCopy[i], baseView.toArray()[i]);
    //            }
    //        }
    //    }
    //
    //    @Test
    //    void testReshapeAfterStridedSlice() {
    //        MemoryView<float[]> sliced = baseView.slice(-1, 0, 5, 2); // Shape: (2, 3, 3)
    //        assertEquals(Shape.of(2, 3, 3), sliced.shape());
    //
    //        // Reshape to 2D
    //        MemoryView<float[]> reshaped = sliced.view(Shape.of(2, 9));
    //        assertEquals(Shape.of(2, 9), reshaped.shape());
    //
    //        // Verify data
    //        float[] expectedRow0 = {0, 2, 4, 5, 7, 9, 10, 12, 14};
    //        float[] expectedRow1 = {15, 17, 19, 20, 22, 24, 25, 27, 29};
    //        assertFloatArrayEquals(expectedRow0, reshaped.toArray(0));
    //        assertFloatArrayEquals(expectedRow1, reshaped.toArray(1));
    //    }
    //
    //    @Test
    //    void testStridedSliceOffsetCalculation() {
    //        // Verify byte offset and stride calculations
    //        MemoryView<float[]> sliced = baseView.slice(1, 1, 3, 2); // Shape: (2, 1, 5)
    //
    //        // Original stride for dim1 was 5*4=20 bytes (assuming float=4 bytes)
    //        // New stride should be 2*20=40 bytes
    //        assertEquals(40, sliced.byteStrides()[1]);
    //
    //        // Offset should point to second row (index 1) of dim1
    //        float[] expected = {5, 6, 7, 8, 9, 20, 21, 22, 23, 24};
    //        assertFloatArrayEquals(expected, sliced.toArray());
    //    }
}
