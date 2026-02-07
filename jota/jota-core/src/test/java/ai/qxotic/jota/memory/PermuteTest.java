package ai.qxotic.jota.memory;

class PermuteTest {
    //
    //    @Test
    //    void testPermuteStridedContiguous() {
    //        // Source data with stride=2 in last dimension
    //        float[] data = new float[2 * 3 * 4 * 2];
    //        for (int i = 0; i < 2 * 3 * 4; i++) {
    //            data[i * 2] = i; // Skip every other element
    //        }
    //
    //        // Create strided view (shape [2,3,4] but only every other element)
    //        long[] strides = {3 * 4 * 2 * 4, 4 * 2 * 4, 2 * 4}; // in bytes (float=4 bytes)
    //        MemoryView<float[]> stridedView = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), 0, Shape.of(2, 3, 4),
    //                strides
    //        );
    //        assertFalse(stridedView.isContiguous());
    //
    //        // Permute dimensions
    //        MemoryView<float[]> permuted = stridedView.permute(2, 0, 1); // [4,2,3]
    //        assertEquals(Shape.of(4, 2, 3), permuted.shape());
    //
    //        // Verify new strides
    //        long[] expectedStrides = {
    //                2 * 4,          // Original last dimension stride (bytes)
    //                3 * 4 * 2 * 4,      // Original first dimension stride
    //                4 * 2 * 4         // Original middle dimension stride
    //        };
    //        assertArrayEquals(expectedStrides, permuted.byteStride());
    //
    //        // Verify data access
    //        MemoryAccess<float[]> access = MemoryAccessFactory.ofFloats();
    //        for (int i = 0; i < 2; i++) {
    //            for (int j = 0; j < 3; j++) {
    //                for (int k = 0; k < 4; k++) {
    //                    float expected = (float) (i * 3 * 4 + j * 4 + k);
    //                    float actual = access.readFloat(
    //                            permuted.memory(),
    //                            permuted.byteOffset() + k * expectedStrides[0]
    //                                    + i * expectedStrides[1]
    //                                    + j * expectedStrides[2]
    //                    );
    //                    assertEquals(expected, actual,
    //                            "Mismatch at permuted[" + k + "," + i + "," + j + "]");
    //                }
    //            }
    //        }
    //    }
    //
    //    @Test
    //    void testPermuteStridedNonContiguous() {
    //        // Create a view where only every other column is accessible
    //        float[] data = new float[5 * 10];
    //        for (int i = 0; i < 5; i++) {
    //            for (int j = 0; j < 5; j++) {
    //                data[i * 10 + j * 2] = i * 5 + j; // Skip every other column
    //            }
    //        }
    //
    //        // Strided view (5x5 matrix with column stride 2)
    //        long[] strides = {10 * 4, 2 * 4}; // in bytes (row_stride=10, col_stride=2)
    //        MemoryView<float[]> stridedView = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), 0, Shape.of(5, 5),
    //                strides
    //        );
    //
    //        // Permute -> becomes row stride 2, column stride 10
    //        MemoryView<float[]> transposed = stridedView.permute(1, 0);
    //        assertArrayEquals(new long[]{2 * 4, 10 * 4}, transposed.byteStride());
    //
    //        // Verify data
    //        MemoryAccess<float[]> access = MemoryAccessFactory.ofFloats();
    //        for (int i = 0; i < 5; i++) {
    //            for (int j = 0; j < 5; j++) {
    //                float expected = j * 5 + i; // transposed indices
    //                float actual = access.readFloat(
    //                        transposed.memory(),
    //                        transposed.byteOffset() + i * 2 * 4 + j * 10 * 4
    //                );
    //                assertEquals(expected, actual,
    //                        "Mismatch at [" + j + "," + i + "]");
    //            }
    //        }
    //    }
    //
    //    @Test
    //    void testPermuteBroadcastedDimension() {
    //        // Create original (3, 4) data
    //        float[] data = arange(3 * 4);  // [0,1,2,3,4,5,6,7,8,9,10,11]
    //
    //        // Create a view where dimension 1 is broadcasted (has stride 0)
    //        // This simulates taking the first column and broadcasting it
    //        long[] strides = {4 * 4, 0}; // Dimension 1 has stride 0 (broadcasted)
    //
    //        // Shape (3, 4) but with stride 0 for dimension 1
    //        MemoryView<float[]> view = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), 0, Shape.of(3, 4),    // NOT (3,1)
    // - we want (3,4) with broadcasting
    //                strides           // [16, 0] - second dim is broadcasted
    //        );
    //
    //        // Permute: (3,4) -> (4,3)
    //        MemoryView<float[]> permuted = view.permute(1, 0);
    //        assertEquals(Shape.of(4, 3), permuted.shape());
    //
    //        // After permute: strides (16, 0) -> (0, 16)
    //        assertArrayEquals(new long[]{0, 4 * 4}, permuted.byteStride());
    //    }
    //
    //    @Test
    //    void testPermuteWithOffset() {
    //        // Create a larger buffer and select a sub-view
    //        float[] data = new float[100];
    //        for (int i = 0; i < 20; i++) {
    //            data[10 + i] = i; // Our view starts at offset 10
    //        }
    //
    //        // 4x5 matrix starting at byte offset 10*4=40
    //        MemoryView<float[]> view = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), 10 * 4, Shape.of(4, 5),
    //                new long[]{5 * 4, 4} // Row-major contiguous
    //                // byteOffset = 10 floats * 4 bytes
    //        );
    //
    //        // Transpose with offset
    //        MemoryView<float[]> transposed = view.permute(1, 0);
    //        assertEquals(Shape.of(5, 4), transposed.shape());
    //        assertEquals(10 * 4, transposed.byteOffset()); // Offset should be preserved
    //
    //        // Verify strides are swapped
    //        assertArrayEquals(new long[]{4, 5 * 4}, transposed.byteStride());
    //
    //        MemoryAccess<float[]> access = MemoryAccessFactory.ofFloats();
    //        for (int i = 0; i < 4; i++) {
    //            for (int j = 0; j < 5; j++) {
    //                float expected = (float) (i * 5 + j);
    //                float actual = access.readFloat(
    //                        transposed.memory(),
    //                        transposed.byteOffset() + j * 4 + i * 5 * 4
    //                );
    //                assertEquals(expected, actual);
    //            }
    //        }
    //    }
}
