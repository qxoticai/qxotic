package ai.qxotic.jota.memory;

class BroadcastTest {
    //
    //    @Test
    //    void testBroadcast() {
    //        float[] data = {1, 2, 3};
    //        MemoryView<float[]> vec = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.of(3)
    //        );
    //
    //        // Broadcast vector to matrix
    //        MemoryView<float[]> mat = vec.broadcast(Shape.of(4, 3));
    //        assertEquals(Shape.of(4, 3), mat.shape());
    //
    //        MemoryAccess<float[]> access = MemoryAccessFactory.ofFloats();
    //
    //        // Verify broadcasting worked
    //        for (int i = 0; i < 4; i++) {
    //            assertEquals(1.0f, readFloat(access, mat, i, 0));
    //            assertEquals(2.0f, readFloat(access, mat, i, 1));
    //            assertEquals(3.0f, readFloat(access, mat, i, 2));
    //        }
    //    }
    //
    //    @Test
    //    void testBroadcastInvalid() {
    //        MemoryView<float[]> mat = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(new float[12]), Shape.of(3, 4)
    //        );
    //
    //        assertThrows(IllegalArgumentException.class,
    //                () -> mat.broadcast(Shape.of(3))); // Fewer dimensions
    //        assertThrows(IllegalArgumentException.class,
    //                () -> mat.broadcast(Shape.of(3, 5))); // Incompatible dims
    //    }
    //
    //    @Test
    //    void testBroadcastVectorToMatrix() {
    //        float[] data = {1.0f, 2.0f, 3.0f};
    //        MemoryView<float[]> vec = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.of(3)
    //        );
    //
    //        // Broadcast (3) -> (2, 3)
    //        MemoryView<float[]> result = vec.broadcast(Shape.of(2, 3));
    //
    //        assertEquals(Shape.of(2, 3), result.shape());
    //        assertFalse(result.isContiguous());
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //
    //        // Verify all rows contain original vector
    //        for (int i = 0; i < 2; i++) {
    //            assertEquals(1.0f, memoryAccess.readFloat(result.memory(),
    // calculateByteOffset(result, i, 0)));
    //            assertEquals(2.0f, memoryAccess.readFloat(result.memory(),
    // calculateByteOffset(result, i, 1)));
    //            assertEquals(3.0f, memoryAccess.readFloat(result.memory(),
    // calculateByteOffset(result, i, 2)));
    //        }
    //    }
    //
    //    @Test
    //    void testBroadcastMatrixToHigherRank() {
    //        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
    //        MemoryView<float[]> mat = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.of(2, 2)
    //        );
    //
    //        // Broadcast (2, 2) -> (3, 2, 2)
    //        MemoryView<float[]> result = mat.broadcast(Shape.of(3, 2, 2));
    //
    //        assertEquals(Shape.of(3, 2, 2), result.shape());
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //
    //        // Verify all batches contain original matrix
    //        for (int i = 0; i < 3; i++) {
    //            assertEquals(1.0f, readFloat(memoryAccess, result, i, 0, 0));
    //            assertEquals(2.0f, readFloat(memoryAccess, result, i, 0, 1));
    //            assertEquals(3.0f, readFloat(memoryAccess, result, i, 1, 0));
    //            assertEquals(4.0f, readFloat(memoryAccess, result, i, 1, 1));
    //        }
    //    }
    //
    //    @Test
    //    void testBroadcastWithExistingSingletons() {
    //        float[] data = {5.0f, 10.0f};
    //        MemoryView<float[]> vec = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.of(2, 1)  // Already has
    // singleton dimension
    //        );
    //
    //        // Broadcast (2, 1) -> (2, 4)
    //        MemoryView<float[]> result = vec.broadcast(Shape.of(2, 4));
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //
    //        assertEquals(Shape.of(2, 4), result.shape());
    //        assertEquals(5.0f, memoryAccess.readFloat(result.memory(), calculateByteOffset(result,
    // 0, 0)));
    //        assertEquals(5.0f, memoryAccess.readFloat(result.memory(), calculateByteOffset(result,
    // 0, 3))); // Broadcasted
    //        assertEquals(10.0f, memoryAccess.readFloat(result.memory(),
    // calculateByteOffset(result, 1, 2)));
    //    }
    //
    //    @Test
    //    void testBroadcastScalar() {
    //        float[] data = {42.0f};
    //        MemoryView<float[]> scalar = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.scalar()
    //        );
    //
    //        // Broadcast () -> (2, 3, 4)
    //        MemoryView<float[]> result = scalar.broadcast(Shape.of(2, 3, 4));
    //
    //        assertEquals(Shape.of(2, 3, 4), result.shape());
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //
    //        // Every position should have the scalar value
    //        for (int i = 0; i < 2; i++) {
    //            for (int j = 0; j < 3; j++) {
    //                for (int k = 0; k < 4; k++) {
    //                    assertEquals(42.0f, readFloat(memoryAccess, result, i, j, k));
    //                }
    //            }
    //        }
    //    }
    //
    //    @Test
    //    void testInvalidBroadcastDimensions() {
    //        float[] data = {1.0f, 2.0f, 3.0f};
    //        MemoryView<float[]> vec = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.of(3)
    //        );
    //
    //        // Incompatible dimension
    //        assertThrows(IllegalArgumentException.class,
    //                () -> vec.broadcast(Shape.of(4)));  // Different size
    //
    //        assertThrows(IllegalArgumentException.class,
    //                () -> vec.broadcast(Shape.of(2, 4)));  // Non-broadcastable dimension
    //    }
    //
    //    @Test
    //    void testBroadcastNonContiguousSource() {
    //        float[] underlying = {0, 1, 0, 2, 0, 3, 0, 4, 0, 5};
    //        long[] strides = {2 * 4}; // stride=2 floats
    //        MemoryView<float[]> strided = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(underlying), 4, Shape.of(5),
    //                strides
    //                // Start at index 1 (4 bytes offset)
    //        );
    //
    //        // Broadcast (5) -> (3, 5)
    //        MemoryView<float[]> result = strided.broadcast(Shape.of(3, 5));
    //
    //        assertEquals(Shape.of(3, 5), result.shape());
    //        assertArrayEquals(new long[]{0, 8}, result.byteStride()); // outer stride=0
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //
    //        // Verify strided access is preserved
    //        assertEquals(1.0f, readFloat(memoryAccess, result, 0, 0));
    //        assertEquals(3.0f, readFloat(memoryAccess, result, 1, 2));
    //    }
    //
    //    @Test
    //    void testBroadcastChainedOperations() {
    //        float[] data = {7.0f, 8.0f};
    //        MemoryView<float[]> vec = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.of(2)
    //        );
    //
    //        // Chain broadcast with other operations
    //        MemoryView<float[]> result = vec
    //                .broadcast(Shape.of(1, 2))  // (1, 2)
    //                .permute(1, 0)              // (2, 1)
    //                .broadcast(Shape.of(2, 3));  // (2, 3)
    //
    //        assertEquals(Shape.of(2, 3), result.shape());
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //        assertEquals(7.0f, readFloat(memoryAccess, result, 0, 0));
    //        assertEquals(7.0f, readFloat(memoryAccess, result, 0, 2));
    //        assertEquals(8.0f, readFloat(memoryAccess, result, 1, 1));
    //    }
    //
    //    @Test
    //    void testBroadcastWithOffsetMemory() {
    //        float[] largeBuffer = new float[100];
    //        largeBuffer[50] = 9.0f;
    //        largeBuffer[51] = 10.0f;
    //
    //        MemoryView<float[]> view = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(largeBuffer), 50 * 4, Shape.of(2),
    //                new long[]{4}
    //                // byte offset to [50]
    //        );
    //
    //        // Broadcast with offset
    //        MemoryView<float[]> result = view.broadcast(Shape.of(4, 2));
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //        assertEquals(9.0f, readFloat(memoryAccess, result, 2, 0));
    //        assertEquals(10.0f, readFloat(memoryAccess, result, 3, 1));
    //    }
    //
    //    @Test
    //    void testBroadcastWithNegativeStride() {
    //        float[] underlyingData = {1.0f, 2.0f, 3.0f, 4.0f};
    //        // Create a view that is the reverse of the original array
    //        MemoryView<float[]> reversed = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(underlyingData), 3 * 4, Shape.of(4),
    //                new long[]{-4} // negative stride
    //                // start at last element (offset=12)
    //        );
    //
    //        // Broadcast reversed to shape (3, 4)
    //        MemoryView<float[]> broadcasted = reversed.broadcast(Shape.of(3, 4));
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //
    //        // Expected: each row is [4,3,2,1]
    //        assertEquals(4.0f, readFloat(memoryAccess, broadcasted, 0, 0)); // first row, first
    // col
    //        assertEquals(1.0f, readFloat(memoryAccess, broadcasted, 0, 3)); // first row, last col
    //        assertEquals(4.0f, readFloat(memoryAccess, broadcasted, 2, 0)); // last row, first col
    //    }
    //
    //    @Test
    //    void testBroadcastEmptyTensor() {
    //        // Empty tensor: shape (0) (but total elements=0)
    //        float[] data = new float[0];
    //        MemoryView<float[]> empty = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.of(0)
    //        );
    //
    //        // Broadcasting to (3, 0) (also empty) should be allowed
    //        MemoryView<float[]> result = empty.broadcast(Shape.of(3, 0));
    //        assertEquals(Shape.of(3, 0), result.shape());
    //        assertEquals(0, result.shape().size());
    //
    //        // But broadcasting to non-empty shape should throw because the original is empty and
    // target isn't?
    //        // Actually, in the example (0) -> (3, 0) is okay because the target is empty.
    //        // But if we try to broadcast (0) to (3) (non-empty) that would be invalid because we
    // cannot expand 0 to 3.
    //        assertThrows(IllegalArgumentException.class,
    //                () -> empty.broadcast(Shape.of(3)));
    //    }
    //
    //    @Test
    //    void testBroadcastToZero() {
    //        float[] data = {1.0f, 2.0f, 3.0f};
    //        MemoryView<float[]> vec = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.of(3)
    //        );
    //
    //        // Broadcast to shape [3,0] -> which is empty
    //        MemoryView<float[]> result = vec.broadcast(Shape.of(3, 0));
    //        assertEquals(Shape.of(3, 0), result.shape());
    //        assertEquals(0, result.shape().size());
    //    }
    //
    //    @Test
    //    void testBroadcastAlreadyBroadcasted() {
    //        // Tensor of shape [1,3] that is actually a broadcasted view over [3] with stride0 in
    // the first dimension.
    //        float[] data = {1.0f, 2.0f, 3.0f};
    //        MemoryView<float[]> alreadyBroadcasted = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), 0, Shape.of(1, 3),
    //                new long[]{0, 4} // first dim broadcasted
    //        );
    //
    //        // Broadcast to [4,1,3]: adds one dimension at front, then expands singleton dims
    //        MemoryView<float[]> result = alreadyBroadcasted.broadcast(Shape.of(4, 1, 3));
    //
    //        // Check new strides:
    //        //  - outermost dimension (size 4) should be stride0
    //        //  - next (size 1) is also stride0
    //        //  - innermost (size 3) has original stride 4
    //        assertArrayEquals(new long[]{0, 0, 4}, result.byteStride());
    //        MemoryAccess<float[]> memoryAccess = MemoryAccessFactory.ofFloats();
    //
    //        // Verify data
    //        for (int i = 0; i < 4; i++) {
    //            for (int j = 0; j < 1; j++) {
    //                for (int k = 0; k < 3; k++) {
    //                    if (k == 0) assertEquals(1.0f, readFloat(memoryAccess, result, i, j, k));
    //                    if (k == 1) assertEquals(2.0f, readFloat(memoryAccess, result, i, j, k));
    //                    if (k == 2) assertEquals(3.0f, readFloat(memoryAccess, result, i, j, k));
    //                }
    //            }
    //        }
    //    }
    //
    //    @Test
    //    void testBroadcastToNonEmptyThenToEmpty() {
    //        float[] data = {1.0f, 2.0f, 3.0f};
    //        MemoryView<float[]> vec = MemoryViewFactory.of(
    //                DataType.F32, MemoryFactory.ofFloats(data), Shape.of(1, 3)
    //        );
    //
    //        // Broadcast to shape [0,3] (empty)
    //        MemoryView<float[]> result = vec.broadcast(Shape.of(0, 3));
    //        assertEquals(Shape.of(0, 3), result.shape());
    //        assertEquals(0, result.shape().size());
    //    }
    //
    //    @Test
    //    void freshViewIsNotBroadcasted() {
    //        MemoryView<float[]> v = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(new
    // float[6]), Shape.of(2, 3));
    //        assertFalse(v.isBroadcasted(), "Fresh compact view should not be broadcasted (no zero
    // strides)");
    //    }
    //
    //    @Test
    //    void scalarBroadcastedTo2DHasZeroStrides() {
    //        MemoryView<float[]> scalar = MemoryViewFactory.of(DataType.F32,
    // MemoryFactory.ofFloats(42f), Shape.scalar());
    //        assertFalse(scalar.isBroadcasted(), "Scalar by itself is not broadcasted");
    //        MemoryView<float[]> b = scalar.broadcast(Shape.of(2, 3));
    //        assertTrue(b.isBroadcasted(), "Broadcasting scalar to (2,3) should produce zero
    // strides");
    //        assertFalse(b.isContiguous(), "Broadcasted views are not contiguous");
    //    }
    //
    //    @Test
    //    void expandAlongSingletonAxisIsBroadcasted() {
    //        // Start with (4,1) and expand to (4,3) along the last axis
    //        MemoryView<float[]> v = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(new
    // float[4]), Shape.of(4, 1));
    //        assertFalse(v.isBroadcasted());
    //        MemoryView<float[]> expanded = v.expand(Shape.of(4, 3));
    //        assertTrue(expanded.isBroadcasted(), "Expanding a singleton axis must create zero
    // stride");
    //        assertFalse(expanded.isContiguous());
    //    }
    //
    //    @Test
    //    void expandAfterPermuteIsBroadcastedOnThatAxis() {
    //        // Make a view with a singleton middle axis and then expand it
    //        MemoryView<float[]> v = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(new
    // float[12]), Shape.of(3, 1, 4));
    //        assertFalse(v.isBroadcasted());
    //        // Expand the singleton middle axis from 1 -> 7
    //        MemoryView<float[]> expanded = v.expand(Shape.of(3, 7, 4));
    //        assertTrue(expanded.isBroadcasted(), "Expanded singleton middle axis should have zero
    // stride");
    //    }
    //
    //    @Test
    //    void broadcastAddsLeadingAxisWhenNeeded() {
    //        // Starting from 1D view (size along last axis), broadcast requires aligning by
    // prepending singleton dims
    //        MemoryView<float[]> v = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(new
    // float[5]), Shape.of(1, 5));
    //        assertFalse(v.isBroadcasted());
    //        // Broadcast last axis 5 stays 5, first axis 1 -> 3
    //        MemoryView<float[]> b = v.broadcast(Shape.of(3, 5));
    //        assertTrue(b.isBroadcasted(), "Broadcasting along first axis should introduce a zero
    // stride axis");
    //    }
    //
    //    @Test
    //    void sliceWithIndexStrideIsNotBroadcasted() {
    //        MemoryView<float[]> v = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(new
    // float[18]), Shape.of(3, 6));
    //        // Take every other column: non-zero stride, but not zero-stride broadcasting
    //        MemoryView<float[]> colStride2 = v.slice(-1, 0, 6, 2);
    //        assertFalse(colStride2.isBroadcasted(), "Strided slicing is not broadcasting (no zero
    // strides)");
    //        assertFalse(colStride2.isContiguous(), "But it's not contiguous either");
    //    }
    //
    //    @Test
    //    void reshapeAddingSingletonAxisIsNotBroadcasted() {
    //        // Reshape should keep non-zero strides; adding a size-1 axis isn't broadcasting
    //        MemoryView<float[]> v = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(new
    // float[12]), Shape.of(3, 4));
    //        MemoryView<float[]> reshaped = v.view(Shape.of(1, 3, 4));
    //        assertFalse(reshaped.isBroadcasted(), "Reshape with singleton axis should not
    // introduce zero strides");
    //        assertFalse(v.transpose(0, 1).view(Shape.of(1, 4, 3)).isBroadcasted(), "Even after
    // transpose, reshape alone is not broadcasting");
    //    }
    //
    //    @Test
    //    void scalarBroadcastedTo3DIsBroadcasted() {
    //        MemoryView<float[]> scalar = MemoryViewFactory.of(DataType.F32,
    // MemoryFactory.ofFloats(7f), Shape.scalar());
    //        MemoryView<float[]> b = scalar.broadcast(Shape.of(2, 3, 4));
    //        assertTrue(b.isBroadcasted(), "Scalar -> (2,3,4) must broadcast with zero strides");
    //    }
    //
    //    @Test
    //    void testBroadcastZeroDim() {
    //        MemoryView<float[]> empty = MemoryViewFactory.of(DataType.F32,
    // MemoryFactory.ofFloats(new float[0]), Shape.of(0));
    //        MemoryView<float[]> result = empty.broadcast(Shape.of(3, 0));
    //        assertEquals(Shape.of(3, 0), result.shape());
    //        assertEquals(0, result.shape().size());
    //    }
    //
    //    @Test
    //    void testBroadcastScalarToZero() {
    //        MemoryView<float[]> scalar = MemoryViewFactory.of(DataType.F32,
    // MemoryFactory.ofFloats(1f), Shape.scalar());
    //        MemoryView<float[]> result = scalar.broadcast(Shape.of(0));
    //        assertEquals(Shape.of(0), result.shape());
    //        assertEquals(0, result.shape().size());
    //    }
    //
    //    @Disabled
    //    @Test
    //    void testBroadcastToZeroDim() {
    //        MemoryView<float[]> empty = MemoryViewFactory.of(DataType.F32,
    // MemoryFactory.ofFloats(new float[0]), Shape.of(0));
    //        MemoryView<float[]> r = empty.broadcast(Shape.of(3, 0));
    //        assertEquals(Shape.of(3, 0), r.shape());
    //        assertEquals(0, r.shape().size());
    //        assertTrue(r.isContiguous());
    //        assertFalse(r.isBroadcasted()); // no axis came from 1 -> N
    //    }
    //
    //    @Disabled
    //    @Test
    //    void testBroadcastToZeroDim2() {
    //        MemoryView<float[]> empty = MemoryViewFactory.of(DataType.F32,
    // MemoryFactory.ofFloats(new float[0]), Shape.of(0));
    //        MemoryView<float[]> r = empty.broadcast(Shape.of(3, 0));
    //        assertEquals(Shape.of(3, 0), r.shape());
    //        assertEquals(0, r.shape().size());
    //        assertTrue(r.isContiguous());
    //        assertFalse(r.isBroadcasted()); // no axis came from 1 -> N
    //    }
    //
    //    @Test
    //    void testBroadcastScalarToZero2() {
    //        MemoryView<float[]> s = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(1f),
    // Shape.scalar());
    //        MemoryView<float[]> r = s.broadcast(Shape.of(0));
    //        assertEquals(Shape.of(0), r.shape());
    //        assertTrue(r.isContiguous());
    //        assertTrue(r.isBroadcasted()); // came from 1 -> 0 along that axis
    //    }
    //
    //    @Test
    //    void testBroadcastShrinkNonSingletonToZero() {
    //        MemoryView<float[]> v = MemoryViewFactory.of(DataType.F32, MemoryFactory.ofFloats(new
    // float[6]), Shape.of(2, 3));
    //        MemoryView<float[]> r = v.broadcast(Shape.of(2, 0));
    //        assertEquals(Shape.of(2, 0), r.shape());
    //        assertEquals(0, r.shape().size());
    //        assertTrue(r.isContiguous()); // empty -> contiguous by convention
    //        assertFalse(r.isBroadcasted());
    //    }
}
