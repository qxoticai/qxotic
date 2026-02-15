package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.DomainFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ViewOpsTest {

    private static MemoryDomain<MemorySegment> domain;
    private static TensorOps ops;

    @BeforeAll
    static void setUpDomain() {
        domain = DomainFactory.ofMemorySegment();
        ops = new EagerTensorOps(domain);
    }

    @Test
    void viewBasicReshape() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(4, 3)));

        assertEquals(Shape.of(4, 3), result.shape());
        assertTrue(result.isMaterialized());
    }

    @Test
    void viewFlatten() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 24).view(Shape.of(2, 3, 4));
        Tensor input = Tensor.of(view);

        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(24)));

        assertEquals(Shape.of(24), result.shape());
    }

    @Test
    void viewAddDimensions() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(12));
        Tensor input = Tensor.of(view);

        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(2, 2, 3)));

        assertEquals(Shape.of(2, 2, 3), result.shape());
    }

    @Test
    void viewWithSingletonDimensions() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 6).view(Shape.of(2, 3));
        Tensor input = Tensor.of(view);

        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(1, 2, 3, 1)));

        assertEquals(Shape.of(1, 2, 3, 1), result.shape());
    }

    @Test
    void viewSharesMemory() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(4, 3)));

        MemoryView<?> inputView = input.materialize();
        MemoryView<?> resultView = result.materialize();
        assertSame(inputView.memory(), resultView.memory());
        assertEquals(inputView.byteOffset(), resultView.byteOffset());
    }

    @Test
    void viewThrowsOnSizeMismatch() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TensorOpsContext.with(ops, () -> input.view(Shape.of(5, 5))));

        assertTrue(ex.getMessage().contains("mismatch"));
    }

    /**
     * Tests that transposed tensors CAN be viewed as flat, but require a copy in eager mode.
     *
     * <p>A (3, 4) tensor transposed to (4, 3):(1, 4) has non-row-major strides. Flattening to (12,)
     * requires either lazy index computation (in TIR codegen) or a copy (in eager mode) to preserve
     * the correct logical element ordering.
     */
    @Test
    void viewTransposedRequiresCopyToFlatten() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        MemoryView<MemorySegment> transposed = view.transpose(0, 1);
        assertFalse(transposed.layout().isSuffixContiguous(0));
        Tensor input = Tensor.of(transposed);

        // Transposed tensor has non-row-major strides, so flatten requires copy in eager mode
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(12)));

        assertEquals(Shape.of(12), result.shape());
        // A copy was made - different memory backing
        assertNotSame(transposed.memory(), result.materialize().memory());
        // Result is now contiguous
        assertTrue(result.materialize().isContiguous());
    }

    /**
     * Tests that sliced tensors with stride can be viewed, but require a copy in eager mode.
     *
     * <p>A sliced tensor with step > 1 has non-contiguous strides and requires lazy indexing or
     * copy.
     */
    @Test
    void viewSlicedWithStrideRequiresCopy() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        MemoryView<MemorySegment> sliced = view.slice(1, 0, 4, 2);
        assertFalse(sliced.isContiguous());
        Tensor input = Tensor.of(sliced);

        // Sliced tensor with stride requires copy in eager mode
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(6)));

        assertEquals(Shape.of(6), result.shape());
        // A copy was made
        assertNotSame(sliced.memory(), result.materialize().memory());
        assertTrue(result.materialize().isContiguous());
    }

    @Test
    void viewAllowsCompatibleNonContiguousReshape() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        MemoryView<MemorySegment> slicedRows = view.slice(0, 0, 3, 2);
        assertFalse(slicedRows.isContiguous());
        Tensor input = Tensor.of(slicedRows);

        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(2, 4)));

        assertEquals(Shape.of(2, 4), result.shape());
    }

    @Test
    void viewScalarToVector() {
        MemoryView<MemorySegment> view = MemoryHelpers.arange(domain, DataType.FP32, 1);
        Tensor input = Tensor.of(view);

        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(1)));

        assertEquals(Shape.of(1), result.shape());
    }

    @Test
    void viewPreservesDataType() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP64, 6).view(Shape.of(2, 3));
        Tensor input = Tensor.of(view);

        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(3, 2)));

        assertEquals(DataType.FP64, result.dataType());
    }

    /**
     * Tests view with non-standard stride layout (2, 2, 2):(4, 1, 2).
     *
     * <p>The strides are not in row-major order (4, 1, 2 vs expected 4, 2, 1). Reshaping requires
     * lazy indexing (in codegen) or copy (in eager mode) to preserve correct element ordering.
     */
    @Test
    void viewNonStandardStrideRequiresCopy() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (2, 2, 2) with stride (4, 1, 2) - NOT row-major order
        Layout layout = Layout.of(Shape.flat(2, 2, 2), Stride.flat(4, 1, 2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        assertFalse(view.layout().isSuffixContiguous(0));
        Tensor input = Tensor.of(view);

        // Non-row-major strides require copy in eager mode
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(2, 4)));

        assertEquals(Shape.of(2, 4), result.shape());
        // A copy was made
        assertNotSame(view.memory(), result.materialize().memory());
        assertTrue(result.materialize().isContiguous());
    }

    /**
     * Tests that (2,2,2):(4,1,2) can be reshaped to ANY compatible shape.
     *
     * <p>Since the layout spans contiguous range [0-7], both (2,4) and (4,2) work with CuTe
     * semantics - the new shape gets row-major strides regardless of original stride ordering.
     */
    @Test
    void viewNonStandardStrideAllowsAnyCompatibleReshape() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (2, 2, 2) with stride (4, 1, 2) spans [0-7] contiguously
        // Condition: (2-1)*4 + (2-1)*1 + (2-1)*2 = 7 = 8-1 ✓
        Layout layout = Layout.of(Shape.flat(2, 2, 2), Stride.flat(4, 1, 2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        // Both reshapes work - new shapes get row-major strides
        Tensor result1 = TensorOpsContext.with(ops, () -> input.view(Shape.of(2, 4)));
        assertEquals(Shape.of(2, 4), result1.shape());

        Tensor result2 = TensorOpsContext.with(ops, () -> input.view(Shape.of(4, 2)));
        assertEquals(Shape.of(4, 2), result2.shape());

        Tensor result3 = TensorOpsContext.with(ops, () -> input.view(Shape.of(8)));
        assertEquals(Shape.of(8), result3.shape());
    }

    /**
     * Tests that contiguous non-standard strides CAN be reshaped.
     *
     * <p>Shape (2, 2, 2) with stride (4, 2, 1) is contiguous (standard row-major) and can be freely
     * reshaped.
     */
    @Test
    void viewContiguousNonTrivialShape() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (2, 2, 2) with standard row-major stride (4, 2, 1)
        Layout layout = Layout.of(Shape.flat(2, 2, 2), Stride.flat(4, 2, 1));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        assertTrue(view.isContiguous());
        Tensor input = Tensor.of(view);

        // Can reshape to any compatible shape
        Tensor result1 = TensorOpsContext.with(ops, () -> input.view(Shape.of(2, 4)));
        assertEquals(Shape.of(2, 4), result1.shape());

        Tensor result2 = TensorOpsContext.with(ops, () -> input.view(Shape.of(4, 2)));
        assertEquals(Shape.of(4, 2), result2.shape());

        Tensor result3 = TensorOpsContext.with(ops, () -> input.view(Shape.of(8)));
        assertEquals(Shape.of(8), result3.shape());
    }

    /**
     * Tests view with nested shape output.
     *
     * <p>A contiguous tensor can be viewed as a nested shape structure.
     */
    @Test
    void viewToNestedShape() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Standard contiguous (2, 2, 2) tensor
        Layout layout = Layout.of(Shape.flat(2, 2, 2), Stride.flat(4, 2, 1));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        // Can view as nested shape ((2, 2), 2)
        Shape nestedShape = Shape.of(Shape.of(2L, 2L), 2L);
        Tensor result = TensorOpsContext.with(ops, () -> input.view(nestedShape));

        assertEquals(nestedShape, result.shape());
        assertEquals(8, result.size());
        assertSame(view.memory(), result.materialize().memory());
    }

    /**
     * Tests CuTe-style stride inference for nested reshape.
     *
     * <p>(2,2,2):(4,1,2) viewed as ((2,2),2) should infer stride ((4,2),1) - hierarchical row-major
     * strides that preserve linear memory order.
     */
    @Test
    void viewNonStandardStrideToNestedShapeInfersCorrectStrides() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Original: (2, 2, 2):(4, 1, 2) - non-standard strides but spans [0-7]
        Layout original = Layout.of(Shape.flat(2, 2, 2), Stride.flat(4, 1, 2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, original);
        Tensor input = Tensor.of(view);

        // View as nested ((2, 2), 2) - should infer stride ((4, 2), 1)
        Shape nestedShape = Shape.of(Shape.of(2L, 2L), 2L);
        Tensor result = TensorOpsContext.with(ops, () -> input.view(nestedShape));

        assertEquals(nestedShape, result.shape());

        // Verify the inferred strides - strides are preserved from original layout
        // Original (2, 2, 2):(4, 1, 2) -> ((2, 2), 2):((4, 1), 2)
        Stride expectedStride = Stride.of(Stride.of(4L, 1L), 2L);
        assertEquals(expectedStride, result.stride());
    }

    // ========== Corner Cases: View Should Fail ==========

    /**
     * Tests that view with holes (skipping elements) requires a copy.
     *
     * <p>Shape (4,) with stride (2) accesses {0, 2, 4, 6} - skipping odd offsets. This requires
     * lazy indexing or copy to reshape correctly.
     */
    @Test
    void viewWithHolesRequiresCopy() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (4,) with stride (2) - accesses every other element: {0, 2, 4, 6}
        Layout layout = Layout.of(Shape.flat(4), Stride.flat(2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        // Holes in memory access pattern require copy in eager mode
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(2, 2)));

        assertEquals(Shape.of(2, 2), result.shape());
        // A copy was made
        assertNotSame(view.memory(), result.materialize().memory());
        assertTrue(result.materialize().isContiguous());
    }

    /**
     * Tests that view with interleaved access pattern requires a copy.
     *
     * <p>Shape (2, 2) with stride (4, 2) accesses {0, 2, 4, 6} - interleaved with holes. This
     * requires lazy indexing or copy to reshape correctly.
     */
    @Test
    void viewWithInterleavedAccessRequiresCopy() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (2, 2) with stride (4, 2) - interleaved access: {0, 2, 4, 6}
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(4, 2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        // Interleaved pattern requires copy in eager mode
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(4)));

        assertEquals(Shape.of(4), result.shape());
        assertNotSame(view.memory(), result.materialize().memory());
        assertTrue(result.materialize().isContiguous());
    }

    /**
     * Tests that view with gaps requires a copy.
     *
     * <p>Shape (2, 2) with stride (3, 1) accesses {0, 1, 3, 4} - missing offset 2. This requires
     * lazy indexing or copy to reshape correctly.
     */
    @Test
    void viewWithGapsRequiresCopy() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (2, 2) with stride (3, 1) - accesses {0, 1, 3, 4}, missing 2
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(3, 1));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        // Gap in memory access requires copy in eager mode
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(4)));

        assertEquals(Shape.of(4), result.shape());
        // A copy was made
        assertNotSame(view.memory(), result.materialize().memory());
        assertTrue(result.materialize().isContiguous());
    }

    /**
     * Tests that view with sparse strided access requires a copy.
     *
     * <p>Shape (2, 2) with stride (6, 3) accesses {0, 3, 6, 9} - very sparse. This requires lazy
     * indexing or copy to reshape correctly.
     */
    @Test
    void viewWithSparseAccessRequiresCopy() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 16);

        // Shape (2, 2) with stride (6, 3) - sparse access: {0, 3, 6, 9}
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(6, 3));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        // Sparse pattern requires copy in eager mode
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(4)));

        assertEquals(Shape.of(4), result.shape());
        assertNotSame(view.memory(), result.materialize().memory());
        assertTrue(result.materialize().isContiguous());
    }

    /**
     * Tests that view with strided slice requires a copy.
     *
     * <p>Taking every other row from a 4x3 matrix creates shape (2, 3) with stride (6, 1),
     * accessing rows 0 and 2. This requires lazy indexing or copy to flatten correctly.
     */
    @Test
    void viewWithStridedSliceRequiresCopy() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(4, 3));

        // Take every other row: rows 0 and 2
        MemoryView<MemorySegment> sliced = view.slice(0, 0, 4, 2); // shape (2, 3), stride (6, 1)
        assertFalse(sliced.isContiguous());
        Tensor input = Tensor.of(sliced);

        // Strided slice requires copy in eager mode
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(6)));

        assertEquals(Shape.of(6), result.shape());
        // A copy was made
        assertNotSame(sliced.memory(), result.materialize().memory());
        assertTrue(result.materialize().isContiguous());
    }

    /**
     * Tests that view with non-adjacent columns requires a copy.
     *
     * <p>Taking every other column from a 3x4 matrix creates shape (3, 2) with stride (4, 2). This
     * requires lazy indexing or copy to flatten correctly.
     */
    @Test
    void viewWithColumnSliceRequiresCopy() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));

        // Take every other column: columns 0 and 2
        MemoryView<MemorySegment> sliced = view.slice(1, 0, 4, 2); // shape (3, 2), stride (4, 2)
        assertFalse(sliced.isContiguous());
        Tensor input = Tensor.of(sliced);

        // Column slice requires copy in eager mode
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(6)));

        assertEquals(Shape.of(6), result.shape());
        assertNotSame(sliced.memory(), result.materialize().memory());
        assertTrue(result.materialize().isContiguous());
    }

    /**
     * Tests that view with broadcast (zero stride) dimensions requires a copy.
     *
     * <p>A broadcasted tensor with stride 0 on some dimension cannot be reshaped without copying,
     * as the same memory location is accessed multiple times and must be materialized.
     */
    @Test
    void viewWithBroadcastedDimensionRequiresCopy() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 4);

        // Shape (2, 4) with stride (0, 1) - first dim is broadcasted
        // Accesses: {0, 1, 2, 3, 0, 1, 2, 3} - duplicates!
        Layout layout = Layout.of(Shape.flat(2, 4), Stride.flat(0, 1));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        assertTrue(view.isBroadcasted());
        Tensor input = Tensor.of(view);

        // Broadcast requires copy to materialize duplicated elements
        Tensor result = TensorOpsContext.with(ops, () -> input.view(Shape.of(8)));

        assertEquals(Shape.of(8), result.shape());
        // A copy was made (new memory with 8 elements, not 4)
        assertNotSame(view.memory(), result.materialize().memory());
        assertTrue(result.materialize().isContiguous());
    }

    // ========== Reshape Tests (view with fallback to copy) ==========

    /** Tests that reshape returns a view when possible (contiguous tensor). */
    @Test
    void reshapeReturnsViewWhenPossible() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        Tensor result = TensorOpsContext.with(ops, () -> input.reshape(Shape.of(4, 3)));

        assertEquals(Shape.of(4, 3), result.shape());
        // Should share memory (view, not copy)
        assertSame(view.memory(), result.materialize().memory());
    }

    /** Tests that reshape copies data when view is not possible (strided slice with holes). */
    @Test
    void reshapeCopiesWhenViewNotPossible() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (4,) with stride (2) - has holes at odd offsets
        Layout layout = Layout.of(Shape.flat(4), Stride.flat(2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        // view would fail, but reshape should succeed by copying
        Tensor result = TensorOpsContext.with(ops, () -> input.reshape(Shape.of(2, 2)));

        assertEquals(Shape.of(2, 2), result.shape());
        // Should be contiguous after copy
        assertTrue(result.materialize().isContiguous());
        // Should NOT share memory (copy was made)
        assertNotSame(view.memory(), result.materialize().memory());
    }

    /** Tests that reshape copies data for transposed tensor. */
    @Test
    void reshapeCopiesTransposedTensor() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        MemoryView<MemorySegment> transposed = view.transpose(0, 1); // (4, 3):(1, 4)
        Tensor input = Tensor.of(transposed);

        // Flatten - requires copy because transpose has non-row-major strides
        Tensor flatView = TensorOpsContext.with(ops, () -> input.view(Shape.of(12)));
        // A copy was made
        assertNotSame(transposed.memory(), flatView.materialize().memory());
        assertTrue(flatView.materialize().isContiguous());

        // reshape should also work (uses view internally)
        Tensor flatReshape = TensorOpsContext.with(ops, () -> input.reshape(Shape.of(12)));
        assertEquals(Shape.of(12), flatReshape.shape());
        assertTrue(flatReshape.materialize().isContiguous());
    }

    /** Tests that reshape copies data for sliced tensor with gaps. */
    @Test
    void reshapeCopiesSlicedTensorWithGaps() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(4, 3));

        // Take every other row: rows 0 and 2 - has gap at row 1
        MemoryView<MemorySegment> sliced = view.slice(0, 0, 4, 2); // (2, 3):(6, 1)
        Tensor input = Tensor.of(sliced);

        // view would fail (gap in memory), but reshape should succeed
        Tensor result = TensorOpsContext.with(ops, () -> input.reshape(Shape.of(6)));

        assertEquals(Shape.of(6), result.shape());
        assertTrue(result.materialize().isContiguous());
        // New memory was allocated
        assertNotSame(sliced.memory(), result.materialize().memory());
    }

    /** Tests that reshape throws on size mismatch. */
    @Test
    void reshapeThrowsOnSizeMismatch() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        assertThrows(
                IllegalArgumentException.class,
                () -> TensorOpsContext.with(ops, () -> input.reshape(Shape.of(5, 5))));
    }

    // ========== Incompatible Shape Size Tests ==========

    @Test
    void viewThrowsWhenNewShapeTooLarge() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        // 12 elements -> 20 elements (too large)
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TensorOpsContext.with(ops, () -> input.view(Shape.of(4, 5))));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void viewThrowsWhenNewShapeTooSmall() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        // 12 elements -> 6 elements (too small)
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TensorOpsContext.with(ops, () -> input.view(Shape.of(2, 3))));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void viewThrowsWhenFlatteningToWrongSize() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        // 12 elements -> 10 elements
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TensorOpsContext.with(ops, () -> input.view(Shape.of(10))));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void reshapeThrowsWhenNewShapeTooLarge() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        // 12 elements -> 24 elements (too large)
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TensorOpsContext.with(ops, () -> input.reshape(Shape.of(4, 6))));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void reshapeThrowsWhenNewShapeTooSmall() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        // 12 elements -> 4 elements (too small)
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TensorOpsContext.with(ops, () -> input.reshape(Shape.of(2, 2))));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void reshapeThrowsWhenFlatteningToWrongSize() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        // 12 elements -> 15 elements
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TensorOpsContext.with(ops, () -> input.reshape(Shape.of(15))));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void viewThrowsWithNestedShapeSizeMismatch() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 8).view(Shape.of(2, 4));
        Tensor input = Tensor.of(view);

        // 8 elements -> nested shape with 12 elements
        Shape nestedShape = Shape.of(Shape.of(2L, 3L), 2L); // (2*3) * 2 = 12
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TensorOpsContext.with(ops, () -> input.view(nestedShape)));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void reshapeThrowsWithNestedShapeSizeMismatch() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 8).view(Shape.of(2, 4));
        Tensor input = Tensor.of(view);

        // 8 elements -> nested shape with 6 elements
        Shape nestedShape = Shape.of(Shape.of(3L, 1L), 2L); // (3*1) * 2 = 6
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TensorOpsContext.with(ops, () -> input.reshape(nestedShape)));
        assertTrue(ex.getMessage().contains("mismatch"));
    }
}
