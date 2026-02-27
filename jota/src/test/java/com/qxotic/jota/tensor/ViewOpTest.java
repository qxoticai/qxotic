package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ViewOpTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = DomainFactory.ofMemorySegment();
    }

    @Test
    void viewBasicReshape() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(4, 3));

        assertEquals(Shape.of(4, 3), result.shape());
    }

    @Test
    void viewConvenienceBasic() {
        Tensor input = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4));

        Tensor result = input.view(4, 3);

        assertEquals(Shape.of(4, 3), result.shape());
    }

    @Test
    void viewConvenienceInfersSingleMinusOne() {
        Tensor input = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4));

        Tensor result = input.view(-1, 3);

        assertEquals(Shape.of(4, 3), result.shape());
    }

    @Test
    void viewConvenienceInfersLastMinusOne() {
        Tensor input = Tensor.iota(24, DataType.FP32).view(Shape.of(2, 3, 4));

        Tensor result = input.view(2, 3, -1);

        assertEquals(Shape.of(2, 3, 4), result.shape());
    }

    @Test
    void viewConvenienceNoArgsToScalar() {
        Tensor input = Tensor.iota(1, DataType.FP32).view(Shape.of(1, 1, 1));

        Tensor result = input.view();

        assertTrue(result.shape().isScalar());
        assertEquals(1, result.size());
    }

    @Test
    void viewConvenienceNoArgsRejectsNonSingleton() {
        Tensor input = Tensor.iota(2, DataType.FP32).view(Shape.of(1, 2));

        assertThrows(IllegalArgumentException.class, input::view);
    }

    @Test
    void viewConvenienceRejectsMultipleMinusOne() {
        Tensor input = Tensor.iota(12, DataType.FP32);

        assertThrows(IllegalArgumentException.class, () -> input.view(-1, -1));
    }

    @Test
    void viewConvenienceRejectsInvalidNegativeDimension() {
        Tensor input = Tensor.iota(12, DataType.FP32);

        assertThrows(IllegalArgumentException.class, () -> input.view(-2, 3));
    }

    @Test
    void viewConvenienceRejectsZeroDimensions() {
        Tensor input = Tensor.iota(12, DataType.FP32);

        assertThrows(IllegalArgumentException.class, () -> input.view(0, 12));
        assertThrows(IllegalArgumentException.class, () -> input.view(12, 0));
        assertThrows(IllegalArgumentException.class, () -> input.view(-1, 0, 3));
    }

    @Test
    void viewConvenienceRejectsMinusOneOnZeroSizedTensor() {
        Tensor input = Tensor.zeros(Shape.of(0, 2));

        assertThrows(IllegalArgumentException.class, () -> input.view(-1, 2));
    }

    @Test
    void viewConvenienceAllowsScalarToVectorInference() {
        Tensor scalar = Tensor.scalar(1.0f);

        Tensor result = scalar.view(-1);

        assertEquals(Shape.of(1), result.shape());
    }

    @Test
    void viewConvenienceRejectsNonDivisibleInference() {
        Tensor input = Tensor.iota(10, DataType.FP32);

        assertThrows(IllegalArgumentException.class, () -> input.view(3, -1));
    }

    @Test
    void viewConvenienceRejectsSizeMismatchWithoutInference() {
        Tensor input = Tensor.iota(10, DataType.FP32);

        assertThrows(IllegalArgumentException.class, () -> input.view(3, 4));
    }

    @Test
    void viewFlatten() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 24).view(Shape.of(2, 3, 4));
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(24));

        assertEquals(Shape.of(24), result.shape());
    }

    @Test
    void viewAddDimensions() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(12));
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(2, 2, 3));

        assertEquals(Shape.of(2, 2, 3), result.shape());
    }

    @Test
    void viewWithSingletonDimensions() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 6).view(Shape.of(2, 3));
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(1, 2, 3, 1));

        assertEquals(Shape.of(1, 2, 3, 1), result.shape());
    }

    @Test
    void viewSharesMemory() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(4, 3));

        MemoryView<?> resultView = result.materialize();
        assertEquals(Shape.of(4, 3), resultView.shape());
    }

    @Test
    void viewThrowsOnSizeMismatch() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.view(Shape.of(5, 5)));

        assertTrue(ex.getMessage().contains("mismatch"));
    }

    /**
     * Tests that transposed tensors CAN be viewed as flat. Through lazy IR codegen, the index
     * computation handles non-row-major strides correctly.
     */
    @Test
    void viewTransposedCanFlatten() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        MemoryView<MemorySegment> transposed = view.transpose(0, 1);
        assertFalse(transposed.layout().isSuffixContiguous(0));
        Tensor input = Tensor.of(transposed);

        Tensor result = input.view(Shape.of(12));

        assertEquals(Shape.of(12), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    /**
     * Tests that sliced tensors with stride can be viewed. Through lazy IR codegen, the index
     * computation handles non-contiguous strides correctly.
     */
    @Test
    void viewSlicedWithStride() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        MemoryView<MemorySegment> sliced = view.slice(1, 0, 4, 2);
        assertFalse(sliced.isContiguous());
        Tensor input = Tensor.of(sliced);

        Tensor result = input.view(Shape.of(6));

        assertEquals(Shape.of(6), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    @Test
    void viewAllowsCompatibleNonContiguousReshape() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        MemoryView<MemorySegment> slicedRows = view.slice(0, 0, 3, 2);
        assertFalse(slicedRows.isContiguous());
        Tensor input = Tensor.of(slicedRows);

        Tensor result = input.view(Shape.of(2, 4));

        assertEquals(Shape.of(2, 4), result.shape());
    }

    @Test
    void viewScalarToVector() {
        MemoryView<MemorySegment> view = MemoryHelpers.arange(domain, DataType.FP32, 1);
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(1));

        assertEquals(Shape.of(1), result.shape());
    }

    @Test
    void viewPreservesDataType() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP64, 6).view(Shape.of(2, 3));
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(3, 2));

        assertEquals(DataType.FP64, result.dataType());
    }

    /**
     * Tests view with non-standard stride layout (2, 2, 2):(4, 1, 2). Through lazy IR codegen, the
     * index computation handles non-standard strides correctly.
     */
    @Test
    void viewNonStandardStride() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (2, 2, 2) with stride (4, 1, 2) - NOT row-major order
        Layout layout = Layout.of(Shape.flat(2, 2, 2), Stride.flat(4, 1, 2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        assertFalse(view.layout().isSuffixContiguous(0));
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(2, 4));

        assertEquals(Shape.of(2, 4), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    /**
     * Tests that (2,2,2):(4,1,2) can be reshaped to ANY compatible shape. Through lazy IR codegen,
     * any compatible shape works via index computation.
     */
    @Test
    void viewNonStandardStrideAllowsAnyCompatibleReshape() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (2, 2, 2) with stride (4, 1, 2) spans [0-7] contiguously
        Layout layout = Layout.of(Shape.flat(2, 2, 2), Stride.flat(4, 1, 2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        Tensor result1 = input.view(Shape.of(2, 4));
        assertEquals(Shape.of(2, 4), result1.shape());

        Tensor result2 = input.view(Shape.of(4, 2));
        assertEquals(Shape.of(4, 2), result2.shape());

        Tensor result3 = input.view(Shape.of(8));
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

        Tensor result1 = input.view(Shape.of(2, 4));
        assertEquals(Shape.of(2, 4), result1.shape());

        Tensor result2 = input.view(Shape.of(4, 2));
        assertEquals(Shape.of(4, 2), result2.shape());

        Tensor result3 = input.view(Shape.of(8));
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
        Tensor result = input.view(nestedShape);

        assertEquals(nestedShape, result.shape());
        assertEquals(8, result.size());
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
        Tensor result = input.view(nestedShape);

        assertEquals(nestedShape, result.shape());

        // Verify the inferred strides
        Stride expectedStride = Stride.of(Stride.of(4L, 1L), 2L);
        assertEquals(expectedStride, result.stride());
    }

    // ========== Corner Cases: View with non-contiguous inputs ==========

    /**
     * Tests that view with holes (skipping elements) works through lazy IR codegen.
     *
     * <p>Shape (4,) with stride (2) accesses {0, 2, 4, 6} - skipping odd offsets.
     */
    @Test
    void viewWithHoles() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (4,) with stride (2) - accesses every other element: {0, 2, 4, 6}
        Layout layout = Layout.of(Shape.flat(4), Stride.flat(2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(2, 2));

        assertEquals(Shape.of(2, 2), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    /**
     * Tests that view with interleaved access pattern works through lazy IR codegen.
     *
     * <p>Shape (2, 2) with stride (4, 2) accesses {0, 2, 4, 6} - interleaved with holes.
     */
    @Test
    void viewWithInterleavedAccess() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (2, 2) with stride (4, 2) - interleaved access: {0, 2, 4, 6}
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(4, 2));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(4));

        assertEquals(Shape.of(4), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    /**
     * Tests that view with gaps works through lazy IR codegen.
     *
     * <p>Shape (2, 2) with stride (3, 1) accesses {0, 1, 3, 4} - missing offset 2.
     */
    @Test
    void viewWithGaps() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 8);

        // Shape (2, 2) with stride (3, 1) - accesses {0, 1, 3, 4}, missing 2
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(3, 1));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(4));

        assertEquals(Shape.of(4), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    /**
     * Tests that view with sparse strided access works through lazy IR codegen.
     *
     * <p>Shape (2, 2) with stride (6, 3) accesses {0, 3, 6, 9} - very sparse.
     */
    @Test
    void viewWithSparseAccess() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 16);

        // Shape (2, 2) with stride (6, 3) - sparse access: {0, 3, 6, 9}
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(6, 3));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(4));

        assertEquals(Shape.of(4), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    /**
     * Tests that view with strided slice works through lazy IR codegen.
     *
     * <p>Taking every other row from a 4x3 matrix creates shape (2, 3) with stride (6, 1),
     * accessing rows 0 and 2.
     */
    @Test
    void viewWithStridedSlice() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(4, 3));

        // Take every other row: rows 0 and 2
        MemoryView<MemorySegment> sliced = view.slice(0, 0, 4, 2); // shape (2, 3), stride (6, 1)
        assertFalse(sliced.isContiguous());
        Tensor input = Tensor.of(sliced);

        Tensor result = input.view(Shape.of(6));

        assertEquals(Shape.of(6), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    /**
     * Tests that view with non-adjacent columns works through lazy IR codegen.
     *
     * <p>Taking every other column from a 3x4 matrix creates shape (3, 2) with stride (4, 2).
     */
    @Test
    void viewWithColumnSlice() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));

        // Take every other column: columns 0 and 2
        MemoryView<MemorySegment> sliced = view.slice(1, 0, 4, 2); // shape (3, 2), stride (4, 2)
        assertFalse(sliced.isContiguous());
        Tensor input = Tensor.of(sliced);

        Tensor result = input.view(Shape.of(6));

        assertEquals(Shape.of(6), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    /**
     * Tests that view with broadcast (zero stride) dimensions works through lazy IR codegen.
     *
     * <p>A broadcasted tensor with stride 0 on some dimension cannot be reshaped without copying,
     * as the same memory location is accessed multiple times and must be materialized.
     */
    @Test
    void viewWithBroadcastedDimension() {
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(DataType.FP32, 4);

        // Shape (2, 4) with stride (0, 1) - first dim is broadcasted
        Layout layout = Layout.of(Shape.flat(2, 4), Stride.flat(0, 1));
        MemoryView<MemorySegment> view = MemoryView.of(memory, DataType.FP32, layout);
        assertTrue(view.isBroadcasted());
        Tensor input = Tensor.of(view);

        Tensor result = input.view(Shape.of(8));

        assertEquals(Shape.of(8), result.shape());
        MemoryView<?> materialized = result.materialize();
        assertTrue(materialized.isContiguous());
    }

    // ========== Incompatible Shape Size Tests ==========

    @Test
    void viewThrowsWhenNewShapeTooLarge() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.view(Shape.of(4, 5)));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void viewThrowsWhenNewShapeTooSmall() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.view(Shape.of(2, 3)));
        assertTrue(ex.getMessage().contains("mismatch"));
    }

    @Test
    void viewThrowsWhenFlatteningToWrongSize() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 12).view(Shape.of(3, 4));
        Tensor input = Tensor.of(view);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.view(Shape.of(10)));
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
                assertThrows(IllegalArgumentException.class, () -> input.view(nestedShape));
        assertTrue(ex.getMessage().contains("mismatch"));
    }
}
