package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.*;

public class MemoryViewTransposeTest extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testTranspose2D(MemoryContext<B> context) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryViewFactory.allocate(context.memoryAllocator(), DataType.F32, shape);

        // Initialize data if memory access is available
        if (context.memoryAccess() != null) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    float value = i * 3 + j + 1; // 1-6
                    context.memoryAccess().writeFloat(
                            view.memory(),
                            (i * 3 + j) * DataType.F32.byteSize(),
                            value
                    );
                }
            }
        }

        // Transpose the matrix (swap axes 0 and 1)
        MemoryView<B> transposed = view.transpose(0, 1);

        // Verify the shape is now 3x2
        assertEquals(Shape.of(3, 2), transposed.shape());

        // Verify the data if memory access is available
        if (context.memoryAccess() != null) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 2; j++) {
                    float expected = j * 3 + i + 1; // Transposed indices
                    float actual = readFloat(context.memoryAccess(), transposed, i, j);
                    assertEquals(expected, actual);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testTranspose3D(MemoryContext<B> context) {
        // Create a 2x3x4 tensor
        Shape shape = Shape.of(2, 3, 4);
        MemoryView<B> view = MemoryViewFactory.allocate(context.memoryAllocator(), DataType.F32, shape);

        // Initialize data if memory access is available
        if (context.memoryAccess() != null) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 4; k++) {
                        float value = i * 12 + j * 4 + k + 1; // 1-24
                        assertTrue(view.isContiguous());
                        context.memoryAccess().writeFloat(
                                view.memory(),
                                (i * 12 + j * 4 + k) * DataType.F32.byteSize(),
                                value
                        );
                    }
                }
            }
        }

        // Transpose axes 0 and 2
        MemoryView<B> transposed = view.transpose(0, 2);

        // Verify the shape is now 4x3x2
        assertEquals(Shape.of(4, 3, 2), transposed.shape());

        // Verify the data if memory access is available
        if (context.memoryAccess() != null) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 2; k++) {
                        float expected = k * 12 + j * 4 + i + 1; // Transposed indices
                        float actual = readFloat(context.memoryAccess(), transposed, i, j, k);
                        assertEquals(expected, actual);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testTransposeNegativeIndices(MemoryContext<B> context) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryViewFactory.allocate(context.memoryAllocator(), DataType.F32, shape);

        // Transpose using negative indices (-1 means last dimension)
        MemoryView<B> transposed = view.transpose(0, -1);

        // Should be equivalent to transpose(0, 1)
        assertEquals(Shape.of(3, 2), transposed.shape());
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testTransposeSameAxis(MemoryContext<B> context) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryViewFactory.allocate(context.memoryAllocator(), DataType.F32, shape);

        // Transposing the same axis should return an equivalent view
        MemoryView<B> transposed = view.transpose(0, 0);
        assertEquals(view.shape(), transposed.shape());

        // The underlying memory should be the same
        assertSame(view.memory(), transposed.memory());
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testTransposeInvalidAxis(MemoryContext<B> context) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryViewFactory.allocate(context.memoryAllocator(), DataType.F32, shape);

        // Should throw for invalid axis indices
        assertThrows(IllegalArgumentException.class, () -> view.transpose(0, 2));
        assertThrows(IllegalArgumentException.class, () -> view.transpose(-3, 0));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testTransposeStrides(MemoryContext<B> context) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryViewFactory.allocate(context.memoryAllocator(), DataType.F32, shape);

        // Get original strides
        Stride originalStrides = view.byteStride();

        // Transpose the matrix
        MemoryView<B> transposed = view.transpose(0, 1);

        // Verify strides are swapped
        Stride transposedStrides = transposed.byteStride();
        assertEquals(originalStrides.modeAt(0), transposedStrides.modeAt(1));
        assertEquals(originalStrides.modeAt(1), transposedStrides.modeAt(0));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testTransposeContiguity(MemoryContext<B> context) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryViewFactory.allocate(context.memoryAllocator(), DataType.F32, shape);

        // Original view should be contiguous
        assertTrue(view.isContiguous());

        // Transposed view may or may not be contiguous depending on implementation
        MemoryView<B> transposed = view.transpose(0, 1);

        // For row-major storage, transposing typically makes it non-contiguous
        // But this depends on the memory layout implementation
        // So we just verify the behavior is consistent
        if (view.isContiguous()) {
            assertFalse(transposed.isContiguous());
        }
    }
}