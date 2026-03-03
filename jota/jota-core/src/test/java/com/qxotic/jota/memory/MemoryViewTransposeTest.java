package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

public class MemoryViewTransposeTest extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testTranspose2D(MemoryDomain<B> domain) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view =
                MemoryViewFactory.allocate(domain.memoryAllocator(), DataType.FP32, shape);

        // Initialize data if memory access is available
        if (domain.directAccess() != null) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    float value = i * 3 + j + 1; // 1-6
                    domain.directAccess()
                            .writeFloat(
                                    view.memory(), (i * 3 + j) * DataType.FP32.byteSize(), value);
                }
            }
        }

        // Transpose the matrix (swap axes 0 and 1)
        MemoryView<B> transposed = view.transpose(0, 1);

        // Verify the shape is now 3x2
        assertEquals(Shape.of(3, 2), transposed.shape());

        // Verify the data if memory access is available
        if (domain.directAccess() != null) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 2; j++) {
                    float expected = j * 3 + i + 1; // Transposed indices
                    float actual = readFloat(domain.directAccess(), transposed, i, j);
                    assertEquals(expected, actual);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testTranspose3D(MemoryDomain<B> domain) {
        // Create a 2x3x4 tensor
        Shape shape = Shape.of(2, 3, 4);
        MemoryView<B> view =
                MemoryViewFactory.allocate(domain.memoryAllocator(), DataType.FP32, shape);

        // Initialize data if memory access is available
        if (domain.directAccess() != null) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 4; k++) {
                        float value = i * 12 + j * 4 + k + 1; // 1-24
                        assertTrue(view.isRowMajorContiguous());
                        domain.directAccess()
                                .writeFloat(
                                        view.memory(),
                                        (i * 12 + j * 4 + k) * DataType.FP32.byteSize(),
                                        value);
                    }
                }
            }
        }

        // Transpose axes 0 and 2
        MemoryView<B> transposed = view.transpose(0, 2);

        // Verify the shape is now 4x3x2
        assertEquals(Shape.of(4, 3, 2), transposed.shape());

        // Verify the data if memory access is available
        if (domain.directAccess() != null) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 2; k++) {
                        float expected = k * 12 + j * 4 + i + 1; // Transposed indices
                        float actual = readFloat(domain.directAccess(), transposed, i, j, k);
                        assertEquals(expected, actual);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testTransposeNegativeIndices(MemoryDomain<B> domain) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view =
                MemoryViewFactory.allocate(domain.memoryAllocator(), DataType.FP32, shape);

        // Transpose using negative indices (-1 means last dimension)
        MemoryView<B> transposed = view.transpose(0, -1);

        // Should be equivalent to transpose(0, 1)
        assertEquals(Shape.of(3, 2), transposed.shape());
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testTransposeSameAxis(MemoryDomain<B> domain) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view =
                MemoryViewFactory.allocate(domain.memoryAllocator(), DataType.FP32, shape);

        // Transposing the same axis should return an equivalent view
        MemoryView<B> transposed = view.transpose(0, 0);
        assertEquals(view.shape(), transposed.shape());

        // The underlying memory should be the same
        assertSame(view.memory(), transposed.memory());
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testTransposeInvalidAxis(MemoryDomain<B> domain) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view =
                MemoryViewFactory.allocate(domain.memoryAllocator(), DataType.FP32, shape);

        // Should throw for invalid axis indices
        assertThrows(IllegalArgumentException.class, () -> view.transpose(0, 2));
        assertThrows(IllegalArgumentException.class, () -> view.transpose(-3, 0));
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testTransposeStrides(MemoryDomain<B> domain) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view =
                MemoryViewFactory.allocate(domain.memoryAllocator(), DataType.FP32, shape);

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
    @MethodSource("domainsSupportingF32")
    <B> void testTransposeContiguity(MemoryDomain<B> domain) {
        // Create a 2x3 matrix
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view =
                MemoryViewFactory.allocate(domain.memoryAllocator(), DataType.FP32, shape);

        // Original view should be row-major contiguous, span-contiguous, and non-overlapping
        assertTrue(view.isRowMajorContiguous());
        assertTrue(view.isSpanContiguous());
        assertTrue(view.isNonOverlapping());

        // Transposed view may or may not be contiguous depending on implementation
        MemoryView<B> transposed = view.transpose(0, 1);

        // Transpose keeps a gapless span but is no longer row-major contiguous
        assertFalse(transposed.isRowMajorContiguous());
        assertTrue(transposed.isSpanContiguous());
        assertTrue(transposed.isNonOverlapping());
    }
}
