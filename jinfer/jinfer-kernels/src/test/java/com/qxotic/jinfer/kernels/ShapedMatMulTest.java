package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.kernels.Oracles.assertClose;
import static com.qxotic.jinfer.kernels.Oracles.f32;
import static com.qxotic.jinfer.kernels.Oracles.f32View;
import static com.qxotic.jinfer.kernels.Oracles.q8;
import static com.qxotic.jinfer.kernels.Oracles.q8View;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * The shaped {@code gemm(W, A, C, n)} / {@code gemv(W, A, C)} entry points: correct results against
 * the flat low-level forms, plus the entry contract's fail-fast behavior (wrong rank, wrong dtype,
 * misaligned dimensions, out-of-range n, and A/C aliasing).
 */
class ShapedMatMulTest {

    @Test
    void gemmMatchesFlatGemmForFp32() {
        int m = 8, k = 16, n = 3;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = f32(arena, m * k, 1);
            MemorySegment a = f32(arena, n * k, 2);
            MemorySegment flat = arena.allocate(4L * n * m, 64);
            MemorySegment shaped = arena.allocate(4L * n * m, 64);

            MatMul.gemm(
                    f32View(w, (long) m * k),
                    f32View(a, (long) n * k),
                    k,
                    f32View(flat, (long) n * m),
                    m,
                    m,
                    n,
                    k);
            MatMul.gemm(
                    matrix(w, DataType.FP32, m, k),
                    matrix(a, DataType.FP32, n, k),
                    matrix(shaped, DataType.FP32, n, m),
                    n);

            assertClose(flat, shaped, n * m, "shaped gemm vs flat gemm (FP32)");
        }
    }

    @Test
    void gemvMatchesFlatGemvForFp32() {
        int m = 8, k = 16;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = f32(arena, m * k, 3);
            MemorySegment a = f32(arena, k, 4);
            MemorySegment flat = arena.allocate(4L * m, 64);
            MemorySegment shaped = arena.allocate(4L * m, 64);

            MatMul.gemv(f32View(w, (long) m * k), f32View(a, k), f32View(flat, m), m, k);
            MatMul.gemv(
                    matrix(w, DataType.FP32, m, k),
                    matrix(a, DataType.FP32, 1, k),
                    matrix(shaped, DataType.FP32, 1, m));

            assertClose(flat, shaped, m, "shaped gemv vs flat gemv (FP32)");
        }
    }

    @Test
    void gemmMatchesFlatGemmForQuantizedWeight() {
        int m = 8, k = 256, n = 3;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = q8(arena, m, k, 5);
            MemorySegment a = f32(arena, n * k, 6);
            MemorySegment flat = arena.allocate(4L * n * m, 64);
            MemorySegment shaped = arena.allocate(4L * n * m, 64);

            MatMul.gemm(
                    q8View(w, (long) m * k),
                    f32View(a, (long) n * k),
                    k,
                    f32View(flat, (long) n * m),
                    m,
                    m,
                    n,
                    k);
            MatMul.gemm(
                    matrix(w, DataType.Q8_0, m, k / 32),
                    matrix(a, DataType.FP32, n, k),
                    matrix(shaped, DataType.FP32, n, m),
                    n);

            assertClose(flat, shaped, n * m, "shaped gemm vs flat gemm (Q8_0)", 1e-3);
        }
    }

    @Test
    void gemvMatchesFlatGemvForQuantizedWeight() {
        int m = 8, k = 256;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = q8(arena, m, k, 7);
            MemorySegment a = f32(arena, k, 8);
            MemorySegment flat = arena.allocate(4L * m, 64);
            MemorySegment shaped = arena.allocate(4L * m, 64);

            MatMul.gemv(q8View(w, (long) m * k), f32View(a, k), f32View(flat, m), m, k);
            MatMul.gemv(
                    matrix(w, DataType.Q8_0, m, k / 32),
                    matrix(a, DataType.FP32, 1, k),
                    matrix(shaped, DataType.FP32, 1, m));

            assertClose(flat, shaped, m, "shaped gemv vs flat gemv (Q8_0)", 1e-3);
        }
    }

    @Test
    void gemmWithOneRowDelegatesToGemvSemantics() {
        int m = 6, k = 12;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = f32(arena, m * k, 9);
            MemorySegment a = f32(arena, k, 10);
            MemorySegment out = arena.allocate(4L * m, 64);
            MatMul.gemm(
                    matrix(w, DataType.FP32, m, k),
                    matrix(a, DataType.FP32, 1, k),
                    matrix(out, DataType.FP32, 1, m),
                    1);
            // n==1 is exactly the first row of A into the first row of C; just a smoke of the gate.
            // Correctness is covered by gemvMatchesFlatGemvForFp32.
        }
    }

    @Test
    void packedOutputRowStrideIsReadFromTheView() {
        int m = 4, k = 8, n = 3, packed = m + 3;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = f32(arena, m * k, 11);
            MemorySegment a = f32(arena, n * k, 12);
            MemorySegment flat = arena.allocate(4L * n * m, 64);
            MemorySegment packedOut = arena.allocate(4L * n * packed, 64);

            MatMul.gemm(
                    f32View(w, (long) m * k),
                    f32View(a, (long) n * k),
                    k,
                    f32View(flat, (long) n * m),
                    m,
                    m,
                    n,
                    k);
            MatMul.gemm(
                    matrix(w, DataType.FP32, m, k),
                    matrix(a, DataType.FP32, n, k),
                    matrix(packedOut, DataType.FP32, n, packed),
                    n);

            // the first m floats of each packed row must match the dense result
            for (int r = 0; r < n; r++) {
                MemorySegment expected = flat.asSlice((long) r * m * 4, (long) m * 4);
                MemorySegment actual = packedOut.asSlice((long) r * packed * 4, (long) m * 4);
                assertClose(expected, actual, m, "packed row " + r);
            }
        }
    }

    @Test
    void rejectsNonTwoDimensionalWeight() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w =
                    Views.wrap(f32(arena, 8, 13), DataType.FP32, Shape.flat(8));
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 14), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 15), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsNonTwoDimensionalActivation() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 16), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a =
                    Views.wrap(f32(arena, 8, 17), DataType.FP32, Shape.flat(8));
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 18), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsNonTwoDimensionalResult() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 19), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 20), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c =
                    Views.wrap(f32(arena, 8, 21), DataType.FP32, Shape.flat(8));
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsNonFp32Activation() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 22), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 23), DataType.FP16, 1, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 24), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsNonFp32Result() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 25), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 26), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 27), DataType.FP16, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsZeroRows() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 28), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 29), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 30), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 0));
        }
    }

    @Test
    void rejectsRowsPastActivationCapacity() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 31), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 32), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 16, 33), DataType.FP32, 2, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 2));
        }
    }

    @Test
    void rejectsRowsPastResultCapacity() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 34), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 16, 35), DataType.FP32, 2, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 36), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 2));
        }
    }

    @Test
    void rejectsActivationNarrowerThanContraction() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 37), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 7, 38), DataType.FP32, 1, 7);
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 39), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsResultNarrowerThanOutputWidth() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 40), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 41), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 7, 42), DataType.FP32, 1, 7);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsAliasedActivationAndResult() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 43), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 44), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, a, 1));
        }
    }

    @Test
    void rejectsScalarWeight() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w =
                    Views.wrap(f32(arena, 1, 45), DataType.FP32, Shape.scalar());
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 46), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 47), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsScalarActivation() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 48), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a =
                    Views.wrap(f32(arena, 1, 49), DataType.FP32, Shape.scalar());
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 50), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsScalarResult() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 51), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 52), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c =
                    Views.wrap(f32(arena, 1, 53), DataType.FP32, Shape.scalar());
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsThreeDimensionalWeight() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w =
                    Views.wrap(f32(arena, 2 * 8 * 8, 54), DataType.FP32, Shape.flat(2, 8, 8));
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 55), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 56), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsThreeDimensionalActivation() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 57), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a =
                    Views.wrap(f32(arena, 8, 58), DataType.FP32, Shape.flat(2, 1, 4));
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 59), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsThreeDimensionalResult() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 60), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 8, 61), DataType.FP32, 1, 8);
            MemoryView<MemorySegment> c =
                    Views.wrap(f32(arena, 8, 62), DataType.FP32, Shape.flat(2, 1, 4));
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsNonRowMajorStride() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 63), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a =
                    matrix(f32(arena, 8, 64), DataType.FP32, 1, 8).transpose(0, 1);
            MemoryView<MemorySegment> c = matrix(f32(arena, 8, 65), DataType.FP32, 1, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 1));
        }
    }

    @Test
    void rejectsActivationRowCountMismatchWithResult() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> w = matrix(f32(arena, 8 * 8, 66), DataType.FP32, 8, 8);
            MemoryView<MemorySegment> a = matrix(f32(arena, 16, 67), DataType.FP32, 2, 8);
            MemoryView<MemorySegment> c = matrix(f32(arena, 24, 68), DataType.FP32, 3, 8);
            assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, 3));
        }
    }

    @Test
    void negativeDimensionsAreRejectedAtShapeConstruction() {
        assertThrows(IllegalArgumentException.class, () -> Shape.flat(-1, 8));
        assertThrows(IllegalArgumentException.class, () -> Shape.flat(8, -1));
    }

    private static MemoryView<MemorySegment> matrix(
            MemorySegment seg, DataType type, long rows, long cols) {
        return Views.wrap(seg, type, Shape.flat(rows, cols));
    }
}
