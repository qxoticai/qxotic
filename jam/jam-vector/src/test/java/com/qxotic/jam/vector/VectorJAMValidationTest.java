package com.qxotic.jam.vector;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jam.JAM;
import com.qxotic.jam.internal.GGMLType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class VectorJAMValidationTest {

    private static final int M = 4, N = 2, K = 256;
    private static final long A_BYTES = N * K * 4L, R_BYTES = N * M * 4L;
    private static final int[] WEIGHT_TYPES = {
        JAM.Q8_0, JAM.Q4_0, JAM.Q4_K, JAM.Q5_K, JAM.Q6_K, JAM.MXFP4, JAM.NVFP4, JAM.Q1_0
    };
    private static final JAM VECTOR = new VectorJAM(JAM.Parallel.INLINE);

    @Test
    void rejectsEveryHeapOperandBeforeReadingAbsoluteAddresses() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = weight(arena, JAM.Q8_0);
            MemorySegment a = arena.allocate(A_BYTES, 64);
            MemorySegment r = arena.allocate(R_BYTES, 64);

            assertThrows(
                    IllegalArgumentException.class,
                    () -> mm(MemorySegment.ofArray(new byte[(int) weightBytes(JAM.Q8_0)]), a, r));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> mm(w, MemorySegment.ofArray(new float[N * K]), r));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> mm(w, a, MemorySegment.ofArray(new float[N * M])));
        }
    }

    @Test
    void rejectsEveryClosedOperand() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = weight(arena, JAM.Q8_0);
            MemorySegment a = arena.allocate(A_BYTES, 64);
            MemorySegment r = arena.allocate(R_BYTES, 64);

            assertThrows(IllegalStateException.class, () -> mm(closed(weightBytes(JAM.Q8_0)), a, r));
            assertThrows(IllegalStateException.class, () -> mm(w, closed(A_BYTES), r));
            assertThrows(IllegalStateException.class, () -> mm(w, a, closed(R_BYTES)));
        }
    }

    @Test
    void boundsCheckEverySupportedWeightDtype() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocate(A_BYTES, 64);
            MemorySegment r = arena.allocate(R_BYTES, 64);
            for (int type : WEIGHT_TYPES) {
                long bytes = weightBytes(type);
                MemorySegment w = arena.allocate(bytes, 64);

                assertEquals(JAM.OK, mm(w, 0, type, a, 0, K, r, 0, M));
                assertThrows(
                        IndexOutOfBoundsException.class,
                        () -> mm(w.asSlice(0, bytes - 1), 0, type, a, 0, K, r, 0, M),
                        GGMLType.byCode(type).name());
            }
        }
    }

    @Test
    void boundsCheckEveryOffset() {
        try (Arena arena = Arena.ofConfined()) {
            long prefix = 7;
            MemorySegment w = arena.allocate(prefix + weightBytes(JAM.Q8_0), 64);
            MemorySegment a = arena.allocate(prefix + A_BYTES, 64);
            MemorySegment r = arena.allocate(prefix + R_BYTES, 64);

            assertEquals(JAM.OK, mm(w, prefix, JAM.Q8_0, a, prefix, K, r, prefix, M));
            assertThrows(
                    IndexOutOfBoundsException.class,
                    () -> mm(w, prefix + 1, JAM.Q8_0, a, prefix, K, r, prefix, M));
            assertThrows(
                    IndexOutOfBoundsException.class,
                    () -> mm(w, prefix, JAM.Q8_0, a, prefix + 1, K, r, prefix, M));
            assertThrows(
                    IndexOutOfBoundsException.class,
                    () -> mm(w, prefix, JAM.Q8_0, a, prefix, K, r, prefix + 1, M));
            assertThrows(
                    IndexOutOfBoundsException.class,
                    () -> mm(w, -1, JAM.Q8_0, a, prefix, K, r, prefix, M));
            assertThrows(
                    IndexOutOfBoundsException.class,
                    () -> mm(w, prefix, JAM.Q8_0, a, -1, K, r, prefix, M));
            assertThrows(
                    IndexOutOfBoundsException.class,
                    () -> mm(w, prefix, JAM.Q8_0, a, prefix, K, r, -1, M));
        }
    }

    @Test
    void boundsCheckActivationAndResultStrides() {
        try (Arena arena = Arena.ofConfined()) {
            int lda = K + 3, ldr = M + 3;
            long aBytes = ((N - 1L) * lda + K) * 4;
            long rBytes = ((N - 1L) * ldr + M) * 4;
            MemorySegment w = weight(arena, JAM.Q8_0);
            MemorySegment a = arena.allocate(aBytes, 64);
            MemorySegment r = arena.allocate(rBytes, 64);

            assertEquals(JAM.OK, mm(w, 0, JAM.Q8_0, a, 0, lda, r, 0, ldr));
            assertThrows(
                    IndexOutOfBoundsException.class,
                    () -> mm(w, 0, JAM.Q8_0, a.asSlice(0, aBytes - 1), 0, lda, r, 0, ldr));
            assertThrows(
                    IndexOutOfBoundsException.class,
                    () -> mm(w, 0, JAM.Q8_0, a, 0, lda, r.asSlice(0, rBytes - 1), 0, ldr));
        }
    }

    @Test
    void invalidShapesReturnStatusBeforeAddressArithmetic() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = weight(arena, JAM.Q8_0);
            MemorySegment a = arena.allocate(A_BYTES, 64);
            MemorySegment r = arena.allocate(R_BYTES, 64);

            assertEquals(
                    JAM.EINVAL,
                    VECTOR.mm(w, 0, JAM.Q8_0, K, a, 0, JAM.F32, K, r, 0, JAM.F32, M, 0, N, K));
            assertEquals(
                    JAM.EINVAL,
                    VECTOR.mm(w, 0, JAM.Q8_0, 0, a, 0, JAM.F32, 0, r, 0, JAM.F32, M, M, N, 0));
            assertEquals(
                    JAM.EINVAL,
                    VECTOR.mm(
                            w, 0, JAM.Q8_0, K, a, 0, JAM.F32, K - 1, r, 0, JAM.F32, M, M, N, K));
            assertEquals(
                    JAM.EINVAL,
                    VECTOR.mm(
                            w, 0, JAM.Q8_0, K, a, 0, JAM.F32, K, r, 0, JAM.F32, M - 1, M, N, K));
        }
    }

    private static int mm(MemorySegment w, MemorySegment a, MemorySegment r) {
        return mm(w, 0, JAM.Q8_0, a, 0, K, r, 0, M);
    }

    private static int mm(
            MemorySegment w,
            long wOffset,
            int weightType,
            MemorySegment a,
            long aOffset,
            int lda,
            MemorySegment r,
            long rOffset,
            int ldr) {
        return VECTOR.mm(
                w,
                wOffset,
                weightType,
                K,
                a,
                aOffset,
                JAM.F32,
                lda,
                r,
                rOffset,
                JAM.F32,
                ldr,
                M,
                N,
                K);
    }

    private static MemorySegment weight(Arena arena, int type) {
        return arena.allocate(weightBytes(type), 64);
    }

    private static long weightBytes(int type) {
        return M * GGMLType.byCode(type).rowBytes(K);
    }

    private static MemorySegment closed(long bytes) {
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(bytes, 64);
        arena.close();
        return segment;
    }
}
