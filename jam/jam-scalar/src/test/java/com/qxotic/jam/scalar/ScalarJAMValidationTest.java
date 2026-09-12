package com.qxotic.jam.scalar;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jam.JAM;
import com.qxotic.jam.internal.GGMLType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * The scalar backend reads through checked accessors, so a heap operand is fine and a bad operand
 * cannot corrupt memory - but it holds the same contract as the other backends: a stride shorter
 * than its row is EINVAL, and an undersized operand is refused whole, by name, before the kernel
 * has written half a result.
 */
class ScalarJAMValidationTest {

    private static final int M = 4, N = 2, K = 64;
    private static final long A_BYTES = N * K * 4L, R_BYTES = N * M * 4L;
    private static final JAM SCALAR = new ScalarJAM(JAM.Parallel.INLINE);

    @Test
    void heapOperandsAreAccepted() {
        MemorySegment w = MemorySegment.ofArray(new byte[(int) weightBytes()]);
        assertEquals(
                JAM.OK,
                mm(
                        w,
                        0,
                        MemorySegment.ofArray(new float[N * K]),
                        0,
                        K,
                        MemorySegment.ofArray(new float[N * M]),
                        0,
                        M));
    }

    @Test
    void stridesShorterThanARowAreInvalid() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = arena.allocate(weightBytes(), 64);
            MemorySegment a = arena.allocate(A_BYTES, 64);
            MemorySegment r = arena.allocate(R_BYTES, 64);
            assertEquals(JAM.OK, mm(w, 0, a, 0, K, r, 0, M));
            assertEquals(JAM.EINVAL, mm(w, 0, a, 0, K - 1, r, 0, M));
            assertEquals(JAM.EINVAL, mm(w, 0, a, 0, K, r, 0, M - 1));
        }
    }

    @Test
    void undersizedOperandsAreRefusedByNameBeforeTheKernelRuns() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = arena.allocate(weightBytes(), 64);
            MemorySegment a = arena.allocate(A_BYTES, 64);
            MemorySegment r = arena.allocate(R_BYTES, 64);
            refused("weight W", () -> mm(w.asSlice(0, weightBytes() - 1), 0, a, 0, K, r, 0, M));
            refused("activation A", () -> mm(w, 0, a.asSlice(0, A_BYTES - 1), 0, K, r, 0, M));
            refused("result R", () -> mm(w, 0, a, 0, K, r.asSlice(0, R_BYTES - 1), 0, M));
            refused("weight W", () -> mm(w, 1, a, 0, K, r, 0, M));
            // a longer stride needs the longer span
            refused("activation A", () -> mm(w, 0, a, 0, K + 1, r, 0, M));
        }
    }

    private static void refused(String operand, Runnable call) {
        IndexOutOfBoundsException e = assertThrows(IndexOutOfBoundsException.class, call::run);
        assertEquals(true, e.getMessage().contains(operand), e.getMessage());
    }

    private static int mm(
            MemorySegment w,
            long wOff,
            MemorySegment a,
            long aOff,
            int lda,
            MemorySegment r,
            long rOff,
            int ldr) {
        return SCALAR.mm(
                w, wOff, JAM.Q8_0, K, a, aOff, JAM.F32, lda, r, rOff, JAM.F32, ldr, M, N, K);
    }

    private static long weightBytes() {
        return M * GGMLType.byCode(JAM.Q8_0).rowBytes(K);
    }
}
