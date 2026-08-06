package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/**
 * Rotary values must not depend on where the fill started. A state fills scratch for the batch it
 * is about to ingest, so position 5000 is written at row 0 of one ingest and row 3 of another - and
 * both must hold what a table built from position 0 held at index 5000, BIT FOR BIT. Anything less
 * and generation depends on how a prompt happened to be chunked.
 */
final class RoPEFillTest {

    private static final int HEAD_SIZE = 128;
    private static final int HALF = HEAD_SIZE / 2;
    private static final double THETA = 1_000_000.0;

    private record Range(F32FloatTensor cos, F32FloatTensor sin) {}

    private static Range fill(int from, int count) {
        Arena arena = Arena.ofAuto();
        F32FloatTensor cos = F32FloatTensor.allocate(arena, count * HALF);
        F32FloatTensor sin = F32FloatTensor.allocate(arena, count * HALF);
        RoPE.fill(cos, sin, from, count, HEAD_SIZE, THETA);
        return new Range(cos, sin);
    }

    private static void assertSameRows(Range whole, int wholeFrom, Range part, int count) {
        for (int row = 0; row < count; row++) {
            for (int i = 0; i < HALF; i++) {
                long a = (long) (wholeFrom + row) * HALF + i, b = (long) row * HALF + i;
                assertEquals(
                        whole.cos().getFloat(a),
                        part.cos().getFloat(b),
                        0f,
                        "cos row " + row + " lane " + i);
                assertEquals(
                        whole.sin().getFloat(a),
                        part.sin().getFloat(b),
                        0f,
                        "sin row " + row + " lane " + i);
            }
        }
    }

    /** A batch starting mid-context holds what the same positions hold in a fill from zero. */
    @Test
    void aRangeMatchesTheSamePositionsFilledFromZero() {
        Range whole = fill(0, 5008);
        assertSameRows(whole, 5000, fill(5000, 8), 8);
        assertSameRows(whole, 0, fill(0, 8), 8);
        assertSameRows(whole, 4999, fill(4999, 1), 1); // a decode step: one row, deep in context
    }

    /** Chunking a prefill must not change a single float. */
    @Test
    void chunkingAPrefillChangesNothing() {
        Range whole = fill(0, 512);
        for (int chunk : new int[] {1, 7, 64, 511}) {
            for (int from = 0; from + chunk <= 512; from += chunk) {
                assertSameRows(whole, from, fill(from, chunk), chunk);
            }
        }
    }

    /**
     * The llama3 and YaRN schedules carry the same guarantee - they differ only in the frequency,
     * which is a function of the lane, never of where the fill began.
     */
    @Test
    void theScaledSchedulesAreTranslationInvariantToo() {
        Arena arena = Arena.ofAuto();
        float[] factors = new float[HALF];
        for (int i = 0; i < HALF; i++) factors[i] = 1f + i / 8f;

        F32FloatTensor wc = F32FloatTensor.allocate(arena, 300 * HALF);
        F32FloatTensor ws = F32FloatTensor.allocate(arena, 300 * HALF);
        RoPE.fillFromFreqs(wc, ws, 0, 300, HEAD_SIZE, THETA, factors);
        F32FloatTensor pc = F32FloatTensor.allocate(arena, 4 * HALF);
        F32FloatTensor ps = F32FloatTensor.allocate(arena, 4 * HALF);
        RoPE.fillFromFreqs(pc, ps, 296, 4, HEAD_SIZE, THETA, factors);
        assertSameRows(new Range(wc, ws), 296, new Range(pc, ps), 4);

        F32FloatTensor yc = F32FloatTensor.allocate(arena, 300 * HALF);
        F32FloatTensor ys = F32FloatTensor.allocate(arena, 300 * HALF);
        RoPE.fillYarn(yc, ys, 0, 300, HEAD_SIZE, THETA, 4f, 4096, 32f, 1f, 1f, 1f);
        F32FloatTensor ypc = F32FloatTensor.allocate(arena, 4 * HALF);
        F32FloatTensor yps = F32FloatTensor.allocate(arena, 4 * HALF);
        RoPE.fillYarn(ypc, yps, 296, 4, HEAD_SIZE, THETA, 4f, 4096, 32f, 1f, 1f, 1f);
        assertSameRows(new Range(yc, ys), 296, new Range(ypc, yps), 4);
    }

    /**
     * And the fill agrees with the front-loaded table it replaces. This pins the port migration:
     * every model's rotation must see the values it saw before.
     */
    @Test
    void theFillAgreesWithThePrecomputedTable() {
        RoPE.Freqs table = RoPE.precomputeFreqsCis(1024, HEAD_SIZE, THETA);
        Range range = fill(1000, 24);
        for (int row = 0; row < 24; row++) {
            for (int i = 0; i < HALF; i++) {
                int a = (1000 + row) * HALF + i;
                long b = (long) row * HALF + i;
                assertEquals(table.cos()[a], range.cos().getFloat(b), 0f);
                assertEquals(table.sin()[a], range.sin().getFloat(b), 0f);
            }
        }
    }
}
