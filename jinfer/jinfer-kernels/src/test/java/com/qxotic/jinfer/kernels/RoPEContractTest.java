package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/** RoPE values depend on absolute positions, never on how ingestion was chunked. */
class RoPEContractTest {

    private static final int HEAD_SIZE = 128;
    private static final int HALF = HEAD_SIZE / 2;
    private static final double THETA = 1_000_000.0;

    private record Range(MemoryView<MemorySegment> cos, MemoryView<MemorySegment> sin) {}

    @Test
    void rangesAndChunksMatchTheSamePositionsFilledFromZero() {
        Range longRange = fill(0, 5008, RoPE.plain(HEAD_SIZE, THETA));
        assertSameRows(longRange, 5000, fill(5000, 8, RoPE.plain(HEAD_SIZE, THETA)), 8);
        assertSameRows(longRange, 4999, fill(4999, 1, RoPE.plain(HEAD_SIZE, THETA)), 1);

        Range whole = fill(0, 512, RoPE.plain(HEAD_SIZE, THETA));
        for (int chunk : new int[] {1, 7, 64, 511}) {
            for (int from = 0; from + chunk <= 512; from += chunk) {
                assertSameRows(whole, from, fill(from, chunk, RoPE.plain(HEAD_SIZE, THETA)), chunk);
            }
        }
    }

    @Test
    void scaledSchedulesAreTranslationInvariant() {
        float[] factors = new float[HALF];
        for (int i = 0; i < HALF; i++) factors[i] = 1f + i / 8f;
        assertTranslationInvariant(RoPE.withFreqFactors(HEAD_SIZE, THETA, factors));
        assertTranslationInvariant(RoPE.yarn(HEAD_SIZE, THETA, 4f, 4096, 32f, 1f, 1f, 1f));
    }

    @Test
    void scatteredPositionsMatchTheirIndividualRanges() {
        int[] positions = {0, 1, 2, 0, 1, 511, 4999};
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            Range scattered =
                    new Range(
                            Views.allocateF32(memory, positions.length * HALF),
                            Views.allocateF32(memory, positions.length * HALF));
            RoPE.fill(
                    scattered.cos(),
                    scattered.sin(),
                    positions,
                    positions.length,
                    HALF,
                    RoPE.plain(HEAD_SIZE, THETA));
            for (int row = 0; row < positions.length; row++) {
                assertSameRows(
                        scattered, row, fill(positions[row], 1, RoPE.plain(HEAD_SIZE, THETA)), 1);
            }
        }
    }

    @Test
    void scheduleAndRotationLayoutAreIndependentAndPartialRotationKeepsTheTail() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            MemoryView<MemorySegment> cos = Views.allocateF32(memory, HALF);
            MemoryView<MemorySegment> sin = Views.allocateF32(memory, HALF);
            RoPE.fill(cos, sin, 7, 1, HALF, RoPE.plain(HEAD_SIZE, THETA));
            MemoryView<MemorySegment> interleaved = sequence(memory, HEAD_SIZE);
            MemoryView<MemorySegment> neox = sequence(memory, HEAD_SIZE);
            RoPE.applyInterleaved(interleaved, 0, 0, cos, sin, HALF);
            RoPE.applyNeox(neox, 0, 0, cos, sin, HALF);
            float c = get(cos, 0), s = get(sin, 0);
            assertEquals(1 * c - 2 * s, get(interleaved, 0), 0f);
            assertEquals(1 * c - (1 + HALF) * s, get(neox, 0), 0f);

            int lanes = 16;
            MemoryView<MemorySegment> partialCos = Views.allocateF32(memory, lanes);
            MemoryView<MemorySegment> partialSin = Views.allocateF32(memory, lanes);
            RoPE.fill(partialCos, partialSin, 3, 1, lanes, RoPE.plain(2 * lanes, THETA));
            MemoryView<MemorySegment> head = sequence(memory, HEAD_SIZE);
            RoPE.applyInterleaved(head, 0, 0, partialCos, partialSin, lanes);
            for (int i = 2 * lanes; i < HEAD_SIZE; i++) {
                assertEquals(i + 1, get(head, i), 0f, "tail dim " + i);
            }
        }
    }

    private static void assertTranslationInvariant(RoPE.Schedule schedule) {
        Range whole = fill(0, 300, schedule);
        assertSameRows(whole, 296, fill(296, 4, schedule), 4);
    }

    private static Range fill(int from, int count, RoPE.Schedule schedule) {
        MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(Arena.ofAuto());
        Range range =
                new Range(
                        Views.allocateF32(memory, count * HALF),
                        Views.allocateF32(memory, count * HALF));
        RoPE.fill(range.cos(), range.sin(), from, count, HALF, schedule);
        return range;
    }

    private static MemoryView<MemorySegment> sequence(MemoryArena<MemorySegment> memory, int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) values[i] = i + 1;
        return Views.fromFloatArray(memory, values);
    }

    private static void assertSameRows(Range whole, int wholeFrom, Range part, int count) {
        for (int row = 0; row < count; row++) {
            for (int lane = 0; lane < HALF; lane++) {
                long expected = (long) (wholeFrom + row) * HALF + lane;
                long actual = (long) row * HALF + lane;
                assertEquals(get(whole.cos(), expected), get(part.cos(), actual), 0f);
                assertEquals(get(whole.sin(), expected), get(part.sin(), actual), 0f);
            }
        }
    }

    private static float get(MemoryView<MemorySegment> view, long index) {
        return Views.getFloat(view, index, "RoPE test view");
    }
}
