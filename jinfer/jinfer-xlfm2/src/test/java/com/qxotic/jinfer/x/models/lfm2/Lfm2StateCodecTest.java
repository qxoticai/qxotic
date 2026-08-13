package com.qxotic.jinfer.x.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

final class Lfm2StateCodecTest {

    @Test
    void restoresAChainWithTheLastBlocksRecurrentEndpoint() {
        Lfm2.Configuration config = config();
        Lfm2StateCodec codec = new Lfm2StateCodec(config);
        assertEquals(64, codec.checkpointBytes(0));
        assertEquals(104, codec.checkpointBytes(5));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.checkpointBytes(2), 11);
            MemorySegment second = patterned(arena, codec.checkpointBytes(3), 71);
            MemorySegment actual = arena.allocate(codec.checkpointBytes(5), 64);
            MemorySegment expected = arena.allocate(codec.checkpointBytes(5), 64);
            Lfm2.State state = new Lfm2.State(config, 8, 4, arena);

            codec.restoreCheckpoint(state, 0, 2, first);
            codec.restoreCheckpoint(state, 2, 5, second);
            state.resumeAt(5);
            codec.saveCheckpoint(state, 0, 5, actual);

            // One attention layer: K rows, then V rows, then the final recurrent residue.
            MemorySegment.copy(first, 0, expected, 0, 8);
            MemorySegment.copy(second, 0, expected, 8, 12);
            MemorySegment.copy(first, 8, expected, 20, 8);
            MemorySegment.copy(second, 12, expected, 28, 12);
            MemorySegment.copy(second, 24, expected, 40, 64);
            assertEquals(-1, expected.mismatch(actual));

            state.reset();
            MemorySegment reset = arena.allocate(codec.checkpointBytes(0), 64);
            codec.saveCheckpoint(state, 0, 0, reset);
            assertEquals(-1, MemorySegment.ofArray(new byte[64]).mismatch(reset));
        }
    }

    @Test
    void rejectsInvalidSpansSizesAndSaveEndpoints() {
        Lfm2StateCodec codec = new Lfm2StateCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            Lfm2.State state = new Lfm2.State(config(), 8, 4, arena);
            MemorySegment block = arena.allocate(codec.checkpointBytes(2), 64);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> codec.restoreCheckpoint(state, -1, 1, block));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> codec.restoreCheckpoint(state, 0, 9, block));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> codec.restoreCheckpoint(state, 0, 2, block.asSlice(1)));
            assertThrows(
                    IllegalStateException.class, () -> codec.saveCheckpoint(state, 0, 2, block));
        }
    }

    private static MemorySegment patterned(Arena arena, long bytes, int seed) {
        MemorySegment segment = arena.allocate(bytes, 64);
        for (long i = 0; i < bytes; i++) {
            segment.set(ValueLayout.JAVA_BYTE, i, (byte) (seed + i * 17));
        }
        return segment;
    }

    private static Lfm2.Configuration config() {
        return new Lfm2.Configuration(
                4,
                new int[] {8, 8, 8},
                3,
                2,
                new int[] {0, 1, 0},
                16,
                8,
                1e-5f,
                10_000f,
                2,
                0,
                3,
                0,
                0,
                0,
                3,
                1,
                true,
                0,
                0);
    }
}
