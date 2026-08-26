package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

final class Lfm2CheckpointCodecTest {

    @Test
    void restoresAChainWithTheLastBlocksRecurrentEndpoint() {
        Lfm2.Configuration config = config();
        Lfm2CheckpointCodec codec = new Lfm2CheckpointCodec(config);
        assertEquals(64, codec.byteSize(0));
        assertEquals(104, codec.byteSize(5));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.byteSize(2), 11);
            MemorySegment second = patterned(arena, codec.byteSize(3), 71);
            MemorySegment actual = arena.allocate(codec.byteSize(5), 64);
            MemorySegment expected = arena.allocate(codec.byteSize(5), 64);
            Lfm2.State state = new Lfm2.State(config, 8, 4, MemoryAllocators.ofArena(arena), false);

            codec.restore(state, 0, 2, first);
            codec.restore(state, 2, 5, second);
            state.resumeAt(5);
            codec.capture(state, 0, 5, actual);

            // One attention layer: K rows, then V rows, then the final recurrent residue.
            MemorySegment.copy(first, 0, expected, 0, 8);
            MemorySegment.copy(second, 0, expected, 8, 12);
            MemorySegment.copy(first, 8, expected, 20, 8);
            MemorySegment.copy(second, 12, expected, 28, 12);
            MemorySegment.copy(second, 24, expected, 40, 64);
            assertEquals(-1, expected.mismatch(actual));

            state.reset();
            MemorySegment reset = arena.allocate(codec.byteSize(0), 64);
            codec.capture(state, 0, 0, reset);
            assertEquals(-1, MemorySegment.ofArray(new byte[64]).mismatch(reset));
        }
    }

    @Test
    void rejectsInvalidSpansSizesAndSaveEndpoints() {
        Lfm2CheckpointCodec codec = new Lfm2CheckpointCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            Lfm2.State state =
                    new Lfm2.State(config(), 8, 4, MemoryAllocators.ofArena(arena), false);
            MemorySegment block = arena.allocate(codec.byteSize(2), 64);
            assertThrows(IllegalArgumentException.class, () -> codec.restore(state, -1, 1, block));
            assertThrows(IllegalArgumentException.class, () -> codec.restore(state, 0, 9, block));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> codec.restore(state, 0, 2, block.asSlice(1)));
            assertThrows(IllegalStateException.class, () -> codec.capture(state, 0, 2, block));
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
