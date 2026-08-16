package com.qxotic.jinfer.x.models.maple;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

final class MapleCheckpointCodecTest {

    @Test
    void restoresMixedRingAndDenseCacheByteExactly() {
        Maple.Configuration config = config();
        MapleCheckpointCodec codec = new MapleCheckpointCodec(config);
        assertEquals(32, codec.byteSize(2));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.byteSize(2), 11);
            MemorySegment second = patterned(arena, codec.byteSize(2), 71);
            MemorySegment actual = arena.allocate(codec.byteSize(4), 64);
            MemorySegment expected = arena.allocate(codec.byteSize(4), 64);
            Maple.State state = new Maple.State(config, 8, 4, new PanamaMemoryArena(arena), false);

            codec.restore(state, 0, 2, first);
            codec.restore(state, 2, 4, second);
            state.resumeAt(4);
            codec.capture(state, 0, 4, actual);

            MemorySegment.copy(first, 0, expected, 0, 8);
            MemorySegment.copy(second, 0, expected, 8, 8);
            MemorySegment.copy(first, 8, expected, 16, 8);
            MemorySegment.copy(second, 8, expected, 24, 8);
            MemorySegment.copy(first, 16, expected, 32, 8);
            MemorySegment.copy(second, 16, expected, 40, 8);
            MemorySegment.copy(first, 24, expected, 48, 8);
            MemorySegment.copy(second, 24, expected, 56, 8);
            assertEquals(-1, expected.mismatch(actual));
        }
    }

    @Test
    void rejectsInvalidCheckpoint() {
        MapleCheckpointCodec codec = new MapleCheckpointCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            Maple.State state =
                    new Maple.State(config(), 8, 4, new PanamaMemoryArena(arena), false);
            MemorySegment block = arena.allocate(codec.byteSize(2), 64);
            assertThrows(IllegalArgumentException.class, () -> codec.restore(state, -1, 1, block));
            assertThrows(IllegalStateException.class, () -> codec.capture(state, 0, 2, block));
        }
    }

    private static MemorySegment patterned(Arena arena, long bytes, int seed) {
        MemorySegment blob = arena.allocate(bytes, 64);
        for (long i = 0; i < bytes; i++) blob.set(ValueLayout.JAVA_BYTE, i, (byte) (seed + i));
        return blob;
    }

    private static Maple.Configuration config() {
        return new Maple.Configuration(
                4,
                2,
                2,
                1,
                2,
                2,
                8,
                16,
                1e-6f,
                10_000,
                4,
                new boolean[] {true, false},
                2,
                1,
                8,
                1f,
                new float[] {7f, 7f});
    }
}
