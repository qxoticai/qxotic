package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

final class LlamaCheckpointCodecTest {

    @Test
    void restoresAChainByteExactly() {
        LlamaCheckpointCodec codec = new LlamaCheckpointCodec(config());
        assertEquals(0, codec.byteSize(0));
        assertEquals(32, codec.byteSize(2));
        assertEquals(64, codec.byteSize(4));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.byteSize(2), 11);
            MemorySegment second = patterned(arena, codec.byteSize(2), 71);
            MemorySegment actual = arena.allocate(codec.byteSize(4), 64);
            MemorySegment expected = arena.allocate(codec.byteSize(4), 64);
            Llama.State state =
                    new Llama.State(config(), 8, 4, MemoryAllocators.ofArena(arena), false);

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
    void rejectsInvalidSpansSizesAndSaveEndpoints() {
        LlamaCheckpointCodec codec = new LlamaCheckpointCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            Llama.State state =
                    new Llama.State(config(), 8, 4, MemoryAllocators.ofArena(arena), false);
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
        MemorySegment blob = arena.allocate(bytes, 64);
        for (long i = 0; i < bytes; i++) blob.set(ValueLayout.JAVA_BYTE, i, (byte) (seed + i));
        return blob;
    }

    private static Llama.Configuration config() {
        return new Llama.Configuration(
                4, 2, 2, 1, 2, 8, 16, 1e-5f, 10_000f, 2, 8, 1f, 1f, 1f, 0f, 0, 0f, 0);
    }
}
