package com.qxotic.jinfer.x.models.qwen3;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

final class Qwen3StateCodecTest {

    @Test
    void restoresAChainByteExactly() {
        Qwen3.Configuration config = config();
        Qwen3StateCodec codec = new Qwen3StateCodec(config);
        assertEquals(0, codec.checkpointBytes(0), "dense attention carries no residue");
        assertEquals(16, codec.checkpointBytes(2));
        assertEquals(40, codec.checkpointBytes(5));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.checkpointBytes(2), 11);
            MemorySegment second = patterned(arena, codec.checkpointBytes(3), 71);
            MemorySegment actual = arena.allocate(codec.checkpointBytes(5), 64);
            MemorySegment expected = arena.allocate(codec.checkpointBytes(5), 64);
            Qwen3.State state = new Qwen3.State(config, 8, 4, arena);

            codec.restoreCheckpoint(state, 0, 2, first);
            codec.restoreCheckpoint(state, 2, 5, second);
            state.resumeAt(5);
            codec.saveCheckpoint(state, 0, 5, actual);

            // One layer: K rows, then V rows - the full span reassembled from the two chunks.
            MemorySegment.copy(first, 0, expected, 0, 8); // K[0,2)
            MemorySegment.copy(second, 0, expected, 8, 12); // K[2,5)
            MemorySegment.copy(first, 8, expected, 20, 8); // V[0,2)
            MemorySegment.copy(second, 12, expected, 28, 12); // V[2,5)
            assertEquals(-1, expected.mismatch(actual));
        }
    }

    @Test
    void rejectsInvalidSpansSizesAndSaveEndpoints() {
        Qwen3StateCodec codec = new Qwen3StateCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            Qwen3.State state = new Qwen3.State(config(), 8, 4, arena);
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
        MemorySegment blob = arena.allocate(bytes, 64);
        for (long i = 0; i < bytes; i++)
            blob.set(java.lang.foreign.ValueLayout.JAVA_BYTE, i, (byte) (seed + i));
        return blob;
    }

    private static Qwen3.Configuration config() {
        return new Qwen3.Configuration(
                4, // embeddingLength
                1, // numberOfLayers
                2, // numberOfHeads
                1, // numberOfKeyValueHeads
                8, // vocabularySize
                16, // contextLength
                8, // hiddenDim
                1e-5f, // rmsNormEps
                10_000f, // ropeTheta
                2, // headSize
                2, // ropeDim
                4, // queryDim
                2, // kvDim
                2); // kvMul
    }
}
