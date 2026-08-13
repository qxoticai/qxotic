package com.qxotic.jinfer.x.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

final class Gemma4StateCodecTest {

    @Test
    void restoresAMixedRingAndDenseChainByteExactly() {
        Gemma4.Configuration config = config();
        Gemma4StateCodec codec = new Gemma4StateCodec(config);
        assertEquals(0, codec.checkpointBytes(0), "rows alone resume, no residue");
        assertEquals(48, codec.checkpointBytes(2));
        assertEquals(96, codec.checkpointBytes(4));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.checkpointBytes(2), 11);
            MemorySegment second = patterned(arena, codec.checkpointBytes(2), 71);
            MemorySegment actual = arena.allocate(codec.checkpointBytes(4), 64);
            MemorySegment expected = arena.allocate(codec.checkpointBytes(4), 64);
            Gemma4.State state = new Gemma4.State(config, 8, 4, arena);

            // spans stay within W=4, so no ring slot aliases: every row lands on its own slot
            codec.restoreCheckpoint(state, 0, 2, first);
            codec.restoreCheckpoint(state, 2, 4, second);
            state.resumeAt(4);
            codec.saveCheckpoint(state, 0, 4, actual);

            // layer 0 is SWA (ring, kvDim 2 -> 4B rows), layer 1 full (kvDim 4 -> 8B rows);
            // each chunk is L0 K,V then L1 K,V; the shared tail layer owns no KV
            MemorySegment.copy(first, 0, expected, 0, 8); // L0 K[0,2)
            MemorySegment.copy(second, 0, expected, 8, 8); // L0 K[2,4)
            MemorySegment.copy(first, 8, expected, 16, 8); // L0 V[0,2)
            MemorySegment.copy(second, 8, expected, 24, 8); // L0 V[2,4)
            MemorySegment.copy(first, 16, expected, 32, 16); // L1 K[0,2)
            MemorySegment.copy(second, 16, expected, 48, 16); // L1 K[2,4)
            MemorySegment.copy(first, 32, expected, 64, 16); // L1 V[0,2)
            MemorySegment.copy(second, 32, expected, 80, 16); // L1 V[2,4)
            assertEquals(-1, expected.mismatch(actual));
        }
    }

    @Test
    void rejectsInvalidSpansSizesAndSaveEndpoints() {
        Gemma4StateCodec codec = new Gemma4StateCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            Gemma4.State state = new Gemma4.State(config(), 8, 4, arena);
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
        for (long i = 0; i < bytes; i++) blob.set(ValueLayout.JAVA_BYTE, i, (byte) (seed + i));
        return blob;
    }

    private static Gemma4.Configuration config() {
        return new Gemma4.Configuration(
                4, // embeddingLength
                new int[] {8, 8, 8}, // feedForwardLength
                3, // numberOfLayers (2 own-KV + 1 shared tail)
                2, // numberOfHeads
                new int[] {1, 2, 2}, // numberOfKeyValueHeadsPerLayer
                8, // vocabularySize
                16, // contextLength
                1e-5f, // rmsNormEps
                10_000f, // ropeThetaFull
                10_000f, // ropeThetaSwa
                2, // headSizeFull
                2, // headSizeSwa
                4, // slidingWindow (power of 2)
                0f, // logitSoftcapping
                new boolean[] {true, false, true}, // isSwa
                2, // ownKvLayers
                0, // embeddingLengthPerLayer
                0, // expertCount (dense)
                0, // expertUsedCount
                0); // expertFeedForwardLength
    }
}
