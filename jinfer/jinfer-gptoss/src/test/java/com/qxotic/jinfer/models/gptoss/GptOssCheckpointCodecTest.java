package com.qxotic.jinfer.models.gptoss;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.PanamaMemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

final class GptOssCheckpointCodecTest {

    @Test
    void restoresAMixedRingAndDenseChainByteExactly() {
        GptOss.Configuration config = config();
        GptOssCheckpointCodec codec = new GptOssCheckpointCodec(config);
        assertEquals(0, codec.byteSize(0), "rows alone resume, no residue");
        assertEquals(32, codec.byteSize(2));
        assertEquals(64, codec.byteSize(4));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.byteSize(2), 11);
            MemorySegment second = patterned(arena, codec.byteSize(2), 71);
            MemorySegment actual = arena.allocate(codec.byteSize(4), 64);
            MemorySegment expected = arena.allocate(codec.byteSize(4), 64);
            GptOss.State state =
                    new GptOss.State(config, 8, 4, new PanamaMemoryArena(arena), false);

            // spans stay within W=4, so no ring slot aliases: every row lands on its own slot
            codec.restore(state, 0, 2, first);
            codec.restore(state, 2, 4, second);
            state.resumeAt(4);
            codec.capture(state, 0, 4, actual);

            // layer 0 is SWA (ring), layer 1 full attention; each chunk is L0 K,V then L1 K,V
            MemorySegment.copy(first, 0, expected, 0, 8); // L0 K[0,2)
            MemorySegment.copy(second, 0, expected, 8, 8); // L0 K[2,4)
            MemorySegment.copy(first, 8, expected, 16, 8); // L0 V[0,2)
            MemorySegment.copy(second, 8, expected, 24, 8); // L0 V[2,4)
            MemorySegment.copy(first, 16, expected, 32, 8); // L1 K[0,2)
            MemorySegment.copy(second, 16, expected, 40, 8); // L1 K[2,4)
            MemorySegment.copy(first, 24, expected, 48, 8); // L1 V[0,2)
            MemorySegment.copy(second, 24, expected, 56, 8); // L1 V[2,4)
            assertEquals(-1, expected.mismatch(actual));
        }
    }

    @Test
    void rejectsInvalidSpansSizesAndSaveEndpoints() {
        GptOssCheckpointCodec codec = new GptOssCheckpointCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            GptOss.State state =
                    new GptOss.State(config(), 8, 4, new PanamaMemoryArena(arena), false);
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

    private static GptOss.Configuration config() {
        return new GptOss.Configuration(
                4, // embeddingLength
                2, // numberOfLayers
                2, // numberOfHeads
                1, // numberOfKeyValueHeads
                2, // headSize
                8, // vocabularySize
                16, // contextLength
                1e-5f, // rmsNormEps
                10_000.0, // ropeTheta
                1.0f, // ropeScalingFactor
                16, // ropeOrigCtx
                4, // slidingWindow (power of 2, ring width for SWA layers)
                new boolean[] {true, false}, // swaMask: layer 0 rings, layer 1 is full attention
                2, // expertCount
                1, // expertUsedCount
                8, // expertFeedForwardLength
                1.0f); // expertWeightsScale
    }
}
