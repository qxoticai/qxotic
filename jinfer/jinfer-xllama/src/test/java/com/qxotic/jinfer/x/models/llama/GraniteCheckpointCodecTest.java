package com.qxotic.jinfer.x.models.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

final class GraniteCheckpointCodecTest {

    @Test
    void restoresAChainByteExactly() {
        GraniteCheckpointCodec codec = new GraniteCheckpointCodec(config());
        assertEquals(0, codec.byteSize(0), "rows alone resume, no residue");
        assertEquals(32, codec.byteSize(2));
        assertEquals(64, codec.byteSize(4));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.byteSize(2), 11);
            MemorySegment second = patterned(arena, codec.byteSize(2), 71);
            MemorySegment actual = arena.allocate(codec.byteSize(4), 64);
            MemorySegment expected = arena.allocate(codec.byteSize(4), 64);
            Granite.State state =
                    new Granite.State(config(), 8, 4, new PanamaMemoryArena(arena), false);

            codec.restore(state, 0, 2, first);
            codec.restore(state, 2, 4, second);
            state.resumeAt(4);
            codec.capture(state, 0, 4, actual);

            // each checkpoint chunk is L0 K,V then L1 K,V (8B each at kvDim 2); the full-span
            // save writes L0 K[0,4), L0 V[0,4), L1 K[0,4), L1 V[0,4)
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
        GraniteCheckpointCodec codec = new GraniteCheckpointCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            Granite.State state =
                    new Granite.State(config(), 8, 4, new PanamaMemoryArena(arena), false);
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

    private static Granite.Configuration config() {
        return new Granite.Configuration(
                4, // embeddingLength
                2, // numberOfLayers
                2, // numberOfHeads
                1, // numberOfKeyValueHeads
                2, // headSize
                8, // vocabularySize
                16, // contextLength
                1e-5f, // rmsNormEps
                10_000f, // ropeTheta
                2, // ropeDimensionCount
                8, // hiddenDim
                1f, // embeddingScale
                1f, // residualScale
                1f, // logitScale
                0f); // attentionScaleValue
    }
}
