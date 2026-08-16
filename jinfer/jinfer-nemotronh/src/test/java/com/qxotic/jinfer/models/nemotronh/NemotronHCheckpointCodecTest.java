package com.qxotic.jinfer.models.nemotronh;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.PanamaMemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

final class NemotronHCheckpointCodecTest {

    @Test
    void restoresAChainWithTheLastBlocksRecurrentEndpoint() {
        NemotronH.Configuration config = config();
        NemotronHCheckpointCodec codec = new NemotronHCheckpointCodec(config);
        assertTrue(codec.byteSize(0) > 0, "recurrent state is fixed checkpoint overhead");
        assertEquals(96, codec.byteSize(0), "one SSM layer: SSM state + conv ring");
        assertEquals(112, codec.byteSize(2));
        assertEquals(136, codec.byteSize(5));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.byteSize(2), 11);
            MemorySegment second = patterned(arena, codec.byteSize(3), 71);
            MemorySegment actual = arena.allocate(codec.byteSize(5), 64);
            MemorySegment expected = arena.allocate(codec.byteSize(5), 64);
            NemotronH.State state =
                    new NemotronH.State(config, 8, 4, new PanamaMemoryArena(arena), false);

            codec.restore(state, 0, 2, first);
            codec.restore(state, 2, 5, second);
            state.resumeAt(5);
            codec.capture(state, 0, 5, actual);

            // One attention layer: K rows, then V rows; then the SSM layer's endpoint residue
            // (SSM state, conv ring) - restored from the LAST chunk, like the rows.
            MemorySegment.copy(first, 0, expected, 0, 8); // K[0,2)
            MemorySegment.copy(second, 0, expected, 8, 12); // K[2,5)
            MemorySegment.copy(first, 8, expected, 20, 8); // V[0,2)
            MemorySegment.copy(second, 12, expected, 28, 12); // V[2,5)
            MemorySegment.copy(second, 24, expected, 40, 32); // SSM state at the endpoint
            MemorySegment.copy(second, 56, expected, 72, 64); // conv ring at the endpoint
            assertEquals(-1, expected.mismatch(actual));

            state.reset();
            MemorySegment reset = arena.allocate(codec.byteSize(0), 64);
            codec.capture(state, 0, 0, reset);
            assertEquals(
                    -1,
                    MemorySegment.ofArray(new byte[96]).mismatch(reset),
                    "reset zeroes exactly the recurrent residue");
        }
    }

    @Test
    void rejectsInvalidSpansSizesAndSaveEndpoints() {
        NemotronHCheckpointCodec codec = new NemotronHCheckpointCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            NemotronH.State state =
                    new NemotronH.State(config(), 8, 4, new PanamaMemoryArena(arena), false);
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

    private static NemotronH.Configuration config() {
        return new NemotronH.Configuration(
                4, // embeddingLength
                2, // numberOfLayers
                2, // numberOfHeads
                1, // numberOfKeyValueHeads
                2, // headSize
                8, // vocabularySize
                16, // contextLength
                1e-5f, // rmsNormEps
                new NemotronH.LayerType[] {NemotronH.LayerType.ATTENTION, NemotronH.LayerType.SSM},
                4, // ssmInnerSize
                1, // ssmGroupCount
                2, // ssmTimeStepRank
                2, // ssmStateSize
                3, // ssmConvKernel
                1, // expertCount
                1, // expertUsedCount
                8, // expertFeedForwardLength
                8, // expertSharedFeedForwardLength
                false, // expertWeightsNorm
                1.0f); // expertWeightsScale
    }
}
