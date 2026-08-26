package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

final class Qwen35CheckpointCodecTest {

    @Test
    void restoresAChainWithTheLastBlocksRecurrentEndpoint() {
        Qwen35.Configuration config = config();
        Qwen35CheckpointCodec codec = new Qwen35CheckpointCodec(config);
        assertTrue(codec.byteSize(0) > 0, "recurrent state is fixed checkpoint overhead");
        assertEquals(96, codec.byteSize(0), "one linear layer: S matrix + conv history");
        assertEquals(112, codec.byteSize(2));
        assertEquals(136, codec.byteSize(5));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = patterned(arena, codec.byteSize(2), 11);
            MemorySegment second = patterned(arena, codec.byteSize(3), 71);
            MemorySegment actual = arena.allocate(codec.byteSize(5), 64);
            MemorySegment expected = arena.allocate(codec.byteSize(5), 64);
            Qwen35.State state =
                    new Qwen35.State(config, 8, 4, MemoryAllocators.ofArena(arena), false);

            codec.restore(state, 0, 2, first);
            codec.restore(state, 2, 5, second);
            state.resumeAt(5);
            codec.capture(state, 0, 5, actual);

            // One full-attention layer: K rows, then V rows; then the linear layer's endpoint
            // residue (S matrix, conv history) - restored from the LAST chunk, like the rows.
            MemorySegment.copy(first, 0, expected, 0, 8); // K[0,2)
            MemorySegment.copy(second, 0, expected, 8, 12); // K[2,5)
            MemorySegment.copy(first, 8, expected, 20, 8); // V[0,2)
            MemorySegment.copy(second, 12, expected, 28, 12); // V[2,5)
            MemorySegment.copy(second, 24, expected, 40, 32); // S at the endpoint
            MemorySegment.copy(second, 56, expected, 72, 64); // conv at the endpoint
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
        Qwen35CheckpointCodec codec = new Qwen35CheckpointCodec(config());
        try (Arena arena = Arena.ofConfined()) {
            Qwen35.State state =
                    new Qwen35.State(config(), 8, 4, MemoryAllocators.ofArena(arena), false);
            MemorySegment block = arena.allocate(codec.byteSize(2), 64);
            assertThrows(IllegalArgumentException.class, () -> codec.restore(state, -1, 1, block));
            assertThrows(IllegalArgumentException.class, () -> codec.restore(state, 0, 9, block));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> codec.restore(state, 0, 2, block.asSlice(1)));
            assertThrows(IllegalStateException.class, () -> codec.capture(state, 0, 2, block));
        }
    }

    @Test
    void includesTheMtpKvRowsAndPendingHidden() {
        Qwen35.Configuration config = config(true);
        Qwen35CheckpointCodec codec = new Qwen35CheckpointCodec(config);
        assertEquals(112, codec.byteSize(0), "recurrent residue plus pending hidden");
        assertEquals(144, codec.byteSize(2), "two full-attention layers");

        try (Arena arena = Arena.ofConfined()) {
            Qwen35.State state =
                    new Qwen35.State(config, 8, 4, MemoryAllocators.ofArena(arena), false);
            MemorySegment expected = patterned(arena, codec.byteSize(2), 37);
            codec.restore(state, 0, 2, expected);
            state.resumeAt(2);
            MemorySegment actual = arena.allocate(codec.byteSize(2), 64);
            codec.capture(state, 0, 2, actual);
            assertEquals(
                    -1,
                    expected.mismatch(actual),
                    "target KV, MTP KV, recurrent state and MTP carry round-trip together");

            state.reset();
            MemorySegment reset = arena.allocate(codec.byteSize(0), 64);
            codec.capture(state, 0, 0, reset);
            assertEquals(
                    -1,
                    MemorySegment.ofArray(new byte[112]).mismatch(reset),
                    "reset clears recurrent state and the MTP carry");
        }
    }

    private static MemorySegment patterned(Arena arena, long bytes, int seed) {
        MemorySegment blob = arena.allocate(bytes, 64);
        for (long i = 0; i < bytes; i++) blob.set(ValueLayout.JAVA_BYTE, i, (byte) (seed + i));
        return blob;
    }

    private static Qwen35.Configuration config() {
        return config(false);
    }

    private static Qwen35.Configuration config(boolean mtp) {
        return new Qwen35.Configuration(
                4, // embeddingLength
                2, // numberOfLayers
                mtp ? 1 : 0, // nextnPredictLayers
                2, // numberOfHeads
                1, // numberOfKeyValueHeads
                2, // headSize
                8, // vocabularySize
                16, // contextLength
                1e-5f, // rmsNormEps
                10_000f, // ropeTheta
                2, // ropeDimensionCount
                8, // hiddenDim
                mtp
                        ? new boolean[] {true, false, true}
                        : new boolean[] {true, false}, // appended MTP is full attention
                4, // ssmInnerSize
                1, // ssmGroupCount
                2, // ssmTimeStepRank (heads)
                2, // ssmStateSize
                3, // ssmConvKernel
                0, // expertCount (dense)
                0, // expertUsedCount
                0, // expertFeedForwardLength
                0); // expertSharedFeedForwardLength
    }
}
