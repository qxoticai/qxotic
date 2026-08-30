package com.qxotic.jinfer.models.bailingmoe3;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.Builder;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class BailingMoe3CheckpointCodecTest {
    @Test
    void parsesScalarAndPerLayerClampsAndRejectsUnknownRopeScaling() {
        var scalar = Builder.newBuilder().putFloat("clamp", 3f).build();
        var array = Builder.newBuilder().putArrayOfFloat("clamp", new float[] {1f, 2f, 3f}).build();
        org.junit.jupiter.api.Assertions.assertArrayEquals(
                new float[] {3f, 3f, 3f}, BailingMoe3.layerFloats(scalar, "clamp", 3));
        org.junit.jupiter.api.Assertions.assertArrayEquals(
                new float[] {1f, 2f, 3f}, BailingMoe3.layerFloats(array, "clamp", 3));

        var unsupported =
                Builder.newBuilder().putString("bailingmoe3.rope.scaling.type", "longrope").build();
        assertThrows(
                IllegalArgumentException.class,
                () -> BailingMoe3.buildRope(unsupported, "bailingmoe3", config(false), Map.of()));
    }

    @Test
    void ingestFailsFastWhenTheWeightArenaWasClosed() {
        MemoryView<MemorySegment> dead;
        try (Arena weightArena = Arena.ofConfined()) {
            dead = Views.allocateF32(MemoryAllocators.ofArena(weightArena), 1);
        }
        BailingMoe3.Configuration config = config(false);
        BailingMoe3.Weights weights =
                new BailingMoe3.Weights(
                        dead,
                        dead,
                        dead,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        RoPE.plain(2, 10_000f),
                        1);
        BailingMoe3 model = new BailingMoe3(config, null, weights);
        try (Arena stateArena = Arena.ofConfined()) {
            BailingMoe3.State state =
                    new BailingMoe3.State(
                            config, 8, 1, MemoryAllocators.ofArena(stateArena), false);
            assertThrows(IllegalStateException.class, () -> model.ingest(state, Batch.step(0)));
        }
    }

    @Test
    void roundTripsMlaRowsAndKdaEndpointState() {
        BailingMoe3.Configuration config = config(false);
        BailingMoe3CheckpointCodec codec = new BailingMoe3CheckpointCodec(config);
        assertEquals(128, codec.byteSize(0));
        assertEquals(144, codec.byteSize(2));

        try (Arena arena = Arena.ofConfined()) {
            BailingMoe3.State state =
                    new BailingMoe3.State(config, 8, 2, MemoryAllocators.ofArena(arena), false);
            MemorySegment expected = patterned(arena, codec.byteSize(2), 23);
            codec.restore(state, 0, 2, expected);
            state.resumeAt(2);
            MemorySegment actual = arena.allocate(codec.byteSize(2), 64);
            codec.capture(state, 0, 2, actual);
            assertEquals(-1, expected.mismatch(actual));

            state.reset();
            MemorySegment reset = arena.allocate(codec.byteSize(0), 64);
            codec.capture(state, 0, 0, reset);
            assertEquals(-1, MemorySegment.ofArray(new byte[128]).mismatch(reset));
        }
    }

    @Test
    void includesMtpCacheAndPendingHidden() {
        BailingMoe3.Configuration config = config(true);
        BailingMoe3CheckpointCodec codec = new BailingMoe3CheckpointCodec(config);
        assertEquals(144, codec.byteSize(0));
        assertEquals(176, codec.byteSize(2));

        try (Arena arena = Arena.ofConfined()) {
            BailingMoe3.State state =
                    new BailingMoe3.State(config, 8, 2, MemoryAllocators.ofArena(arena), false);
            MemorySegment expected = patterned(arena, codec.byteSize(2), 51);
            codec.restore(state, 0, 2, expected);
            state.resumeAt(2);
            MemorySegment actual = arena.allocate(codec.byteSize(2), 64);
            codec.capture(state, 0, 2, actual);
            assertEquals(-1, expected.mismatch(actual));
        }
    }

    @Test
    void rejectsInvalidSpanSizeAndEndpoint() {
        BailingMoe3CheckpointCodec codec = new BailingMoe3CheckpointCodec(config(false));
        try (Arena arena = Arena.ofConfined()) {
            BailingMoe3.State state =
                    new BailingMoe3.State(
                            config(false), 8, 2, MemoryAllocators.ofArena(arena), false);
            MemorySegment block = arena.allocate(codec.byteSize(2), 64);
            assertThrows(IllegalArgumentException.class, () -> codec.restore(state, -1, 1, block));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> codec.restore(state, 0, 2, block.asSlice(1)));
            assertThrows(IllegalStateException.class, () -> codec.capture(state, 0, 2, block));
        }
    }

    private static MemorySegment patterned(Arena arena, long bytes, int seed) {
        MemorySegment blob = arena.allocate(bytes, 64);
        for (long i = 0; i < bytes; i++) blob.set(ValueLayout.JAVA_BYTE, i, (byte) (seed + 17 * i));
        return blob;
    }

    private static BailingMoe3.Configuration config(boolean mtp) {
        return new BailingMoe3.Configuration(
                4,
                2,
                mtp ? 1 : 0,
                2,
                8,
                16,
                1e-5f,
                10_000f,
                2,
                mtp ? new boolean[] {true, false, true} : new boolean[] {true, false},
                2,
                3,
                true,
                -5f,
                2,
                2,
                2,
                2,
                8,
                1,
                4,
                1,
                2,
                1,
                2,
                2,
                true,
                1f,
                mtp ? new float[3] : new float[2],
                mtp ? new float[3] : new float[2]);
    }
}
