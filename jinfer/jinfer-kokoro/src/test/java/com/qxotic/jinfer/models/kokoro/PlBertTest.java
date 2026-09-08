package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

final class PlBertTest {

    @Test
    void sharedAlbertLayerUsesBidirectionalAttentionAndPostNormResiduals() {
        try (Arena arena = Arena.ofShared()) {
            MemoryAllocator<MemorySegment> allocator = MemoryAllocators.ofArena(arena);
            MemoryView<MemorySegment> zeroVector = floats(allocator, new float[2], 2);
            MemoryView<MemorySegment> oneVector = floats(allocator, new float[] {1, 1}, 2);
            KokoroLayers.Linear zero =
                    new KokoroLayers.Linear(
                            floats(allocator, new float[4], 2, 2), zeroVector, 2, 2);
            KokoroLayers.Linear identity =
                    new KokoroLayers.Linear(
                            floats(allocator, new float[] {1, 0, 0, 1}, 2, 2), zeroVector, 2, 2);
            PlBert.Layer layer =
                    new PlBert.Layer(
                            zero,
                            zero,
                            identity,
                            identity,
                            oneVector,
                            zeroVector,
                            zero,
                            zero,
                            oneVector,
                            zeroVector);
            MemoryView<MemorySegment> input = floats(allocator, new float[] {1, 3, 5, 7}, 2, 2);

            PlBert.transform(layer, input, 1, 2, 1, allocator);

            assertArrayEquals(
                    new float[] {-1, 1, -1, 1}, Views.toFloatArray(input, "output"), 1e-6f);
        }
    }

    private static MemoryView<MemorySegment> floats(
            MemoryAllocator<MemorySegment> allocator, float[] values, long... shape) {
        MemoryView<MemorySegment> view = Views.allocateF32(allocator, shape);
        Views.copyFromArray(view, 0, values, 0, values.length, "fixture");
        return view;
    }
}
