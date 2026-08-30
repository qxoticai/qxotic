package com.qxotic.jinfer.cache;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The media fingerprint law after content keys: a batch carrying a SOURCE digest fingerprints by
 * that digest and its decoder positions - row bits do not matter (they drift by an ulp while the
 * JIT warms, which made the same image miss across a server's early requests) - while a key-less
 * batch keeps the exact row-hash behavior every existing API caller and test relies on.
 */
final class MediaFingerprintTest {

    private static Batch embeddings(Arena arena, ContentKey key, float seed) {
        return embeddings(arena, key, seed, null);
    }

    private static Batch embeddings(
            Arena arena, ContentKey key, float seed, Batch.Positions positions) {
        MemoryView<MemorySegment> rows = Views.allocateF32(MemoryAllocators.ofArena(arena), 2, 4);
        float[] values = new float[8];
        for (int i = 0; i < values.length; i++) values[i] = seed + i;
        Views.copyFromArray(rows, 0, values, 0, values.length, "rows");
        return Batch.embeddings(rows, 2, true, key, positions);
    }

    @Test
    void contentKeyedBatchesFingerprintByTheKeyNotTheRows() {
        ContentKey key = new ContentKey("media:one");
        try (Arena arena = Arena.ofConfined()) {
            long[] a = CachedSession.fingerprints(List.of(embeddings(arena, key, 0.5f)));
            long[] b = CachedSession.fingerprints(List.of(embeddings(arena, key, 99.5f)));
            assertArrayEquals(
                    a, b, "same source key must fingerprint identically across row drift");

            ContentKey other = new ContentKey("media:other");
            long[] c = CachedSession.fingerprints(List.of(embeddings(arena, other, 0.5f)));
            assertFalse(Arrays.equals(a, c), "a different source must fingerprint differently");
            long[] positioned =
                    CachedSession.fingerprints(
                            List.of(
                                    embeddings(
                                            arena,
                                            key,
                                            0.5f,
                                            new Batch.Positions(
                                                    3, new int[] {0, 0, 0, 0, 1, 1}, 2))));
            assertFalse(
                    Arrays.equals(a, positioned),
                    "different decoder positions must fingerprint differently");
        }
    }

    @Test
    void keylessBatchesKeepTheRowHashBehavior() {
        try (Arena arena = Arena.ofConfined()) {
            long[] a = CachedSession.fingerprints(List.of(embeddings(arena, null, 0.5f)));
            long[] sameRows = CachedSession.fingerprints(List.of(embeddings(arena, null, 0.5f)));
            long[] otherRows = CachedSession.fingerprints(List.of(embeddings(arena, null, 99.5f)));
            long[] otherPositions =
                    CachedSession.fingerprints(
                            List.of(
                                    embeddings(
                                            arena,
                                            null,
                                            0.5f,
                                            new Batch.Positions(
                                                    3, new int[] {0, 0, 0, 0, 1, 1}, 2))));
            assertArrayEquals(a, sameRows, "identical rows, identical fingerprints");
            assertFalse(Arrays.equals(a, otherRows), "different rows, different fingerprints");
            assertFalse(
                    Arrays.equals(a, otherPositions),
                    "different decoder positions, different fingerprints");
        }
    }
}
