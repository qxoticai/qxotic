package com.qxotic.jinfer.cache;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.F32FloatTensor;
import java.lang.foreign.Arena;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The media fingerprint law after content keys: a batch carrying a SOURCE digest fingerprints by
 * that digest alone - row bits do not matter (they drift by an ulp while the JIT warms, which made
 * the same image miss across a server's early requests) - while a key-less batch keeps the exact
 * row-hash behavior every existing API caller and test relies on.
 */
final class MediaFingerprintTest {

    private static Batch embeddings(Arena arena, byte[] key, float seed) {
        F32FloatTensor rows = F32FloatTensor.allocate(arena, 8);
        for (int i = 0; i < 8; i++) rows.setFloat(i, seed + i);
        return Batch.embeddings(rows, 2, true, key);
    }

    @Test
    void contentKeyedBatchesFingerprintByTheKeyNotTheRows() throws Exception {
        byte[] key = MessageDigest.getInstance("SHA-256").digest(new byte[] {1, 2, 3});
        try (Arena arena = Arena.ofConfined()) {
            long[] a = CachedSession.fingerprints(List.of(embeddings(arena, key, 0.5f)));
            long[] b = CachedSession.fingerprints(List.of(embeddings(arena, key, 99.5f)));
            assertArrayEquals(
                    a, b, "same source key must fingerprint identically across row drift");

            byte[] other = MessageDigest.getInstance("SHA-256").digest(new byte[] {9});
            long[] c = CachedSession.fingerprints(List.of(embeddings(arena, other, 0.5f)));
            assertFalse(Arrays.equals(a, c), "a different source must fingerprint differently");
        }
    }

    @Test
    void keylessBatchesKeepTheRowHashBehavior() {
        try (Arena arena = Arena.ofConfined()) {
            long[] a = CachedSession.fingerprints(List.of(embeddings(arena, null, 0.5f)));
            long[] sameRows = CachedSession.fingerprints(List.of(embeddings(arena, null, 0.5f)));
            long[] otherRows = CachedSession.fingerprints(List.of(embeddings(arena, null, 99.5f)));
            assertArrayEquals(a, sameRows, "identical rows, identical fingerprints");
            assertFalse(Arrays.equals(a, otherRows), "different rows, different fingerprints");
        }
    }
}
