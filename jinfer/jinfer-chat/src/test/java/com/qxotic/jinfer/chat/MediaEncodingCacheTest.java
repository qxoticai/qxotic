package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

final class MediaEncodingCacheTest {

    @Test
    void repeatedImageAndVideoProjectOnlyOnce() {
        assertProjectedOnce(
                new Media.Image(new float[] {0, 0, 0}, 1, 1, 3), new ContentKey("image"));
        Media.Image frame = new Media.Image(new float[] {0, 0, 0}, 1, 1, 3);
        assertProjectedOnce(
                new Media.Video(List.of(new Media.Video.Frame(frame, Duration.ZERO))),
                new ContentKey("video"));
    }

    private static void assertProjectedOnce(Media source, ContentKey key) {
        MediaEncodingCache cache = new MediaEncodingCache();
        AtomicInteger projections = new AtomicInteger();
        List<List<Float>> outputs = new ArrayList<>();

        for (int pass = 0; pass < 2; pass++) {
            List<Float> output = new ArrayList<>();
            cache.replayOrRecord(
                    key,
                    8,
                    sink -> {
                        projections.incrementAndGet();
                        try (Arena arena = Arena.ofConfined()) {
                            MemoryView<MemorySegment> rows =
                                    Views.allocateF32(MemoryAllocators.ofArena(arena), 1, 2);
                            float marker = source instanceof Media.Video ? 2 : 1;
                            Views.copyFromArray(
                                    rows, 0, new float[] {marker, marker + 1}, 0, 2, "test rows");
                            sink.accept(Batch.prefill(new int[] {7}));
                            sink.accept(Batch.embeddings(rows, 1, true, key));
                        }
                    },
                    batch -> {
                        if (batch.input() instanceof Batch.Input.Embeddings embeddings) {
                            MemoryView<MemorySegment> rows =
                                    Views.castToSegmentBacked(embeddings.rows(), "test rows");
                            for (float value : Views.toFloatArray(rows, "test rows")) {
                                output.add(value);
                            }
                        }
                    });
            outputs.add(output);
        }

        assertEquals(1, projections.get(), source.getClass().getSimpleName());
        assertEquals(outputs.get(0), outputs.get(1));
        MediaEncodingCache.Sample sample = cache.sample();
        assertEquals(1, sample.entries());
        assertEquals(1, sample.misses());
        assertEquals(1, sample.hits());
        assertEquals(0, sample.refusals());
    }

    @Test
    void oversizedEntriesAreServedButNotRetained() {
        // the bound is HARD: one int[] {7, 8} is 8 bytes, the budget holds only 4
        MediaEncodingCache cache = new MediaEncodingCache(4);
        AtomicInteger projections = new AtomicInteger();
        List<int[]> served = new ArrayList<>();

        for (int pass = 0; pass < 2; pass++) {
            cache.replayOrRecord(
                    new ContentKey("oversized"),
                    8,
                    sink -> {
                        projections.incrementAndGet();
                        sink.accept(Batch.prefill(new int[] {7, 8}));
                    },
                    batch -> served.add(((Batch.Input.Tokens) batch.input()).ids()));
        }

        assertEquals(2, projections.get(), "an oversized entry re-projects on every use");
        assertEquals(2, served.size(), "oversized or not, the caller is always served");
        MediaEncodingCache.Sample sample = cache.sample();
        assertEquals(0, sample.entries());
        assertEquals(0, sample.bytes());
        assertEquals(2, sample.misses());
        assertEquals(0, sample.hits());
        assertEquals(2, sample.refusals());
    }

    @Test
    void theBudgetEvictsEldestWithoutException() {
        // two 8-byte entries, an 8-byte budget: the second insert must evict the first
        MediaEncodingCache cache = new MediaEncodingCache(8);
        cache.replayOrRecord(
                new ContentKey("one"),
                8,
                sink -> sink.accept(Batch.prefill(new int[] {1, 2})),
                batch -> {});
        cache.replayOrRecord(
                new ContentKey("two"),
                8,
                sink -> sink.accept(Batch.prefill(new int[] {3, 4})),
                batch -> {});

        MediaEncodingCache.Sample sample = cache.sample();
        assertEquals(1, sample.entries());
        assertEquals(8, sample.bytes());
        assertEquals(8, sample.budgetBytes());

        // and the survivor is the NEWER entry: key 1 misses again, key 2 hits
        AtomicInteger projections = new AtomicInteger();
        cache.replayOrRecord(
                new ContentKey("one"),
                8,
                sink -> {
                    projections.incrementAndGet();
                    sink.accept(Batch.prefill(new int[] {1, 2}));
                },
                batch -> {});
        assertEquals(1, projections.get(), "the eldest was evicted");
        MediaEncodingCache.Sample after = cache.sample();
        assertEquals(0, after.refusals(), "both entries fit the budget; eviction is not refusal");
        assertEquals(1, after.entries());
        assertEquals(8, after.bytes());
    }
}
