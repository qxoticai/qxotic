package com.qxotic.jinfer.x.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Media;
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
        assertProjectedOnce(new Media.Image(new float[] {0, 0, 0}, 1, 1, 3), new byte[] {1});
        Media.Image frame = new Media.Image(new float[] {0, 0, 0}, 1, 1, 3);
        assertProjectedOnce(
                new Media.Video(List.of(new Media.Video.Frame(frame, Duration.ZERO))),
                new byte[] {2});
    }

    private static void assertProjectedOnce(Media source, byte[] key) {
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
                                    Views.allocateF32(new PanamaMemoryArena(arena), 1, 2);
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
    }
}
