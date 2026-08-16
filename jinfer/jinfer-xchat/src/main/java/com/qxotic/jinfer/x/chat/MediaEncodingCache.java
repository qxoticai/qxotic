package com.qxotic.jinfer.x.chat;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.ContentKey;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/** Bounded per-engine LRU of encoder-projected media batches. */
public final class MediaEncodingCache {

    static final long DEFAULT_BUDGET_BYTES =
            Math.multiplyExact(Math.max(0, Long.getLong("jinfer.mediaCacheMB", 192L)), 1L << 20);

    private record Key(ContentKey content, int batchCapacity) {}

    private sealed interface CachedBatch {
        Batch replay();

        long bytes();

        record Tokens(int[] ids, Batch.Outputs outputs) implements CachedBatch {
            @Override
            public Batch replay() {
                return new Batch(new Batch.Input.Tokens(ids.clone()), outputs);
            }

            @Override
            public long bytes() {
                return (long) ids.length * Integer.BYTES;
            }
        }

        record Embeddings(
                float[] rows,
                int count,
                int dimension,
                boolean bidirectional,
                ContentKey contentKey,
                Batch.Outputs outputs)
                implements CachedBatch {
            @Override
            public Batch replay() {
                MemoryView<MemorySegment> view =
                        Views.wrap(
                                MemorySegment.ofArray(rows),
                                DataType.FP32,
                                Shape.flat(count, dimension));
                return new Batch(
                        new Batch.Input.Embeddings(view, count, bidirectional, contentKey),
                        outputs);
            }

            @Override
            public long bytes() {
                return (long) rows.length * Float.BYTES;
            }
        }
    }

    private final long budgetBytes;
    private final LinkedHashMap<Key, List<CachedBatch>> entries =
            new LinkedHashMap<>(16, 0.75f, true);
    private long usedBytes;
    private long hits, misses, refusals; // refusals: oversized, served but not retained

    /** The whole cache in one reading - what a server's /metrics reports. */
    public record Sample(
            int entries, long bytes, long budgetBytes, long hits, long misses, long refusals) {}

    public MediaEncodingCache() {
        this(DEFAULT_BUDGET_BYTES);
    }

    MediaEncodingCache(long budgetBytes) {
        if (budgetBytes < 0) throw new IllegalArgumentException("negative media cache budget");
        this.budgetBytes = budgetBytes;
    }

    /** Replays a hit, or records one synchronous projection while forwarding it to {@code sink}. */
    // ponytail: the monitor covers the whole projection, so concurrent misses for DIFFERENT
    // content serialize (same-content misses dedupe into one projection, which is the point).
    // Upgrade to a per-key in-flight future if multimodal concurrency ever shows in a profile.
    synchronized void replayOrRecord(
            ContentKey contentKey,
            int batchCapacity,
            Consumer<Consumer<Batch>> projection,
            Consumer<Batch> sink) {
        if (contentKey == null || budgetBytes == 0) {
            projection.accept(sink);
            return;
        }
        Key key = new Key(contentKey, batchCapacity);
        List<CachedBatch> hit = entries.get(key);
        if (hit != null) {
            hits++;
            hit.forEach(batch -> sink.accept(batch.replay()));
            return;
        }
        misses++;

        List<CachedBatch> recorded = new ArrayList<>();
        projection.accept(
                batch -> {
                    recorded.add(copy(batch));
                    sink.accept(batch);
                });
        List<CachedBatch> value = List.copyOf(recorded);
        long valueBytes = bytes(value);
        // The bound is HARD: an entry bigger than the whole budget is served but never
        // retained - retaining it would make the budget a lie. Repeated use of such media
        // re-projects every time; raise jinfer.mediaCacheMB to make it cacheable.
        if (valueBytes > budgetBytes) {
            refusals++;
            return;
        }
        entries.put(key, value);
        usedBytes += valueBytes;
        var eldest = entries.entrySet().iterator();
        while (usedBytes > budgetBytes) {
            Map.Entry<Key, List<CachedBatch>> entry = eldest.next();
            usedBytes -= bytes(entry.getValue());
            eldest.remove();
        }
    }

    /** A consistent snapshot of the counters and the resident footprint. */
    public synchronized Sample sample() {
        return new Sample(entries.size(), usedBytes, budgetBytes, hits, misses, refusals);
    }

    synchronized void clear() {
        entries.clear();
        usedBytes = 0;
    }

    private static CachedBatch copy(Batch batch) {
        return switch (batch.input()) {
            case Batch.Input.Tokens tokens ->
                    new CachedBatch.Tokens(tokens.ids().clone(), batch.outputs());
            case Batch.Input.Embeddings embeddings -> {
                MemoryView<MemorySegment> rows =
                        Views.castToSegmentBacked(embeddings.rows(), "media embedding rows");
                yield new CachedBatch.Embeddings(
                        Views.toFloatArray(rows, "media embedding rows"),
                        embeddings.count(),
                        Math.toIntExact(rows.shape().flatAt(1)),
                        embeddings.bidirectional(),
                        embeddings.contentKey(),
                        batch.outputs());
            }
            case Batch.Input.Sequences ignored ->
                    throw new IllegalArgumentException("media encoding emitted packed sequences");
        };
    }

    private static long bytes(List<CachedBatch> batches) {
        long total = 0;
        for (CachedBatch batch : batches) total = Math.addExact(total, batch.bytes());
        return total;
    }
}
