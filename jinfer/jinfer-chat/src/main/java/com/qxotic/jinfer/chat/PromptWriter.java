package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.telemetry.MediaProjectionEvent;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;

/** Streaming prompt builder shared by native templates. */
public final class PromptWriter {
    private final Tokenizer tokenizer;
    private final int batchCapacity;
    private final Consumer<Batch> sink;
    private final MediaEncodingCache mediaCache;
    private final Set<String> specialSpellings;
    private final StringBuilder text = new StringBuilder();
    private IntSequence.Builder tokens = IntSequence.newBuilder();
    private boolean finished;

    public PromptWriter(Tokenizer tokenizer, int batchCapacity, Consumer<Batch> sink) {
        this(tokenizer, batchCapacity, null, sink);
    }

    public PromptWriter(
            Tokenizer tokenizer,
            int batchCapacity,
            MediaEncodingCache mediaCache,
            Consumer<Batch> sink) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        if (batchCapacity <= 0)
            throw new IllegalArgumentException("batchCapacity " + batchCapacity);
        this.batchCapacity = batchCapacity;
        this.mediaCache = mediaCache;
        this.sink = Objects.requireNonNull(sink, "sink");
        this.specialSpellings = SpecialTokens.spellings(tokenizer);
    }

    /** Emits one trusted control token. */
    public PromptWriter id(int token) {
        checkOpen();
        flushText();
        tokens.add(token);
        return this;
    }

    /** Appends untrusted text to the current plain-tokenization run. */
    public PromptWriter text(String value) {
        checkOpen();
        text.append(Objects.requireNonNull(value, "value"));
        return this;
    }

    /** Appends template-authored text, recognizing trusted special-token spellings. */
    public PromptWriter trusted(String value) {
        checkOpen();
        Objects.requireNonNull(value, "value");
        int from = 0;
        while (from < value.length()) {
            int at = value.length();
            String hit = null;
            for (String spelling : specialSpellings) {
                int found = value.indexOf(spelling, from);
                if (found >= 0
                        && (found < at
                                || (found == at
                                        && (hit == null || spelling.length() > hit.length())))) {
                    at = found;
                    hit = spelling;
                }
            }
            text.append(value, from, at);
            if (hit == null) break;
            id(tokenizer.vocabulary().id(hit));
            from = at + hit.length();
        }
        return this;
    }

    /** Splices exact generated payload ids; an empty sequence falls back to plain text upstream. */
    public PromptWriter verbatim(IntSequence value) {
        checkOpen();
        flushText();
        tokens.addAll(Objects.requireNonNull(value, "value"));
        return this;
    }

    /** Emits a model-specific non-token batch at this exact prompt position. */
    public PromptWriter batch(Batch batch) {
        checkOpen();
        Objects.requireNonNull(batch, "batch");
        flushTokens();
        if (batch.count() > batchCapacity)
            throw new IllegalArgumentException(
                    "batch " + batch.count() + " exceeds batchCapacity " + batchCapacity);
        sink.accept(batch);
        return this;
    }

    /**
     * Encodes media and forwards each borrowed embedding chunk synchronously. The chunk is valid
     * only inside the sink call - neither its liveness nor its contents survive the callback (see
     * {@link MediaProjector#project}), so a sink that defers ingestion must copy.
     */
    public <M extends Media> PromptWriter media(
            M source, ContentKey contentKey, MediaProjector<M> projector, boolean bidirectional) {
        checkOpen();
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(projector, "projector");
        flushTokens();
        int expected = projector.positions(source);
        Batch.Positions positions = projector.decoderPositions(source);
        if (positions != null && positions.count() != expected)
            throw new IllegalArgumentException(
                    "projector positions " + positions.count() + " != planned rows " + expected);
        int[] projected = {0};
        projector.project(
                source,
                batchCapacity,
                rows -> {
                    int count = Math.toIntExact(rows.shape().flatAt(0));
                    if (count > batchCapacity)
                        throw new IllegalArgumentException(
                                "projector returned "
                                        + count
                                        + " rows for maxChunkSize "
                                        + batchCapacity);
                    int next = Math.addExact(projected[0], count);
                    if (next > expected)
                        throw new IllegalArgumentException(
                                "projector emitted more than its " + expected + " planned rows");
                    Batch.Positions chunkPositions =
                            positions == null
                                    ? null
                                    : positions.slice(projected[0], count, next == expected);
                    sink.accept(
                            Batch.embeddings(
                                    rows, count, bidirectional, contentKey, chunkPositions));
                    projected[0] = next;
                });
        if (projected[0] != expected)
            throw new IllegalArgumentException(
                    "projector emitted " + projected[0] + " rows, expected " + expected);
        return this;
    }

    /**
     * Emits one structural media item, replaying its projected batches when source-keyed.
     *
     * <p>On a miss (or without caching), {@code encode} runs synchronously with a fresh
     * fragment-scoped writer which this method finishes. The callback must neither retain nor
     * finish it. On a hit the recorded fragment is replayed and the callback is not invoked.
     */
    public void cachedMedia(Media source, ContentKey contentKey, Consumer<PromptWriter> encode) {
        checkOpen();
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(encode, "encode");
        flushTokens();
        if (mediaCache == null) {
            projectMedia(source, encode, sink);
            return;
        }
        mediaCache.replayOrRecord(
                contentKey, batchCapacity, output -> projectMedia(source, encode, output), sink);
    }

    /**
     * Ends the current turn-aligned batch. The block cache commits one block per batch, so a flush
     * IS a cache-block boundary: a later serve restores whole turns, and a conversation that grows
     * by one turn still reuses everything before it. Templates flush after every turn; the
     * generation prompt then lands in its own final batch at {@link #finish()}. A no-op when
     * nothing is pending.
     */
    public PromptWriter flush() {
        checkOpen();
        flushTokens();
        return this;
    }

    /** Flushes the final token run. Exactly one call is required. */
    public void finish() {
        checkOpen();
        finished = true;
        flushTokens();
    }

    private void flushText() {
        if (text.isEmpty()) return;
        tokens.addAll(tokenizer.encode(text.toString()));
        text.setLength(0);
    }

    private void flushTokens() {
        flushText();
        int[] ids = tokens.build().toArray();
        tokens = IntSequence.newBuilder();
        for (int from = 0; from < ids.length; from += batchCapacity) {
            sink.accept(
                    Batch.prefill(
                            Arrays.copyOfRange(
                                    ids, from, Math.min(ids.length, from + batchCapacity))));
        }
    }

    private void checkOpen() {
        if (finished) throw new IllegalStateException("prompt writer already finished");
    }

    private void projectMedia(Media source, Consumer<PromptWriter> encode, Consumer<Batch> output) {
        MediaProjectionEvent event = MediaProjectionEvent.started(source);
        try {
            PromptWriter fragment = new PromptWriter(tokenizer, batchCapacity, null, output);
            encode.accept(fragment);
            fragment.finish();
        } catch (RuntimeException | Error failure) {
            event.errorType = failure.getClass().getSimpleName();
            throw failure;
        } finally {
            event.end();
            event.commit();
        }
    }
}
