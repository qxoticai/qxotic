package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Model;
import com.qxotic.jinfer.RuntimeState;

import java.util.Arrays;
import java.util.List;

/** The dual representation bound to a state: the exact ingested fingerprint stream (token ids)
 *  alongside the KV, with a cache commit at every ingestion boundary so the committed chain is
 *  always contiguous — the protocol {@link PromptCache} requires, provided once here instead of
 *  re-written by every driver. Ingestion chunks at the state's batch capacity; each chunk is one
 *  block (large blocks), each decode {@link #step} is one block (single-token blocks).
 *
 *  <p>Attach with {@link #resume}: restores the longest cached prefix of {@code expected} into the
 *  fresh state; the caller re-ingests everything past {@link #position()}. Text-only for now — a
 *  media batch's fingerprint (content hash) lands with the multimodal wiring. */
public final class CachedSession<S extends RuntimeState> {

    private final Model<?, ?, S> model;
    private final S state;
    private final PromptCache<S>.Cursor cursor;
    private long[] fp;
    private int len;

    private CachedSession(Model<?, ?, S> model, S state, PromptCache<S>.Cursor cursor, long[] fp, int len) {
        this.model = model;
        this.state = state;
        this.cursor = cursor;
        this.fp = fp;
        this.len = len;
    }

    /** A fresh session on a fresh state, resuming the longest cached prefix of {@code expected}
     *  (empty for a brand-new conversation). */
    public static <S extends RuntimeState> CachedSession<S> resume(
            Model<?, ?, S> model, PromptCache<S> cache, S state, long[] expected) {
        PromptCache<S>.Cursor cursor = cache.resume(expected, expected.length, state);
        long[] fp = Arrays.copyOf(expected, Math.max(256, expected.length));
        return new CachedSession<>(model, state, cursor, fp, cursor.position());
    }

    /** Ingests token batches (chunked at the state's batch capacity), committing each chunk. */
    public void ingest(List<Batch> batches) {
        for (Batch b : Batch.prepare(batches, state.batchCapacity())) {
            if (!(b.input() instanceof Batch.Input.Tokens t)) {
                throw new IllegalArgumentException("CachedSession is token-only for now: " + b.input().getClass().getSimpleName());
            }
            int[] ids = t.ids();
            model.ingest(state, b);
            int off = len;
            for (int id : ids) append(id);
            cursor.commit(fp, off, ids.length, state);
        }
    }

    /** Ingests one decode step and commits it as a single-token block. */
    public void step(int token) {
        model.ingest(state, Batch.step(token));
        append(token);
        cursor.commit(token, state);
    }

    public int position() {
        return state.position();
    }

    public S state() {
        return state;
    }

    /** The exact ingested fingerprint stream so far (the low-level half of the dual view). */
    public long[] fingerprints() {
        return Arrays.copyOf(fp, len);
    }

    private void append(long fingerprint) {
        if (len == fp.length) fp = Arrays.copyOf(fp, fp.length * 2);
        fp[len++] = fingerprint;
    }
}
