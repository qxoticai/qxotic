package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Model;
import com.qxotic.jinfer.RuntimeState;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.List;

/** The dual representation bound to a state: the exact ingested fingerprint stream (token ids)
 *  alongside the KV, with a cache commit at every ingestion boundary so the committed chain is
 *  always contiguous — the protocol {@link PromptCache} requires, provided once here instead of
 *  re-written by every driver. Ingestion chunks at the state's batch capacity; each chunk is one
 *  block (large blocks), each decode {@link #step} is one block (single-token blocks).
 *
 *  <p>Attach with {@link #resume}: restores the longest cached prefix of {@code expected} into the
 *  fresh state; the caller re-ingests everything past {@link #position()}.
 *
 *  <p>Media just works: an {@link Batch.Input.Embeddings} batch contributes per-position
 *  fingerprints derived from a SHA-256 of its raw row bits, spread across positions
 *  ({@code digest[i & 3] + GOLDEN * i}) so the full 256-bit content identity enters the chained
 *  block key — same media, same encoder, same fingerprints; different media diverge at the block.
 *  {@link Batch#prepare} keeps each embeddings batch isolated, so a bidirectional image block (one
 *  attention group) commits as exactly one cache block. Note the fingerprints hash the ENCODED
 *  rows: re-fingerprinting an echoed conversation needs either the retained stream
 *  ({@link #fingerprints()}, the dual view) or a re-encode — servers keep the stream. */
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
        return resume(model, cache, state, expected, expected.length);
    }

    /** Like {@link #resume(Model, PromptCache, Object, long[])} but restoring at most
     *  {@code maxPositions} — e.g. the prompt length minus its final block, so a whole-prompt hit
     *  still re-ingests that block and leaves fresh logits at the cursor. */
    public static <S extends RuntimeState> CachedSession<S> resume(
            Model<?, ?, S> model, PromptCache<S> cache, S state, long[] expected, int maxPositions) {
        PromptCache<S>.Cursor cursor = cache.resume(expected, Math.min(expected.length, maxPositions), state);
        long[] fp = Arrays.copyOf(expected, Math.max(256, expected.length));
        return new CachedSession<>(model, state, cursor, fp, cursor.position());
    }

    /** Ingests batches (chunked at the state's batch capacity), committing each chunk: token ids
     *  fingerprint as themselves, embeddings by rows content hash (one block per media group). */
    public void ingest(List<Batch> batches) {
        for (Batch b : Batch.prepare(batches, state.batchCapacity())) {
            int off = len;
            switch (b.input()) {
                case Batch.Input.Tokens t -> {
                    model.ingest(state, b);
                    for (int id : t.ids()) append(id);
                }
                case Batch.Input.Embeddings e -> {
                    long[] digest = rowsDigest(e);
                    model.ingest(state, b);
                    for (int i = 0; i < e.count(); i++) append(digest[i & 3] + GOLDEN * i);
                }
                default -> throw new IllegalArgumentException(
                        "CachedSession cannot fingerprint " + b.input().getClass().getSimpleName());
            }
            cursor.commit(fp, off, len - off, state);
        }
    }

    private static final long GOLDEN = 0x9E3779B97F4A7C15L;

    /** SHA-256 of the raw row bits, as 4 longs — the media block's content identity. */
    private static long[] rowsDigest(Batch.Input.Embeddings e) {
        MessageDigest sha;
        try {
            sha = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException ex) {
            throw new AssertionError(ex);
        }
        long size = e.rows().size();
        ByteBuffer buf = ByteBuffer.allocate(8192).order(ByteOrder.LITTLE_ENDIAN);
        for (long i = 0; i < size; i++) {
            if (buf.remaining() < Integer.BYTES) {
                sha.update(buf.array(), 0, buf.position());
                buf.clear();
            }
            buf.putInt(Float.floatToRawIntBits(e.rows().getFloat(i)));
        }
        sha.update(buf.array(), 0, buf.position());
        ByteBuffer out = ByteBuffer.wrap(sha.digest()).order(ByteOrder.LITTLE_ENDIAN);
        return new long[]{out.getLong(), out.getLong(), out.getLong(), out.getLong()};
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
