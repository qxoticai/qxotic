package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Model;
import com.qxotic.jinfer.RuntimeState;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;

/**
 * The dual representation bound to a state: the exact ingested fingerprint stream (token ids)
 * alongside the KV, with a cache commit at every ingestion boundary so the committed chain is
 * always contiguous. This is the cache's WRITE HANDLE — it holds the tip of the committed chain and
 * extends it by exactly each ingested span. Ingestion chunks at the state's batch capacity; each
 * chunk is one block (large blocks), each decode {@link #step} is one block (single-token blocks).
 *
 * <p>Attach with {@link #resume}: restores the longest cached prefix of {@code expected} into the
 * fresh state; the caller re-ingests everything past {@link #position()}.
 *
 * <p>Media just works: an {@link Batch.Input.Embeddings} batch contributes per-position
 * fingerprints derived from a SHA-256 of its raw row bits, spread across positions ({@code digest[i
 * & 3] + GOLDEN * i}) so the full 256-bit content identity enters the chained block key — same
 * media, same encoder, same fingerprints; different media diverge at the block. {@link
 * Batch#prepare} keeps each embeddings batch isolated, so a bidirectional image block (one
 * attention group) commits as exactly one cache block. Note the fingerprints hash the ENCODED rows:
 * re-fingerprinting an echoed conversation needs either the retained stream ({@link
 * #fingerprints()}, the dual view) or a re-encode — servers keep the stream.
 */
public final class CachedSession<S extends RuntimeState> {

    private final Model<?, ?, S> model;
    private final S state;
    private final PromptCache<S> cache;
    private PromptCache<S>.Block tip;
    private long[] fp;
    private int len;

    private CachedSession(
            Model<?, ?, S> model,
            S state,
            PromptCache<S> cache,
            PromptCache<S>.Block tip,
            long[] fp,
            int len) {
        this.model = model;
        this.state = state;
        this.cache = cache;
        this.tip = tip;
        this.fp = fp;
        this.len = len;
    }

    /**
     * A fresh session on a fresh state, resuming the longest cached prefix of {@code expected}
     * (empty for a brand-new conversation).
     */
    public static <S extends RuntimeState> CachedSession<S> resume(
            Model<?, ?, S> model, PromptCache<S> cache, S state, long[] expected) {
        return resume(model, cache, state, expected, expected.length);
    }

    /**
     * Like {@link #resume(Model, PromptCache, Object, long[])} but restoring at most {@code
     * maxPositions} — e.g. the prompt length minus its final block, so a whole-prompt hit still
     * re-ingests that block and leaves fresh logits at the cursor.
     */
    public static <S extends RuntimeState> CachedSession<S> resume(
            Model<?, ?, S> model,
            PromptCache<S> cache,
            S state,
            long[] expected,
            int maxPositions) {
        PromptCache<S>.Block tip =
                cache.resume(expected, Math.min(expected.length, maxPositions), state);
        long[] fp = Arrays.copyOf(expected, Math.max(256, expected.length));
        return new CachedSession<>(model, state, cache, tip, fp, tip.to);
    }

    /**
     * Ingests batches (chunked at the state's batch capacity), committing each chunk: token ids
     * fingerprint as themselves, embeddings by rows content hash (one block per media group).
     */
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
                default ->
                        throw new IllegalArgumentException(
                                "CachedSession cannot fingerprint "
                                        + b.input().getClass().getSimpleName());
            }
            tip = cache.commit(tip, fp, off, len - off, state);
        }
    }

    /**
     * Ingests turn-aligned groups whose flattened fingerprints formed this session's resume stream,
     * skipping what the resume already restored: whole groups before the tip, and the restored HEAD
     * of a partially-covered group. A cache hit ends on a BLOCK boundary, which need not be a group
     * boundary (a previous generation prompt is a byte-exact prefix of the echoed assistant turn;
     * long turns commit as several blocks) - re-ingesting the whole group there would duplicate its
     * restored head in the context and poison the cache.
     */
    public void ingestGroups(List<List<Batch>> groups) {
        int restored = state.position();
        int pos = 0;
        for (List<Batch> group : groups) {
            int glen = 0;
            for (Batch b : group) glen += b.count();
            int end = pos + glen;
            if (end <= restored) { // fully restored: skip
                pos = end;
                continue;
            }
            ingest(pos >= restored ? group : tail(group, restored - pos));
            pos = end;
        }
    }

    /**
     * The group minus its first {@code skip} positions - whole batches drop, a token batch at the
     * seam is sliced. A hit strictly inside a media batch cannot happen (a media group is one
     * block, and its content-hash fingerprints never collide with token ids).
     */
    private static List<Batch> tail(List<Batch> group, int skip) {
        List<Batch> out = new java.util.ArrayList<>();
        for (Batch b : group) {
            int n = b.count();
            if (skip >= n) {
                skip -= n;
                continue;
            }
            if (skip == 0) {
                out.add(b);
                continue;
            }
            if (!(b.input() instanceof Batch.Input.Tokens t)) {
                throw new IllegalStateException("cache hit inside a non-token batch");
            }
            out.add(Batch.prefill(Arrays.copyOfRange(t.ids(), skip, n)));
            skip = 0;
        }
        return out;
    }

    private static final long GOLDEN = 0x9E3779B97F4A7C15L;

    /**
     * SHA-256 of the raw row bits, as 4 longs — the media block's content identity. Flat F32
     * tensors (the embedder output) hash via bulk raw copies; the per-element path remains the
     * encoding-generic fallback (same LE float-bit stream either way).
     */
    private static long[] rowsDigest(Batch.Input.Embeddings e) {
        MessageDigest sha = Sha256.sha256();
        long size = e.rows().size();
        ByteBuffer buf = ByteBuffer.allocate(1 << 16).order(ByteOrder.LITTLE_ENDIAN);
        try {
            MemorySegment chunk = MemorySegment.ofBuffer(buf);
            for (long off = 0; off < size; off += buf.capacity() / Float.BYTES) {
                long n = Math.min(buf.capacity() / Float.BYTES, size - off);
                long bytes = e.rows().copyRawTo(off, chunk, 0, n); // bulk: one segment copy
                sha.update(buf.array(), 0, (int) bytes);
            }
        } catch (UnsupportedOperationException fallback) { // non-flat encodings
            sha.reset();
            buf.clear();
            for (long i = 0; i < size; i++) {
                if (buf.remaining() < Integer.BYTES) {
                    sha.update(buf.array(), 0, buf.position());
                    buf.clear();
                }
                buf.putInt(Float.floatToRawIntBits(e.rows().getFloat(i)));
            }
            sha.update(buf.array(), 0, buf.position());
        }
        return Sha256.digestLongs(sha);
    }

    /** Ingests one decode step and commits it as a single-token block. */
    public void step(int token) {
        model.ingest(state, Batch.step(token));
        append(token);
        tip = cache.commit(tip, fp, len - 1, 1, state);
    }

    /**
     * Adopts decode-loop tokens that were ingested directly on the state (the generator steps the
     * state itself, not the session): appends their fingerprints and commits them as ONE block -
     * the right granularity, since a block's fixed residue would otherwise be duplicated per decode
     * token. The caller passes exactly the ingested tokens ({@code state.position() - length()} of
     * them - a trailing stop or budget-final token is sampled but never ingested); {@code commit}'s
     * position check enforces the accounting.
     */
    public void adopt(List<Integer> ingested) {
        if (ingested.isEmpty()) return;
        int off = len;
        for (int id : ingested) append(id);
        tip = cache.commit(tip, fp, off, len - off, state);
    }

    /**
     * Fingerprint stream length. Equals {@link #position()} while every ingestion goes through the
     * session.
     */
    public int length() {
        return len;
    }

    /**
     * True when this session's WHOLE stream is a strict prefix of {@code req[0..reqLen)} — the
     * append-only reuse test ({@link SessionPool}): the live state can continue with the remainder,
     * nothing to rewind, and at least one position is left to ingest.
     */
    public boolean streamIsStrictPrefixOf(long[] req, int reqLen) {
        return len < reqLen && Arrays.equals(fp, 0, len, req, 0, len);
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
