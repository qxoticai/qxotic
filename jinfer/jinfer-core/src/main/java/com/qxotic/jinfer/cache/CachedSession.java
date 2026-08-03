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
 * alongside the KV. In its full mode a cache commit lands at every ingestion boundary, keeping the
 * committed chain contiguous - the cache's WRITE HANDLE. Two reduced modes exist for the facade:
 * HOT-ONLY (no tree: the stream still tracks every position, nothing commits or restores) and
 * READ-ONLY (coarse serving: restores, never writes). It is otherwise the write handle — it holds
 * the tip of the committed chain and extends it by exactly each ingested span. Ingestion chunks at
 * the state's batch capacity; each chunk is one block (large blocks), each decode {@link #step} is
 * one block (single-token blocks).
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
 * #fingerprints(List)}, the dual view) or a re-encode — servers keep the stream.
 *
 * <p>{@code start} opens a brand-new conversation (ingest incrementally); {@code resume(model,
 * cache, state, prompt)} serves a prompt against the cache (longest cached prefix restored, the
 * caller ingests the tail) - {@code maxPositions} (pass {@code total - 1}) leaves the final block
 * re-ingested so the cursor holds fresh logits.
 */
public final class CachedSession<S extends RuntimeState> {

    private final Model<?, ?, S> model;
    private final S state;
    private final BlockTree<S> cache; // null: HOT-ONLY (stream tracked, nothing committed)
    private final boolean commits; // false: read-only serving (coarse codecs restore, never write)
    private BlockTree<S>.Block tip;
    private long[] fp;
    private int len;

    private CachedSession(
            Model<?, ?, S> model,
            S state,
            BlockTree<S> cache,
            boolean commits,
            BlockTree<S>.Block tip,
            long[] fp,
            int len) {
        this.model = model;
        this.state = state;
        this.cache = cache;
        this.commits = commits && cache != null;
        this.tip = tip;
        this.fp = fp;
        this.len = len;
    }

    /**
     * One commit site for every write path; a hot-only or read-only session records only the
     * fingerprint stream - the state itself is the cache.
     */
    private void commitSpan(int off, int len) {
        if (commits) tip = cache.commit(tip, fp, off, len, state);
    }

    /**
     * A session over NO tree: the fingerprint stream still tracks every ingested position (hot
     * matching needs it), but nothing is committed and nothing can be restored.
     */
    static <S extends RuntimeState> CachedSession<S> hot(Model<?, ?, S> model, S state) {
        // a non-zero position would seed a stream of zero fingerprints for content the state
        // actually holds - a false prefix match away from serving wrong bytes. Refuse.
        if (state.position() != 0) {
            throw new IllegalArgumentException(
                    "hot session needs a wiped state (position " + state.position() + ")");
        }
        return new CachedSession<>(model, state, null, false, null, new long[256], 0);
    }

    /**
     * As {@link #resume(Model, BlockTree, Object, long[], int)} but READ-ONLY: the longest cached
     * prefix restores, and no ingestion ever writes back - the coarse-codec serving mode (a residue
     * per served block would grow the store by ~MBs per request).
     */
    static <S extends RuntimeState> CachedSession<S> resumeReadOnly(
            Model<?, ?, S> model, BlockTree<S> cache, S state, long[] expected, int maxPositions) {
        BlockTree<S>.Block tip =
                cache.resume(expected, Math.min(expected.length, maxPositions), state);
        long[] fp = Arrays.copyOf(expected, Math.max(256, expected.length));
        return new CachedSession<>(model, state, cache, false, tip, fp, tip.to);
    }

    /** A fresh session on a fresh state for a brand-new conversation (nothing to resume). */
    public static <S extends RuntimeState> CachedSession<S> start(
            Model<?, ?, S> model, BlockTree<S> cache, S state) {
        return resume(model, cache, state, new long[0], 0);
    }

    /**
     * A fresh session on a fresh state, resuming the longest cached prefix of {@code prompt} - the
     * encoded batches themselves are the key; their content addressing (token ids as themselves,
     * media rows by content digest) is this package's internal law.
     */
    public static <S extends RuntimeState> CachedSession<S> resume(
            Model<?, ?, S> model, BlockTree<S> cache, S state, List<Batch> prompt) {
        long[] fp = fingerprints(prompt);
        return resume(model, cache, state, fp, fp.length);
    }

    /**
     * As {@link #resume(Model, BlockTree, RuntimeState, List)} but restoring at most {@code
     * maxPositions} - e.g. the prompt length minus its final block, so a whole-prompt hit still
     * re-ingests that block and leaves fresh logits at the cursor.
     */
    public static <S extends RuntimeState> CachedSession<S> resume(
            Model<?, ?, S> model,
            BlockTree<S> cache,
            S state,
            List<Batch> prompt,
            int maxPositions) {
        return resume(model, cache, state, fingerprints(prompt), maxPositions);
    }

    /**
     * A fresh session on a fresh state, resuming the longest cached prefix of {@code expected}
     * (empty for a brand-new conversation).
     */
    static <S extends RuntimeState> CachedSession<S> resume(
            Model<?, ?, S> model, BlockTree<S> cache, S state, long[] expected) {
        return resume(model, cache, state, expected, expected.length);
    }

    /**
     * Like {@link #resume(Model, BlockTree, Object, long[])} but restoring at most {@code
     * maxPositions} — e.g. the prompt length minus its final block, so a whole-prompt hit still
     * re-ingests that block and leaves fresh logits at the cursor.
     */
    static <S extends RuntimeState> CachedSession<S> resume(
            Model<?, ?, S> model, BlockTree<S> cache, S state, long[] expected, int maxPositions) {
        BlockTree<S>.Block tip =
                cache.resume(expected, Math.min(expected.length, maxPositions), state);
        long[] fp = Arrays.copyOf(expected, Math.max(256, expected.length));
        return new CachedSession<>(model, state, cache, true, tip, fp, tip.to);
    }

    /** Token ids widened to the fingerprint stream they are (media rows fingerprint by hash). */
    static long[] fingerprints(int[] tokens) {
        long[] fp = new long[tokens.length];
        for (int i = 0; i < tokens.length; i++) fp[i] = tokens[i];
        return fp;
    }

    /**
     * The expected fingerprint stream of a batch list, media included - EXACTLY what {@link
     * #ingest} appends (token ids as themselves, embedding rows by content digest), so a caller can
     * {@link #resume} against a prompt before ingesting it.
     */
    static long[] fingerprints(List<Batch> batches) {
        int total = batches.stream().mapToInt(Batch::count).sum();
        long[] fp = new long[total];
        int at = 0;
        for (Batch b : batches) {
            switch (b.input()) {
                case Batch.Input.Tokens t -> {
                    for (int id : t.ids()) fp[at++] = id;
                }
                case Batch.Input.Embeddings e -> {
                    long[] digest = rowsDigest(e);
                    for (int i = 0; i < e.count(); i++) fp[at++] = digest[i & 3] + GOLDEN * i;
                }
                default ->
                        throw new IllegalArgumentException(
                                "CachedSession cannot fingerprint "
                                        + b.input().getClass().getSimpleName());
            }
        }
        return fp;
    }

    /**
     * Ingests batches (chunked at the state's batch capacity), committing each chunk: token ids
     * fingerprint as themselves, embeddings by rows content hash (one block per media group).
     */
    public void ingest(List<Batch> batches) {
        for (Batch b : Batch.prepare(batches, state.batchCapacity())) {
            int off = len;
            long[] f = fingerprints(List.of(b)); // the ONE fingerprint law (see fingerprints)
            model.ingest(state, b);
            for (long v : f) append(v);
            commitSpan(off, len - off);
        }
    }

    /**
     * Ingests {@code ids[from..)} with the final token committed as its own block - the
     * PROMPT-COMPILER CONVENTION shared by {@link FrozenBlocks#compile} and the accumulating CLI
     * cache: a later resume capped at N-1 lands exactly one token short, and the single-token
     * re-ingest materializes fresh logits.
     */
    public void ingestSplitLast(int[] ids, int from) {
        if (from >= ids.length) return;
        int n = ids.length;
        if (n - from > 1) {
            ingest(List.of(Batch.prefill(java.util.Arrays.copyOfRange(ids, from, n - 1))));
        }
        ingest(List.of(Batch.prefill(new int[] {ids[n - 1]})));
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
     * The batch list minus its first {@code skip} positions: whole batches drop, a token batch at
     * the seam is sliced. A seam strictly inside a media batch cannot happen when {@code skip} came
     * from a block-aligned resume (media groups commit and restore whole) - it throws loudly.
     */
    public static List<Batch> tail(List<Batch> group, int skip) {
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
        commitSpan(len - 1, 1);
    }

    /**
     * Records ONE decode token the generator just ingested (the state's frontier includes it):
     * appends its fingerprint and commits it as its own single-token block - {@link #step} minus
     * the ingestion. Call it step-time, from the decode loop's after-ingest hook; a commit must
     * save at the frontier (ring rows alias, residues move), which is why there is no per-token
     * back-fill overload.
     *
     * <p>This granularity is the APPEND-ONLY CONVERSATION CONTRACT: interior content resumes at
     * block boundaries, but the stream's tail - the reply the next request will echo - keeps EVERY
     * position resumable, so an echo that truncates at a stop string or edits the trailing text
     * resumes token-exact instead of re-prefilling from the last chunk boundary.
     *
     * <p>Cost: one residue per decode token. Free where the residue is ~0 (dense KV, ring rows);
     * lfm2's ~340KB conv residue makes long replies heavy - if that ever matters, the upgrade path
     * is compacting a reply's singles into one block once the next turn extends past them, not
     * coarsening the tail.
     */
    public void adopt(int token) {
        append(token);
        commitSpan(len - 1, 1);
    }

    /**
     * Bulk adoption of decode-loop tokens ingested directly on the state, committed as ONE block
     * saved at the frontier (sound exactly like a prompt chunk: rows and residue are read at the
     * position the state is actually at) - the speculative-decode shape, where several accepted
     * tokens land on the state before control returns. For the serving path's reply tail, prefer
     * per-token {@link #adopt(int)}: one bulk block resumes only at its end.
     *
     * <p>The caller passes exactly the ingested tokens ({@code state.position() - length()} of them
     * - a trailing stop or budget-final token is sampled but never ingested); {@code commit}'s
     * position check enforces the accounting.
     */
    public void adopt(int[] ingested) {
        if (ingested.length == 0) return;
        int off = len;
        for (int id : ingested) append(id);
        commitSpan(off, len - off);
    }

    /**
     * True when the budget refused a commit and DETACHED this session's tip: the state and stream
     * stay valid (tier-1 reuse unaffected) but nothing further reaches the shared tree. Permanent
     * for this session's lifetime - the cue for a definer to fail loudly rather than pretend.
     */
    public boolean detached() {
        return commits && !tip.live;
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
     * append-only reuse test (the facade's hot layer): the live state can continue with the
     * remainder, nothing to rewind, and at least one position is left to ingest.
     */
    boolean streamIsStrictPrefixOf(long[] req, int reqLen) {
        return len < reqLen && Arrays.equals(fp, 0, len, req, 0, len);
    }

    public int position() {
        return state.position();
    }

    public S state() {
        return state;
    }

    /**
     * The ingested conversation as plain token ids - only meaningful for text-only sessions; throws
     * if any position is a media digest (media cannot be replayed from ids).
     */
    public int[] tokenIds() {
        int[] ids = new int[len];
        for (int i = 0; i < len; i++) {
            long v = fp[i];
            if (v != (int) v) {
                throw new IllegalStateException("session ingested media; no token-id replay");
            }
            ids[i] = (int) v;
        }
        return ids;
    }

    /**
     * The position count of {@code prefix} when its whole content stream is a STRICT prefix of
     * {@code whole}'s (at least one position of {@code whole} remains), else -1 - the append-only
     * reuse test, batches-in so callers never touch the content addressing.
     */
    public static int strictPrefixPositions(List<Batch> prefix, List<Batch> whole) {
        long[] a = fingerprints(prefix);
        long[] b = fingerprints(whole);
        if (a.length >= b.length || !Arrays.equals(a, 0, a.length, b, 0, a.length)) return -1;
        return a.length;
    }

    private void append(long fingerprint) {
        if (len == fp.length) fp = Arrays.copyOf(fp, fp.length * 2);
        fp[len++] = fingerprint;
    }
}
