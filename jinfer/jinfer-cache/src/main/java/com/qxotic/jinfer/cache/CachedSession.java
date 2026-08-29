package com.qxotic.jinfer.cache;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContextModel;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BooleanSupplier;

/**
 * The dual representation bound to a state: the exact ingested fingerprint stream (token ids)
 * alongside the KV. In its full mode a cache commit lands at every ingestion boundary, keeping the
 * committed chain contiguous - the cache's WRITE HANDLE. Two reduced modes exist for the facade:
 * SESSIONS-ONLY (no tree: the stream still tracks every position, nothing commits or restores) and
 * READ-ONLY (define-only serving: restores, never writes). It is otherwise the write handle - it
 * holds the tip of the committed chain and extends it by exactly each ingested span. Ingestion
 * chunks at the state's batch capacity; each chunk is one block (large blocks), each decode {@link
 * #step} is one block (single-token blocks).
 *
 * <p>Attach with {@link #resume}: restores the longest cached prefix of {@code expected} into the
 * fresh state; the caller re-ingests everything past {@link #position()}.
 *
 * <p>Media just works: an {@link Batch.Input.Embeddings} batch contributes per-position
 * fingerprints derived from its stable content key, or from a SHA-256 of its raw row bits when no
 * key is supplied. The digest is spread across positions ({@code digest[i & 3] + GOLDEN * i}) so
 * the full 256-bit identity enters the chained block key. {@link Batch#prepare} keeps each
 * embeddings batch isolated, so a bidirectional image block (one attention group) commits as
 * exactly one cache block.
 *
 * <p>{@code start} opens a brand-new conversation (ingest incrementally); {@code resume(model,
 * cache, state, prompt)} serves a prompt against the cache (longest cached prefix restored, the
 * caller ingests the tail) - {@code maxPositions} (pass {@code total - 1}) leaves the final block
 * re-ingested so the state holds fresh logits at its frontier.
 */
public final class CachedSession<S extends ContextState> {

    private final ContextModel<?, ?, S> model;
    private final S state;
    private final BlockTree<S> cache; // null: SESSIONS-ONLY (stream tracked, nothing committed)
    private final boolean commits; // false: define-only mode restores but never writes
    private BlockTree<S>.Block tip;
    private long[] fp;
    private int len;
    // TAIL SNAPSHOT (define-only mode): recurrent state at the last prompt boundary.
    // The attention rows [0, snapshotLen) stay valid in the state itself (linear append-only KV;
    // decode only appends), so rewinding needs ONLY the residue restored and the position moved -
    // which un-does the reply's recurrent influence and lets a thinking-stripped echo (never a
    // strict extension of the generated stream) continue from the boundary instead of
    // re-prefilling the whole conversation.
    private MemorySegment snapshot; // residue blob, null = no snapshot
    private Arena snapshotArena;
    private int snapshotLen = -1;

    private CachedSession(
            ContextModel<?, ?, S> model,
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
     * One commit site for every write path; a sessions-only or read-only session records only the
     * fingerprint stream - the state itself is the cache.
     */
    private void commitSpan(int off, int len) {
        if (commits) tip = cache.commit(tip, fp, off, len, state);
    }

    /**
     * A session over NO tree: the fingerprint stream still tracks every ingested position (session
     * matching needs it), but nothing is committed and nothing can be restored.
     */
    static <S extends ContextState> CachedSession<S> fresh(ContextModel<?, ?, S> model, S state) {
        // a non-zero position would seed a stream of zero fingerprints for content the state
        // actually holds - a false prefix match away from serving wrong bytes. Refuse.
        if (state.position() != 0) {
            throw new IllegalArgumentException(
                    "fresh session needs a wiped state (position " + state.position() + ")");
        }
        return new CachedSession<>(model, state, null, false, null, new long[256], 0);
    }

    /** A fresh session on a fresh state for a brand-new conversation (nothing to resume). */
    public static <S extends ContextState> CachedSession<S> start(
            ContextModel<?, ?, S> model, BlockTree<S> cache, S state) {
        return resume(model, cache, state, new long[0], 0, true);
    }

    /**
     * A fresh session on a fresh state, resuming the longest cached prefix of {@code prompt} - the
     * encoded batches themselves are the key; their content addressing (token ids as themselves,
     * media rows by content digest) is this package's internal law.
     */
    public static <S extends ContextState> CachedSession<S> resume(
            ContextModel<?, ?, S> model, BlockTree<S> cache, S state, List<Batch> prompt) {
        long[] fp = fingerprints(prompt);
        return resume(model, cache, state, fp, fp.length, true);
    }

    /**
     * As {@link #resume(ContextModel, BlockTree, ContextState, List)} but restoring at most {@code
     * maxPositions} - e.g. the prompt length minus its final block, so a whole-prompt hit still
     * re-ingests that block and holds fresh logits at the frontier.
     */
    public static <S extends ContextState> CachedSession<S> resume(
            ContextModel<?, ?, S> model,
            BlockTree<S> cache,
            S state,
            List<Batch> prompt,
            int maxPositions) {
        return resume(model, cache, state, fingerprints(prompt), maxPositions, true);
    }

    /**
     * The one attach point: resumes the longest cached prefix of {@code expected[0..maxPositions)}
     * into the fresh state. {@code commits=false} is READ-ONLY serving - the longest cached prefix
     * restores, and no ingestion ever writes back (define-only mode: fixed overhead per served
     * block would grow the store by ~MBs per request).
     */
    static <S extends ContextState> CachedSession<S> resume(
            ContextModel<?, ?, S> model,
            BlockTree<S> cache,
            S state,
            long[] expected,
            int maxPositions,
            boolean commits) {
        BlockTree<S>.Block tip =
                cache.resume(expected, Math.min(expected.length, maxPositions), state);
        long[] fp = Arrays.copyOf(expected, Math.max(256, expected.length));
        return new CachedSession<>(model, state, cache, commits, tip, fp, tip.to);
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
        int total = Batch.positions(batches);
        long[] fp = new long[total];
        int at = 0;
        for (Batch b : batches) {
            switch (b.input()) {
                case Batch.Input.Tokens t -> {
                    for (int id : t.ids()) fp[at++] = id;
                }
                case Batch.Input.Embeddings e -> {
                    // Stable source identity when supplied; encoded rows otherwise (exact, but
                    // early JIT passes can drift - see Batch.embeddings' contentKey doc).
                    var key = e.contentKey();
                    long[] digest =
                            key == null
                                    ? rowsDigest(e)
                                    : Sha256.longs(
                                            Sha256.sha256().digest(key.value().getBytes(UTF_8)));
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
        ingest(batches, null);
    }

    /**
     * As {@link #ingest(List)} with the request's precomputed fingerprint stream, indexed by
     * absolute stream position - so a serve that already ran {@code fingerprints} does not re-hash
     * media rows (MBs per image). Either way the appended stream is byte-for-byte what {@code
     * fingerprints} defines: the ONE fingerprint law.
     */
    void ingest(List<Batch> batches, long[] expected) {
        ingest(batches, expected, () -> false);
    }

    /**
     * As above, with a cooperative interrupt consulted BEFORE each chunk: when it first reports
     * true, ingestion stops at the last completed chunk (a forward in flight always completes) and
     * false is returned. The appended stream stays position-exact - interrupted chunks are simply
     * absent, never half-recorded.
     */
    boolean ingest(List<Batch> batches, long[] expected, BooleanSupplier interrupt) {
        for (Batch b : Batch.prepare(batches, state.batchCapacity())) {
            if (interrupt.getAsBoolean()) {
                return false; // stop at the last completed chunk
            }
            int off = len;
            long[] f = expected != null ? null : fingerprints(List.of(b));
            model.ingest(state, b);
            int n = b.count();
            for (int i = 0; i < n; i++) append(expected != null ? expected[off + i] : f[i]);
            commitSpan(off, len - off);
        }
        return true;
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
        ingestGroups(groups, null);
    }

    /** As {@link #ingestGroups(List)} with the precomputed fingerprint stream (see ingest). */
    void ingestGroups(List<List<Batch>> groups, long[] expected) {
        ingestGroups(groups, expected, () -> false);
    }

    /** As above, stopping at the last completed chunk when the interrupt fires (see ingest). */
    boolean ingestGroups(List<List<Batch>> groups, long[] expected, BooleanSupplier interrupt) {
        int restored = state.position();
        int pos = 0;
        for (List<Batch> group : groups) {
            int end = pos + Batch.positions(group);
            if (end <= restored) { // fully restored: skip
                pos = end;
                continue;
            }
            if (!ingest(
                    pos >= restored ? group : tail(group, restored - pos), expected, interrupt)) {
                return false;
            }
            pos = end;
        }
        return true;
    }

    /**
     * The batch list minus its first {@code skip} positions: whole batches drop, a token batch at
     * the seam is sliced. A seam strictly inside a media batch cannot happen when {@code skip} came
     * from a block-aligned resume (media groups commit and restore whole) - it throws loudly.
     */
    private static List<Batch> tail(List<Batch> group, int skip) {
        List<Batch> out = new ArrayList<>();
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
     * SHA-256 of the raw row bits, as 4 longs - the media block's content identity. The rows are
     * FP32 dense row-major ({@link Batch.Input.Embeddings} enforces it), so the raw backing bytes
     * ARE the content stream; hashed straight off the segment in 1 MiB chunks. All jinfer media
     * rows are Panama-backed (even heap arrays materialize native) - anything else fails loudly
     * rather than silently keying on a fallback encoding.
     */
    private static long[] rowsDigest(Batch.Input.Embeddings e) {
        MemoryView<?> rows = e.rows();
        if (!(rows.memory().base() instanceof MemorySegment base)) {
            throw new UnsupportedOperationException(
                    "cannot fingerprint non-Panama embedding rows: "
                            + rows.memory().base().getClass().getName());
        }
        MessageDigest sha = Sha256.sha256();
        long bytes = rows.shape().size() * Float.BYTES;
        long start = rows.byteOffset();
        for (long off = 0; off < bytes; ) {
            int n = (int) Math.min(1 << 20, bytes - off);
            sha.update(base.asSlice(start + off, n).asByteBuffer());
            off += n;
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
    public void adopt(IntSequence ingested) {
        int n = ingested.length();
        if (n == 0) return;
        int off = len;
        for (int i = 0; i < n; i++) append(ingested.intAt(i));
        commitSpan(off, len - off);
    }

    /** As {@link #adopt(IntSequence)} over a plain array (zero-copy wrap). */
    public void adopt(int[] ingested) {
        adopt(IntSequence.wrap(ingested));
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
     * True when this session's WHOLE stream is a strict prefix of {@code req} - the append-only
     * reuse test (the facade's retained layer): the live state can continue with the remainder,
     * nothing to rewind, and at least one position is left to ingest.
     */
    boolean streamIsStrictPrefixOf(long[] req) {
        return len < req.length && Arrays.equals(fp, 0, len, req, 0, len);
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

    private void append(long fingerprint) {
        if (len == fp.length) fp = Arrays.copyOf(fp, fp.length * 2);
        fp[len++] = fingerprint;
    }

    // ---- tail snapshot (define-only mode; see the field comment) -----------------------------

    /**
     * Captures the residue at the CURRENT frontier - call at the prompt boundary, before decode.
     * Replaces any previous snapshot (one per session: the tail is the only sound rewind point; see
     * {@code save}'s contract - the residue only exists at the position the state is at).
     */
    void snapshotTail(CheckpointCodec<S> codec) {
        dropSnapshot();
        long bytes = codec.byteSize(0); // an empty span = the endpoint snapshot alone
        Arena arena = Arenas.newCrossThread();
        MemorySegment blob = arena.allocate(Math.max(bytes, 1), 8);
        codec.capture(state, state.position(), state.position(), blob);
        this.snapshotArena = arena;
        this.snapshot = blob;
        this.snapshotLen = state.position();
    }

    /** Bytes held by the tail snapshot (0 when none) - the facade's memory accounting. */
    long snapshotBytes() {
        return snapshot == null ? 0 : snapshot.byteSize();
    }

    /**
     * True when the snapshot exists and its stream strictly prefixes {@code req} - the rewind
     * eligibility test (at least one position left to ingest, so logits are fresh).
     */
    boolean snapshotIsStrictPrefixOf(long[] req) {
        return snapshotLen > 0
                && snapshotLen < req.length
                && Arrays.equals(fp, 0, snapshotLen, req, 0, snapshotLen);
    }

    /**
     * Rewinds the state to the snapshot boundary: residue restored, position moved, the stream
     * truncated to the boundary (the reply's fingerprints drop with its recurrent influence). Rows
     * past the boundary become dead; the caller re-ingests the request's tail over them.
     */
    void rewindToSnapshot(CheckpointCodec<S> codec) {
        codec.restore(state, snapshotLen, snapshotLen, snapshot);
        state.resumeAt(snapshotLen);
        len = snapshotLen;
    }

    /** Frees the snapshot blob; idempotent. Every path that ends the session must call this. */
    void dropSnapshot() {
        if (snapshotArena != null) Arenas.close(snapshotArena);
        snapshotArena = null;
        snapshot = null;
        snapshotLen = -1;
    }
}
