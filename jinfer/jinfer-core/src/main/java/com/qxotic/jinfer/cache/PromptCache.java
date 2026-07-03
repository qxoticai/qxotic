package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.RuntimeState;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Model-agnostic, storage-agnostic prompt cache: a prefix tree of variable-length KV blocks,
 *  content-addressed by a CHAINED SHA-256 key over per-position fingerprints (token id for text,
 *  content-hash for media). The chained digest names the whole token prefix — same conversation
 *  prefix, same key, across sessions and restarts — and, being cryptographic, is trusted as
 *  identity (matching recomputes the digest from the request's fingerprints; no fingerprint
 *  storage, no collision handling: git/IPFS regime).
 *
 *  <p>Blocks match COMPLETELY or not at all: recurrent checkpoints exist only at block boundaries
 *  (see {@link KvCodec}), so the longest reusable prefix is a chain of whole blocks. The cache is
 *  a pure optimization — every miss degrades to recompute, never to a wrong answer.
 *
 *  <p>{@link #resume} matches and restores once per request and returns a {@link Cursor}; the
 *  cursor commits each subsequently ingested span in O(span) — one digest of the span against the
 *  tip key, no re-walk — which is what makes single-token commits during decode natural. Large
 *  blocks and single-token blocks are the same mechanism at different spans.
 *
 *  <p>The model contributes only a {@link KvCodec}; storage only a {@link CacheStore}. This class
 *  is pure policy: keys, the prefix tree, matching, budget/LRU-leaf eviction. Single-threaded by
 *  design (the generation worker), like the store. */
public final class PromptCache<S extends RuntimeState> {

    /** 256-bit chained content address. */
    public record BlockKey(long a, long b, long c, long d) {}

    private static final BlockKey ROOT = new BlockKey(0, 0, 0, 0);

    private final class Block {
        final BlockKey key;
        final Block parent;                    // sentinel for depth-0 blocks
        final int from, to;
        final MemorySegment mem;               // null only on the sentinel
        final List<Block> children = new ArrayList<>(2);
        long lastUsed;
        boolean live = true;

        Block(BlockKey key, Block parent, int from, int to, MemorySegment mem) {
            this.key = key;
            this.parent = parent;
            this.from = from;
            this.to = to;
            this.mem = mem;
        }
    }

    /** A session's write handle: the tip of its committed chain. {@link #commit} extends the chain
     *  by exactly the span just ingested (the cache enforces {@code state.position()} agreement).
     *  Detaches — commits become no-ops — if the budget rejects a block or the tip is evicted;
     *  reads (resume) are unaffected. */
    public final class Cursor {
        private Block tip;

        private Cursor(Block tip) {
            this.tip = tip;
        }

        /** Positions covered by the committed chain. */
        public int position() {
            return tip.to;
        }

        /** Commit one decode step (a single-token block). */
        public void commit(long fingerprint, S state) {
            commit(new long[]{fingerprint}, 0, 1, state);
        }

        /** Commit the span {@code spanFp[off..off+len)} just ingested. */
        public void commit(long[] spanFp, int off, int len, S state) {
            if (!tip.live || len == 0) return;
            if (state.position() != tip.to + len) {
                throw new IllegalStateException("commit of " + len + " at chain position " + tip.to
                        + " but state is at " + state.position());
            }
            BlockKey key = digest(tip.key, spanFp, off, len);
            Block existing = blocks.get(key);
            if (existing != null) {                        // dedup: same prefix, same span
                touch(existing);
                tip = existing;
                return;
            }
            long bytes = codec.bytes(len);
            if (!ensureBudget(bytes)) {                    // budget refused: detach softly
                tip = DETACHED;
                return;
            }
            MemorySegment mem = store.allocate(bytes);
            codec.save(state, tip.to, tip.to + len, mem);
            store.validate(mem);
            Block block = new Block(key, tip, tip.to, tip.to + len, mem);
            blocks.put(key, block);
            leaves.remove(tip);
            leaves.add(block);
            tip.children.add(block);
            touch(block);
            tip = block;
        }
    }

    private final KvCodec<S> codec;
    private final CacheStore store;
    private final long budgetBytes;
    private final Map<BlockKey, Block> blocks = new HashMap<>();
    private final Set<Block> leaves = new HashSet<>();
    private final Block sentinel = new Block(ROOT, null, 0, 0, null);
    private final Block DETACHED = new Block(ROOT, null, 0, 0, null);
    private final MessageDigest sha;
    private ByteBuffer scratch = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN);
    private long clock;
    private long hits, misses, evictions;

    public PromptCache(KvCodec<S> codec, CacheStore store, long budgetBytes) {
        this.codec = codec;
        this.store = store;
        this.budgetBytes = budgetBytes;
        this.DETACHED.live = false;
        try {
            this.sha = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /** Matches the longest cached complete-block prefix of {@code fingerprints[0..len)}, restores
     *  it into {@code state} (which resumes at the matched position), and returns the session's
     *  write cursor sitting at the match — 0 on a cold start. */
    public Cursor resume(long[] fingerprints, int len, S state) {
        Block tip = sentinel;
        while (true) {
            Block next = null;
            for (Block b : tip.children) {                 // few children; longest matching span wins
                if (b.to <= len && (next == null || b.to > next.to)
                        && digest(tip.key, fingerprints, b.from, b.to - b.from).equals(b.key)) {
                    next = b;
                }
            }
            if (next == null) break;
            touch(next);
            tip = next;
        }
        if (tip != sentinel) {
            hits++;
            for (Block b = tip; b != sentinel; b = b.parent) chainScratch.add(b);
            for (int i = chainScratch.size() - 1; i >= 0; i--) {
                Block b = chainScratch.get(i);
                store.validate(b.mem);
                codec.restore(state, b.from, b.to, b.mem);
            }
            chainScratch.clear();
            state.resumeAt(tip.to);
        } else {
            misses++;
        }
        return new Cursor(tip);
    }

    private final List<Block> chainScratch = new ArrayList<>();

    private void touch(Block b) {
        b.lastUsed = ++clock;
    }

    /** Evicts LRU leaves until {@code needed} fits the budget. Leaf-only eviction keeps every
     *  remaining chain contiguous. */
    private boolean ensureBudget(long needed) {
        if (needed > budgetBytes) return false;
        while (store.usedBytes() + needed > budgetBytes) {
            Block lru = null;
            for (Block b : leaves) {
                if (lru == null || b.lastUsed < lru.lastUsed) lru = b;
            }
            if (lru == null) return false;
            lru.live = false;
            blocks.remove(lru.key);
            leaves.remove(lru);
            lru.parent.children.remove(lru);
            if (lru.parent != sentinel && lru.parent.children.isEmpty()) leaves.add(lru.parent);
            store.free(lru.mem);
            evictions++;
        }
        return true;
    }

    private BlockKey digest(BlockKey parent, long[] fp, int off, int len) {
        int bytes = 32 + len * Long.BYTES;
        if (scratch.capacity() < bytes) scratch = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        scratch.clear();
        scratch.putLong(parent.a()).putLong(parent.b()).putLong(parent.c()).putLong(parent.d());
        for (int i = 0; i < len; i++) scratch.putLong(fp[off + i]);
        sha.reset();
        sha.update(scratch.array(), 0, bytes);
        ByteBuffer out = ByteBuffer.wrap(sha.digest()).order(ByteOrder.LITTLE_ENDIAN);
        return new BlockKey(out.getLong(), out.getLong(), out.getLong(), out.getLong());
    }

    public String stats() {
        return "blocks=" + blocks.size() + " bytes=" + store.usedBytes()
                + " hits=" + hits + " misses=" + misses + " evictions=" + evictions;
    }
}
