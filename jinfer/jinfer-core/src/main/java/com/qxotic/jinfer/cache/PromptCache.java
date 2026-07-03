package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.CacheStore;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Model-agnostic, storage-agnostic prompt cache: a prefix tree of variable-length KV blocks,
 *  content-addressed by a CHAINED SHA-256 key over per-position fingerprints (token id for text,
 *  content-hash for media). The chained digest names the whole token prefix — same conversation
 *  prefix, same key, across sessions and restarts — and, being cryptographic, is trusted as
 *  identity (a match is verified by recomputing the digest from the request's fingerprints; no
 *  fingerprint storage, no collision handling: git/IPFS regime).
 *
 *  <p>Blocks match COMPLETELY or not at all: recurrent checkpoints exist only at block boundaries
 *  (see {@link KvCodec}), so the longest reusable prefix is a chain of whole blocks. Divergence
 *  mid-block costs at most that block's span of re-prefill — the cache is a pure optimization and
 *  every miss degrades to recompute, never to a wrong answer.
 *
 *  <p>The model contributes only a {@link KvCodec}; storage only a {@link CacheStore}. This class
 *  is pure policy: keys, the prefix tree, matching, budget/LRU-leaf eviction. Single-threaded by
 *  design (the generation worker), like the store. */
public final class PromptCache<S> {

    /** 256-bit chained content address. */
    public record BlockKey(long a, long b, long c, long d) {}

    private static final BlockKey ROOT = new BlockKey(0, 0, 0, 0);

    private final class Block {
        final BlockKey key;
        final Block parent;                    // null for roots (children of the empty prefix)
        final int from, to;
        final MemorySegment mem;
        final List<Block> children = new ArrayList<>(2);
        long lastUsed;

        Block(BlockKey key, Block parent, int from, int to, MemorySegment mem) {
            this.key = key;
            this.parent = parent;
            this.from = from;
            this.to = to;
            this.mem = mem;
        }
    }

    private final KvCodec<S> codec;
    private final CacheStore store;
    private final long budgetBytes;
    private final Map<BlockKey, Block> blocks = new HashMap<>();
    private final List<Block> roots = new ArrayList<>();
    private final MessageDigest sha;
    private long clock;
    private long hits, misses, evictions;

    public PromptCache(KvCodec<S> codec, CacheStore store, long budgetBytes) {
        this.codec = codec;
        this.store = store;
        this.budgetBytes = budgetBytes;
        try {
            this.sha = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /** Restores the longest cached complete-block prefix of {@code fingerprints} into
     *  {@code state} and returns the number of positions resumed (0 = cold). The caller ingests
     *  from there. */
    public int restore(long[] fingerprints, S state) {
        List<Block> chain = match(fingerprints, fingerprints.length);
        for (Block b : chain) {
            store.validate(b.mem);
            codec.restore(state, b.from, b.to, b.mem);
        }
        int positions = chain.isEmpty() ? 0 : chain.get(chain.size() - 1).to;
        if (positions > 0) hits++; else misses++;
        return positions;
    }

    /** Commits the span {@code [from,to)} — just ingested, so {@code state.position() == to} — as
     *  one block chained onto the cached prefix {@code [0,from)}. That prefix must itself be fully
     *  cached (commit at every ingestion boundary and it always is); returns false when it isn't
     *  or when the block cannot fit the budget. Committing an already-cached span is a cheap
     *  dedup no-op. */
    public boolean commit(long[] fingerprints, int from, int to, S state) {
        if (from == to) return true;
        List<Block> chain = match(fingerprints, from);
        int covered = chain.isEmpty() ? 0 : chain.get(chain.size() - 1).to;
        if (covered != from) return false;                        // prefix gap: unreachable block
        Block parent = chain.isEmpty() ? null : chain.get(chain.size() - 1);
        BlockKey key = digest(parent == null ? ROOT : parent.key, fingerprints, from, to);
        Block existing = blocks.get(key);
        if (existing != null) {
            existing.lastUsed = ++clock;                          // dedup: same prefix, same span
            return true;
        }
        long bytes = codec.bytes(from, to);
        if (!ensureBudget(bytes)) return false;
        MemorySegment mem = store.allocate(bytes);
        codec.save(state, from, to, mem);
        store.validate(mem);
        Block block = new Block(key, parent, from, to, mem);
        blocks.put(key, block);
        (parent == null ? roots : parent.children).add(block);
        return true;
    }

    /** Longest chain of whole blocks matching {@code fingerprints[0..limit)}: at each depth, the
     *  candidate child whose recomputed digest equals its key (longest span first). */
    private List<Block> match(long[] fp, int limit) {
        List<Block> chain = new ArrayList<>();
        BlockKey parentKey = ROOT;
        List<Block> candidates = roots;
        while (true) {
            Block next = null;
            for (Block b : candidates) {                          // few children; longest wins
                if (b.to <= limit && (next == null || b.to > next.to)
                        && digest(parentKey, fp, b.from, b.to).equals(b.key)) {
                    next = b;
                }
            }
            if (next == null) return chain;
            next.lastUsed = ++clock;
            chain.add(next);
            parentKey = next.key;
            candidates = next.children;
        }
    }

    /** Evicts LRU leaves until {@code needed} fits the budget. Leaf-only eviction keeps every
     *  remaining chain contiguous (a block never outlives its parent's reachability). */
    private boolean ensureBudget(long needed) {
        if (needed > budgetBytes) return false;
        while (store.usedBytes() + needed > budgetBytes) {
            Block lru = null;
            for (Block b : blocks.values()) {
                if (b.children.isEmpty() && (lru == null || b.lastUsed < lru.lastUsed)) lru = b;
            }
            if (lru == null) return false;
            blocks.remove(lru.key);
            (lru.parent == null ? roots : lru.parent.children).remove(lru);
            store.free(lru.mem);
            evictions++;
        }
        return true;
    }

    private BlockKey digest(BlockKey parent, long[] fp, int from, int to) {
        sha.reset();
        ByteBuffer buf = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(parent.a()).putLong(parent.b()).putLong(parent.c()).putLong(parent.d());
        sha.update(buf.array());
        ByteBuffer span = ByteBuffer.allocate((to - from) * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = from; i < to; i++) span.putLong(fp[i]);
        sha.update(span.array());
        ByteBuffer out = ByteBuffer.wrap(sha.digest()).order(ByteOrder.LITTLE_ENDIAN);
        return new BlockKey(out.getLong(), out.getLong(), out.getLong(), out.getLong());
    }

    public String stats() {
        return "blocks=" + blocks.size() + " bytes=" + store.usedBytes()
                + " hits=" + hits + " misses=" + misses + " evictions=" + evictions;
    }
}
