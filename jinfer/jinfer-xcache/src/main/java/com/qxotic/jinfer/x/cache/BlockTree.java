package com.qxotic.jinfer.x.cache;

import com.qxotic.jinfer.x.boundary.ContentKey;
import com.qxotic.jinfer.x.boundary.RuntimeState;
import com.qxotic.jinfer.x.boundary.StateCodec;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The block layer's physics - the low-level tree under {@link PromptCache}: a prefix tree of
 * variable-length KV blocks, content-addressed by a CHAINED SHA-256 key over per-position
 * fingerprints (token id for text, content-hash for media). The chained digest names the whole
 * token prefix — same conversation prefix, same key, across sessions and restarts — and, being
 * cryptographic, is trusted as identity (matching recomputes the digest from the request's
 * fingerprints; no fingerprint storage, no collision handling: git/IPFS regime).
 *
 * <p>Every block is SELF-CONTAINED ({@link StateCodec}): restoring a matched chain in position
 * order leaves the state live at the chain's end, so EVERY block boundary is a resume point —
 * blocks match completely or not at all, and the longest matching chain is the resume. The cache is
 * a pure optimization — every miss degrades to recompute, never to a wrong answer.
 *
 * <p>{@link #resume} matches and restores once per request; {@link CachedSession} is the write
 * handle, committing each subsequently ingested span in O(span) — one digest of the span against
 * the tip key, no re-walk — which is what makes single-token commits during decode natural. Large
 * blocks and single-token blocks are the same mechanism at different spans.
 *
 * <p>The model contributes only a {@link StateCodec}; storage only a {@link CacheStore}; identity
 * only a {@link ContentKey} seed. This class is pure policy: keys, the prefix tree, matching,
 * budget/LRU-leaf eviction. Single-threaded by design (the generation worker), like the store.
 */
public final class BlockTree<S extends RuntimeState> {

    /** 256-bit chained content address. */
    public record BlockKey(long a, long b, long c, long d) {}

    private static final BlockKey ROOT = new BlockKey(0, 0, 0, 0);

    final class Block {
        final BlockKey key;
        final Block parent; // sentinel for depth-0 blocks
        final int from, to;
        final MemorySegment mem; // null only on the sentinel
        final List<Block> children = new ArrayList<>(2);
        long lastUsed;
        boolean live = true;
        boolean frozen; // grafted from a FrozenBlocks artifact: never evicted, never freed
        int frozenCrc; // artifact-carried CRC32C, verified lazily on first restore
        boolean frozenVerified;

        Block(BlockKey key, Block parent, int from, int to, MemorySegment mem) {
            this.key = key;
            this.parent = parent;
            this.from = from;
            this.to = to;
            this.mem = mem;
        }
    }

    private final StateCodec<S> codec;
    private final CacheStore store;
    private final long budgetBytes;
    private final ContentKey modelSeed;
    private final Map<BlockKey, Block> blocks = new HashMap<>();
    private final Set<Block> leaves = new HashSet<>();
    private final List<Block> freshBlocks = new ArrayList<>(); // committed here, not yet on disk
    private FrozenBlocks base; // the grafted artifact, target of appendTo; null when standalone
    private final Block sentinel;
    private final Block detached;
    private final List<Block> chainScratch = new ArrayList<>(); // resume()'s chain walk, reused
    private final MessageDigest sha = Sha256.sha256();
    private ByteBuffer scratch = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN);
    private long clock;
    private long hits, misses, evictions, discards, refusals;

    /**
     * A writable cache layered over a read-only {@link FrozenBlocks} base: the artifact's blocks
     * graft into this cache's key space (resume matches through them, commits dedup against them)
     * but never count against the budget and are never evicted or freed. The artifact must have
     * been built with the SAME model seed - chained keys only line up when the roots agree. A null
     * {@code base} is a plain standalone cache.
     */
    public BlockTree(
            StateCodec<S> codec,
            CacheStore store,
            long budgetBytes,
            ContentKey modelSeed,
            FrozenBlocks base) {
        this(codec, store, budgetBytes, modelSeed);
        if (base == null) return;
        this.base = base;
        for (FrozenBlocks.Entry e : base.entries()) { // BFS: parents precede children
            Block parent =
                    e.parentKey().equals(sentinel.key) ? sentinel : blocks.get(e.parentKey());
            if (parent == null) {
                throw new IllegalStateException(
                        base
                                + " is not rooted in this cache's model seed (unknown parent for "
                                + "block ["
                                + e.from()
                                + ","
                                + e.to()
                                + "))");
            }
            Block b = new Block(e.key(), parent, e.from(), e.to(), e.mem());
            b.frozen = true;
            b.frozenCrc = e.crc();
            blocks.put(e.key(), b);
            parent.children.add(b);
            // deliberately NOT a leaf: eviction never sees frozen blocks
        }
    }

    /**
     * {@code modelSeed} folds the model's identity (and implicitly the codec's blob layout) into
     * the ROOT of the key chain, so two models — even sharing a tokenizer, where fingerprint
     * streams collide — can never match each other's blocks. One cache instance per model is the
     * deployment shape; the seed is produced model-load-side (xchat {@code Models.modelSeed}).
     */
    public BlockTree(
            StateCodec<S> codec, CacheStore store, long budgetBytes, ContentKey modelSeed) {
        this.codec = codec;
        this.store = store;
        this.budgetBytes = budgetBytes;
        this.modelSeed = modelSeed;
        BlockKey root = ROOT;
        byte[] seedBytes = modelSeed.digestBytes();
        if (seedBytes.length > 0) {
            sha.reset();
            sha.update(seedBytes);
            long[] d = Sha256.digestLongs(sha);
            root = new BlockKey(d[0], d[1], d[2], d[3]);
        }
        this.sentinel = new Block(root, null, 0, 0, null);
        this.detached = new Block(root, null, 0, 0, null);
        this.detached.live = false;
    }

    /**
     * Matches the longest cached complete-block prefix of {@code fingerprints[0..len)}, restores it
     * into {@code state}, and returns the matched tip — the session's resume point; the caller
     * re-ingests everything past {@code tip.to}. Returns the sentinel (position 0) on a cold start.
     */
    Block resume(long[] fingerprints, int len, S state) {
        Block tip = sentinel;
        while (true) {
            Block next = null;
            for (Block b : tip.children) { // few children; longest matching span wins
                if (b.to <= len
                        && (next == null || b.to > next.to)
                        && digest(tip.key, fingerprints, b.from, b.to - b.from).equals(b.key)) {
                    next = b;
                }
            }
            if (next == null) break;
            touch(next);
            tip = next;
        }
        if (tip != sentinel) {
            for (Block b = tip; b != sentinel; b = b.parent) chainScratch.add(b);
            for (int i = chainScratch.size() - 1; i >= 0; i--) {
                Block b = chainScratch.get(i);
                boolean valid =
                        !b.frozen
                                || b.frozenVerified
                                || (b.frozenVerified = FrozenBlocks.crc32c(b.mem) == b.frozenCrc);
                if (!valid) { // failed verification = a miss, never restored
                    discard(b); // the block and everything chained on it
                    chainScratch.clear();
                    return resume(
                            fingerprints, len, state); // the tree changed: re-match from scratch
                }
                codec.restoreCheckpoint(state, b.from, b.to, b.mem);
            }
            chainScratch.clear();
            hits++;
            state.resumeAt(tip.to);
        } else {
            misses++;
        }
        return tip;
    }

    /**
     * Commits the span {@code spanFp[off..off+len)} just ingested as one block chained on {@code
     * tip}, returning the new tip. Dedups against an existing identical block; detaches (returns a
     * dead tip whose commits no-op) when the budget refuses the block.
     */
    Block commit(Block tip, long[] spanFp, int off, int len, S state) {
        if (!tip.live || len == 0) return tip;
        if (state.position() != tip.to + len) {
            throw new IllegalStateException(
                    "commit of "
                            + len
                            + " at chain position "
                            + tip.to
                            + " but state is at "
                            + state.position());
        }
        BlockKey key = digest(tip.key, spanFp, off, len);
        Block existing = blocks.get(key);
        if (existing != null) { // dedup: same prefix, same span
            touch(existing);
            return existing;
        }
        int to = tip.to + len;
        long bytes = codec.checkpointBytes(len);
        if (!ensureBudget(bytes, tip)) { // budget refused: detach softly
            refusals++;
            return detached;
        }
        MemorySegment mem = store.allocate(bytes);
        codec.saveCheckpoint(state, tip.to, to, mem);
        Block block = new Block(key, tip, tip.to, to, mem);
        blocks.put(key, block);
        freshBlocks.add(block); // creation order is parents-first: a tip precedes its children
        leaves.remove(tip);
        leaves.add(block);
        tip.children.add(block);
        touch(block);
        return block;
    }

    /**
     * Removes a block and its whole subtree from the tree and frees their blobs — used when a
     * stored blob fails verification: the cache degrades to a miss, never to a wrong answer.
     */
    private void discard(Block b) {
        for (Block child : List.copyOf(b.children)) discard(child);
        b.children.clear();
        unlink(b);
        discards++;
    }

    /** Detach one block from the tree and free its blob; leaf-promotes a live parent. */
    private void unlink(Block b) {
        b.live = false;
        blocks.remove(b.key);
        leaves.remove(b);
        freshBlocks.remove(b); // its blob is about to be freed; never write it
        b.parent.children.remove(b);
        if (b.parent != sentinel
                && b.parent.live
                && b.parent.children.isEmpty()
                && !b.parent.frozen) {
            leaves.add(b.parent); // dead parents stay out: their blob is freed
        }
        if (!b.frozen) store.free(b.mem); // frozen blobs are mmap slices, not store allocations
    }

    private void touch(Block b) {
        b.lastUsed = ++clock;
    }

    /**
     * Evicts LRU leaves until {@code needed} fits the budget. Leaf-only eviction keeps every
     * remaining chain contiguous. {@code keep} (the committing session's tip) is never evicted: a
     * chain's only leaf is its tip, so without the guard a commit under pressure would evict its
     * own chain, then link the new block under the freed corpse (double-free on the next eviction
     * pass). When only {@code keep} remains, the commit detaches instead.
     */
    private boolean ensureBudget(long needed, Block keep) {
        if (needed > budgetBytes) return false;
        while (store.usedBytes() + needed > budgetBytes) {
            Block lru = null;
            for (Block b : leaves) {
                if (b != keep && (lru == null || b.lastUsed < lru.lastUsed)) lru = b;
            }
            if (lru == null) return false;
            unlink(lru);
            evictions++;
        }
        return true;
    }

    private BlockKey digest(BlockKey parent, long[] fp, int off, int len) {
        int bytes = 32 + len * Long.BYTES;
        if (scratch.capacity() < bytes)
            scratch = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        scratch.clear();
        scratch.putLong(parent.a()).putLong(parent.b()).putLong(parent.c()).putLong(parent.d());
        for (int i = 0; i < len; i++) scratch.putLong(fp[off + i]);
        sha.reset();
        sha.update(scratch.array(), 0, bytes);
        long[] d = Sha256.digestLongs(sha);
        return new BlockKey(d[0], d[1], d[2], d[3]);
    }

    /**
     * Serializes every block reachable from the root into {@code out} - a read-only, shareable
     * {@link FrozenBlocks} artifact holding any number of prompts, shared prefixes stored once. A
     * cache layered over a frozen base re-freezes base and growth into one merged artifact.
     */
    public void freeze(Path out) throws IOException {
        if (base != null && Files.exists(out) && Files.isSameFile(base.file(), out)) {
            // frozen blocks' mem are slices of that very file's mapping: TRUNCATE_EXISTING would
            // pull the bytes out from under the copy loop (SIGBUS / corrupt artifact)
            throw new IllegalStateException(
                    "freeze(" + out + ") would rewrite the mounted artifact; use appendTo");
        }
        List<Block> order = new ArrayList<>(blocks.size());
        ArrayDeque<Block> queue = new ArrayDeque<>();
        queue.add(sentinel);
        while (!queue.isEmpty()) { // BFS: parents before children
            Block b = queue.poll();
            if (b != sentinel) order.add(b);
            queue.addAll(b.children);
        }
        long[] offsets = new long[order.size()];
        long off = FrozenBlocks.HEADER_BYTES;
        for (int i = 0; i < order.size(); i++) {
            offsets[i] = off;
            off = FrozenBlocks.align(off + order.get(i).mem.byteSize());
        }
        long indexOffset = off;
        long total = indexOffset + (long) order.size() * FrozenBlocks.INDEX_ENTRY_BYTES;
        try (FileChannel ch =
                        FileChannel.open(
                                out,
                                StandardOpenOption.CREATE,
                                StandardOpenOption.TRUNCATE_EXISTING,
                                StandardOpenOption.READ,
                                StandardOpenOption.WRITE);
                Arena arena = Arena.ofConfined()) {
            MemorySegment map = ch.map(FileChannel.MapMode.READ_WRITE, 0, total, arena);
            ByteBuffer h =
                    map.asSlice(0, FrozenBlocks.HEADER_BYTES)
                            .asByteBuffer()
                            .order(ByteOrder.LITTLE_ENDIAN);
            h.putInt(FrozenBlocks.MAGIC)
                    .putInt(FrozenBlocks.FORMAT_VERSION)
                    .put(modelSeed.digestBytes())
                    .putInt(order.size())
                    .putLong(indexOffset);
            ByteBuffer idx =
                    map.asSlice(indexOffset, total - indexOffset)
                            .asByteBuffer()
                            .order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < order.size(); i++) {
                Block b = order.get(i);
                MemorySegment.copy(b.mem, 0, map, offsets[i], b.mem.byteSize());
                FrozenBlocks.putEntry(
                        idx,
                        b.key,
                        b.parent.key,
                        b.from,
                        b.to,
                        offsets[i],
                        b.mem.byteSize(),
                        FrozenBlocks.crc32c(map.asSlice(offsets[i], b.mem.byteSize())));
            }
            map.force();
        }
    }

    /**
     * Appends this cache's blocks committed since the last append to {@code out} - see {@link
     * FrozenBlocks#append} for the on-disk protocol and its cost. Creates the file (plain {@link
     * #freeze}) when it does not exist; otherwise the cache must be layered over the artifact at
     * {@code out}, so keys line up and grafted blocks are already on disk. Partial state (an
     * uncommitted span, the live session tail) never touches disk - blocks only exist complete.
     */
    public void appendTo(Path out) throws IOException {
        if (!Files.exists(out)) {
            // recovery path (the catalog was deleted mid-run): the whole tree re-freezes - and
            // the parsed base view must follow, or the NEXT append would compare offsets against
            // the old file's layout and refuse with a misleading "another writer" error
            freeze(out);
            freshBlocks.clear();
            base = FrozenBlocks.open(out, modelSeed);
            return;
        }
        if (base == null || !Files.isSameFile(base.file(), out)) {
            throw new IllegalStateException(
                    "appendTo(" + out + ") needs this cache layered over that artifact");
        }
        List<FrozenBlocks.Entry> fresh = new ArrayList<>(freshBlocks.size());
        for (Block b : freshBlocks) {
            fresh.add(
                    new FrozenBlocks.Entry(
                            b.key,
                            b.parent.key,
                            b.from,
                            b.to,
                            -1,
                            b.mem,
                            FrozenBlocks.crc32c(b.mem)));
        }
        base.append(fresh);
        freshBlocks.clear();
    }

    /** A cache health reading: sizes now, counters cumulative since construction. */
    public record Sample(
            int blocks,
            long bytes,
            long budgetBytes,
            long hits,
            long misses,
            long evictions,
            long discards,
            long refusals) {

        /** The hot-only reading: no block layer, every field zero. */
        public static final Sample ZERO = new Sample(0, 0, 0, 0, 0, 0, 0, 0);
    }

    /** The one source of the cache's numbers; {@link #stats} is its formatting. */
    public Sample sample() {
        return new Sample(
                blocks.size(),
                store.usedBytes(),
                budgetBytes,
                hits,
                misses,
                evictions,
                discards,
                refusals);
    }

    public String stats() {
        Sample s = sample();
        return "blocks="
                + s.blocks()
                + " bytes="
                + s.bytes()
                + " hits="
                + s.hits()
                + " misses="
                + s.misses()
                + " evictions="
                + s.evictions()
                + " discards="
                + s.discards()
                + " refusals="
                + s.refusals();
    }
}
