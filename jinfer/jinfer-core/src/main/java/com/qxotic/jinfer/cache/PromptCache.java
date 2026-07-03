package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.RuntimeState;

import java.lang.foreign.Arena;
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
            if (frozen || !tip.live || len == 0) return;
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
    private final byte[] modelSeed;
    private final boolean frozen;
    private final Map<BlockKey, Block> blocks = new HashMap<>();
    private final Set<Block> leaves = new HashSet<>();
    private final Block sentinel;
    private final Block DETACHED;
    private final MessageDigest sha;
    private ByteBuffer scratch = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN);
    private long clock;
    private long hits, misses, evictions;

    public PromptCache(KvCodec<S> codec, CacheStore store, long budgetBytes) {
        this(codec, store, budgetBytes, new byte[0]);
    }

    /** {@code modelSeed} folds the model's identity (and implicitly the codec's blob layout) into
     *  the ROOT of the key chain, so two models — even sharing a tokenizer, where fingerprint
     *  streams collide — can never match each other's blocks. A codec layout change ships with a
     *  seed change and auto-invalidates persisted blocks (miss → recompute, no migration). One
     *  cache instance/file per model is the deployment shape; see {@link #modelSeed}. */
    public PromptCache(KvCodec<S> codec, CacheStore store, long budgetBytes, byte[] modelSeed) {
        this(codec, store, budgetBytes, modelSeed, false);
    }

    private PromptCache(KvCodec<S> codec, CacheStore store, long budgetBytes, byte[] modelSeed, boolean frozen) {
        this.codec = codec;
        this.store = store;
        this.budgetBytes = budgetBytes;
        this.modelSeed = modelSeed.clone();
        this.frozen = frozen;
        try {
            this.sha = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
        BlockKey root = ROOT;
        if (modelSeed.length > 0) {
            sha.reset();
            sha.update(modelSeed);
            ByteBuffer out = ByteBuffer.wrap(sha.digest()).order(ByteOrder.LITTLE_ENDIAN);
            root = new BlockKey(out.getLong(), out.getLong(), out.getLong(), out.getLong());
        }
        this.sentinel = new Block(root, null, 0, 0, null);
        this.DETACHED = new Block(root, null, 0, 0, null);
        this.DETACHED.live = false;
    }

    /** A fast, stable model identity for {@link #PromptCache(KvCodec, CacheStore, long, byte[])}
     *  and for naming a persisted cache file: file length + SHA-256 of the first and last MiB of
     *  the GGUF (full-content hashing of multi-GB weights is not worth it — length + head/tail
     *  covers metadata, tensor table and data edges). */
    public static byte[] modelSeed(java.nio.file.Path gguf) {
        try (var ch = java.nio.channels.FileChannel.open(gguf, java.nio.file.StandardOpenOption.READ)) {
            MessageDigest d = MessageDigest.getInstance("SHA-256");
            long size = ch.size();
            ByteBuffer len = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(0, size);
            d.update(len);
            ByteBuffer buf = ByteBuffer.allocate((int) Math.min(1 << 20, size));
            ch.read(buf, 0);
            buf.flip();
            d.update(buf);
            if (size > (1 << 20)) {
                buf.clear();
                ch.read(buf, size - buf.capacity());
                buf.flip();
                d.update(buf);
            }
            return d.digest();
        } catch (java.io.IOException | NoSuchAlgorithmException e) {
            throw new IllegalStateException("modelSeed(" + gguf + ")", e);
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

    // ═══ freeze / open: the frozen multi-prompt cache file (use case B) ═══
    //   JKVF, formatVersion, modelSeed[32], blockCount, indexOffset | KV blobs (64-aligned) |
    //   index: per block {key[4], parentKey[4], from, to, byteOffset, byteLen} in BFS order
    //   (parents precede children, so the tree rebuilds in one pass).

    private static final int MAGIC_FROZEN = 0x46564B4A;    // "JKVF"
    private static final int FORMAT_VERSION = 1;
    private static final int HEADER_BYTES = 64;             // 4+4+32+4+8, padded
    private static final int INDEX_ENTRY_BYTES = 88;        // 32+32+4+4+8+8
    private static final int ALIGN = 64;

    /** Serializes every block into {@code out} — a read-only, shareable artifact holding any
     *  number of prompts, shared prefixes stored once (content addressing). Open it with
     *  {@link #open}. */
    public void freeze(java.nio.file.Path out) throws java.io.IOException {
        if (frozen) throw new IllegalStateException("already frozen");
        List<Block> order = new ArrayList<>(blocks.size());
        java.util.ArrayDeque<Block> queue = new java.util.ArrayDeque<>();
        queue.add(sentinel);
        while (!queue.isEmpty()) {                          // BFS: parents before children
            Block b = queue.poll();
            if (b != sentinel) order.add(b);
            queue.addAll(b.children);
        }
        long[] offsets = new long[order.size()];
        long off = HEADER_BYTES;
        for (int i = 0; i < order.size(); i++) {
            offsets[i] = off;
            off = align(off + order.get(i).mem.byteSize());
        }
        long indexOffset = off;
        long total = indexOffset + (long) order.size() * INDEX_ENTRY_BYTES;
        try (java.nio.channels.FileChannel ch = java.nio.channels.FileChannel.open(out,
                java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.TRUNCATE_EXISTING,
                java.nio.file.StandardOpenOption.READ, java.nio.file.StandardOpenOption.WRITE);
             Arena arena = Arena.ofConfined()) {
            MemorySegment map = ch.map(java.nio.channels.FileChannel.MapMode.READ_WRITE, 0, total, arena);
            ByteBuffer h = map.asSlice(0, HEADER_BYTES).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            h.putInt(MAGIC_FROZEN).putInt(FORMAT_VERSION).put(seed32(modelSeed)).putInt(order.size()).putLong(indexOffset);
            ByteBuffer idx = map.asSlice(indexOffset, total - indexOffset).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < order.size(); i++) {
                Block b = order.get(i);
                MemorySegment.copy(b.mem, 0, map, offsets[i], b.mem.byteSize());
                putKey(idx, b.key);
                putKey(idx, b.parent.key);
                idx.putInt(b.from).putInt(b.to).putLong(offsets[i]).putLong(b.mem.byteSize());
            }
            map.force();
        }
    }

    /** Maps a frozen cache file lazily (header + index read; KV bytes untouched until a restore)
     *  and validates it belongs to the model identified by {@code modelSeed} — throws a
     *  descriptive error when it does not. The returned cache is FROZEN: {@link #resume} serves
     *  every stored prompt; cursor commits are silent no-ops, so live conversation turns are not
     *  cached on it (a RAM overlay tier is future work). The mapping lives for the process
     *  (global arena). */
    public static <S extends RuntimeState> PromptCache<S> open(
            java.nio.file.Path file, KvCodec<S> codec, byte[] modelSeed) throws java.io.IOException {
        MemorySegment map;
        try (java.nio.channels.FileChannel ch = java.nio.channels.FileChannel.open(file, java.nio.file.StandardOpenOption.READ)) {
            map = ch.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, ch.size(), Arena.global());
        }
        ByteBuffer h = map.asSlice(0, HEADER_BYTES).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        if (h.getInt() != MAGIC_FROZEN) {
            throw new IllegalStateException(file + " is not a frozen prompt cache (bad magic)");
        }
        int version = h.getInt();
        if (version != FORMAT_VERSION) {
            throw new IllegalStateException(file + " has frozen-cache format v" + version
                    + ", this build reads v" + FORMAT_VERSION + "; rebuild the cache");
        }
        byte[] stored = new byte[32];
        h.get(stored);
        if (!java.util.Arrays.equals(stored, seed32(modelSeed))) {
            throw new IllegalStateException("frozen cache " + file + " (model seed "
                    + java.util.HexFormat.of().formatHex(stored, 0, 8) + "...) was built for a different model than the one loaded (seed "
                    + java.util.HexFormat.of().formatHex(seed32(modelSeed), 0, 8)
                    + "...); the cache is model-specific - rebuild it or load the matching GGUF");
        }
        int count = h.getInt();
        long indexOffset = h.getLong();
        long kvBytes = indexOffset - HEADER_BYTES;
        CacheStore readOnly = new CacheStore() {
            @Override public MemorySegment allocate(long bytes) { throw new UnsupportedOperationException("frozen cache"); }
            @Override public void free(MemorySegment blob) { throw new UnsupportedOperationException("frozen cache"); }
            @Override public long usedBytes() { return kvBytes; }
        };
        PromptCache<S> cache = new PromptCache<>(codec, readOnly, 0, modelSeed, true);
        ByteBuffer idx = map.asSlice(indexOffset, (long) count * INDEX_ENTRY_BYTES).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            BlockKey key = getKey(idx);
            BlockKey parentKey = getKey(idx);
            int from = idx.getInt(), to = idx.getInt();
            long offset = idx.getLong(), len = idx.getLong();
            PromptCache<S>.Block parent = parentKey.equals(cache.sentinel.key) ? cache.sentinel : cache.blocks.get(parentKey);
            if (parent == null) {
                throw new IllegalStateException(file + " index is corrupt: block " + i + " has an unknown parent");
            }
            PromptCache<S>.Block b = cache.new Block(key, parent, from, to, map.asSlice(offset, len));
            cache.blocks.put(key, b);
            parent.children.add(b);
        }
        return cache;
    }

    private static byte[] seed32(byte[] seed) {
        return java.util.Arrays.copyOf(seed, 32);
    }

    private static void putKey(ByteBuffer buf, BlockKey k) {
        buf.putLong(k.a()).putLong(k.b()).putLong(k.c()).putLong(k.d());
    }

    private static BlockKey getKey(ByteBuffer buf) {
        return new BlockKey(buf.getLong(), buf.getLong(), buf.getLong(), buf.getLong());
    }

    private static long align(long offset) {
        return (offset + ALIGN - 1) & -ALIGN;
    }

    public String stats() {
        return "blocks=" + blocks.size() + " bytes=" + store.usedBytes()
                + " hits=" + hits + " misses=" + misses + " evictions=" + evictions;
    }
}
