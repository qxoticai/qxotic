package com.qxotic.jinfer.cache;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

/**
 * A read-only prompt-cache artifact: any number of prompts as one content-addressed block tree,
 * shared prefixes stored once, produced by {@link PromptCache#freeze} at compile time and mapped
 * lazily at serve time (header and index pages only; KV bytes are untouched until a chain
 * restores). Grafted under a live {@link PromptCache} as its immutable base, its blocks join the
 * cache's key space: resume matches through them, commits dedup against them, eviction never
 * touches them - the "several prompts, read only" tier between the sealed single prompt and the
 * writable RAM cache.
 *
 * <p>Layout (little-endian): {@code JKVF, formatVersion, modelSeed[32], blockCount, indexOffset |
 * KV blobs (64-aligned) | index: per block {key[4], parentKey[4], from, to, byteOffset, byteLen}}
 * in BFS order (parents precede children, so the tree grafts in one pass). The model seed also
 * covers the codec's blob layout: a layout change ships as a format bump, so a stale file fails
 * with a clear error instead of restoring garbage.
 */
public final class FrozenBlocks {

    static final int MAGIC = 0x46564B4A; // "JKVF"
    static final int FORMAT_VERSION = 3; // v3: self-contained blocks (ring rows + residue)
    static final int HEADER_BYTES = 64; // 4+4+32+4+8, padded
    static final int INDEX_ENTRY_BYTES = 96; // 32+32+4+4+8+8+4(crc)+4(pad)
    static final int ALIGN = 64;

    /** One frozen block: an opaque self-contained blob plus its position in the key chain. */
    record Entry(
            PromptCache.BlockKey key,
            PromptCache.BlockKey parentKey,
            int from,
            int to,
            MemorySegment mem,
            int crc) {}

    private final Path file;
    private final List<Entry> entries; // BFS order: parents precede children
    private final long kvBytes;

    private FrozenBlocks(Path file, List<Entry> entries, long kvBytes) {
        this.file = file;
        this.entries = entries;
        this.kvBytes = kvBytes;
    }

    /**
     * Maps {@code file} lazily and validates it belongs to the model identified by {@code
     * modelSeed} - throws a descriptive error when it does not. The mapping lives for the process
     * (global arena); frozen artifacts are process-lifetime.
     */
    public static FrozenBlocks open(Path file, byte[] modelSeed) throws IOException {
        MemorySegment map;
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            map = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), Arena.global());
        }
        ByteBuffer h = map.asSlice(0, HEADER_BYTES).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        if (h.getInt() != MAGIC) {
            throw new IllegalStateException(file + " is not a frozen prompt cache (bad magic)");
        }
        int version = h.getInt();
        if (version != FORMAT_VERSION) {
            throw new IllegalStateException(
                    file
                            + " has frozen-cache format v"
                            + version
                            + ", this build reads v"
                            + FORMAT_VERSION
                            + "; rebuild the cache");
        }
        byte[] stored = new byte[32];
        h.get(stored);
        if (!java.util.Arrays.equals(stored, seed32(modelSeed))) {
            throw new IllegalStateException(
                    "frozen cache "
                            + file
                            + " (model seed "
                            + HexFormat.of().formatHex(stored, 0, 8)
                            + "...) was built for a different model than the one loaded (seed "
                            + HexFormat.of().formatHex(seed32(modelSeed), 0, 8)
                            + "...); the cache is model-specific - rebuild it or load the matching"
                            + " GGUF");
        }
        int count = h.getInt();
        long indexOffset = h.getLong();
        ByteBuffer idx =
                map.asSlice(indexOffset, (long) count * INDEX_ENTRY_BYTES)
                        .asByteBuffer()
                        .order(ByteOrder.LITTLE_ENDIAN);
        List<Entry> entries = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            PromptCache.BlockKey key = getKey(idx);
            PromptCache.BlockKey parentKey = getKey(idx);
            int from = idx.getInt(), to = idx.getInt();
            long offset = idx.getLong(), len = idx.getLong();
            int crc = idx.getInt();
            idx.getInt(); // pad
            entries.add(new Entry(key, parentKey, from, to, map.asSlice(offset, len), crc));
        }
        return new FrozenBlocks(file, entries, indexOffset - HEADER_BYTES);
    }

    List<Entry> entries() {
        return entries;
    }

    /**
     * Compiles a prompt into a frozen artifact: ingests {@code prompt} through a throwaway cache on
     * {@code state} and freezes the resulting chain to {@code out}, returning the fingerprint
     * stream. Owns the PROMPT-COMPILER CONVENTION: the final token of a token-final prompt is
     * committed as its own block, so a serve-time resume with {@code maxPositions = fp.length - 1}
     * lands exactly one token short and a single-token ingest materializes fresh logits.
     */
    public static <S extends com.qxotic.jinfer.RuntimeState> long[] compile(
            Path out,
            com.qxotic.jinfer.Model<?, ?, S> model,
            StateCodec<S> codec,
            byte[] modelSeed,
            S state,
            List<com.qxotic.jinfer.Batch> prompt)
            throws IOException {
        PromptCache<S> build =
                new PromptCache<>(
                        codec, com.qxotic.jinfer.CacheStore.inMemory(), Long.MAX_VALUE, modelSeed);
        CachedSession<S> session = CachedSession.resume(model, build, state, new long[0]);
        // split a trailing token batch so the last token commits as its own block
        List<com.qxotic.jinfer.Batch> batches = new ArrayList<>(prompt);
        int lastIdx = batches.size() - 1;
        if (!batches.isEmpty()
                && batches.get(lastIdx).input() instanceof com.qxotic.jinfer.Batch.Input.Tokens t
                && t.ids().length > 1) {
            int[] ids = t.ids();
            batches.set(
                    lastIdx,
                    com.qxotic.jinfer.Batch.prefill(java.util.Arrays.copyOf(ids, ids.length - 1)));
            session.ingest(batches);
            session.ingest(
                    List.of(com.qxotic.jinfer.Batch.prefill(new int[] {ids[ids.length - 1]})));
        } else {
            session.ingest(batches);
        }
        build.freeze(out);
        return session.fingerprints();
    }

    /**
     * Serves from this artifact: a fresh serve-only cache (no writable budget) layered over it,
     * resuming the longest frozen chain matching {@code fp[0..maxPositions)}. The returned
     * session's {@code position()} is the restore depth; the caller re-ingests the rest.
     */
    public <S extends com.qxotic.jinfer.RuntimeState> CachedSession<S> serve(
            com.qxotic.jinfer.Model<?, ?, S> model,
            StateCodec<S> codec,
            byte[] modelSeed,
            S state,
            long[] fp,
            int maxPositions) {
        PromptCache<S> cache =
                new PromptCache<>(
                        codec, com.qxotic.jinfer.CacheStore.inMemory(), 0, modelSeed, this);
        return CachedSession.resume(model, cache, state, fp, maxPositions);
    }

    /** As {@link #serve} restoring up to the whole stream. */
    public <S extends com.qxotic.jinfer.RuntimeState> CachedSession<S> serve(
            com.qxotic.jinfer.Model<?, ?, S> model,
            StateCodec<S> codec,
            byte[] modelSeed,
            S state,
            long[] fp) {
        return serve(model, codec, modelSeed, state, fp, fp.length);
    }

    /** CRC32C of a blob - the frozen-block integrity stamp (store CRCs cover only pool blobs). */
    static int crc32c(MemorySegment mem) {
        java.util.zip.CRC32C crc = new java.util.zip.CRC32C();
        crc.update(mem.asByteBuffer());
        return (int) crc.getValue();
    }

    /** Total frozen KV bytes (informational; frozen blocks never count against a live budget). */
    public long kvBytes() {
        return kvBytes;
    }

    public int blockCount() {
        return entries.size();
    }

    static byte[] seed32(byte[] seed) {
        return java.util.Arrays.copyOf(seed, 32);
    }

    static void putKey(ByteBuffer buf, PromptCache.BlockKey k) {
        buf.putLong(k.a()).putLong(k.b()).putLong(k.c()).putLong(k.d());
    }

    static PromptCache.BlockKey getKey(ByteBuffer buf) {
        return new PromptCache.BlockKey(buf.getLong(), buf.getLong(), buf.getLong(), buf.getLong());
    }

    static long align(long offset) {
        return (offset + ALIGN - 1) & -ALIGN;
    }

    @Override
    public String toString() {
        return "FrozenBlocks["
                + entries.size()
                + " blocks, "
                + (kvBytes >> 20)
                + "MB, "
                + file
                + "]";
    }
}
