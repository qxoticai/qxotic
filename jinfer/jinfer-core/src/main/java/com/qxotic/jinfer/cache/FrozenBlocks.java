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
 * The prompt-cache artifact FORMAT (JKVF) - this class knows files, not models: any number of
 * prompts as one content-addressed block tree, shared prefixes stored once, produced by {@link
 * BlockTree#freeze} / {@link BlockTree#appendTo} and mapped lazily by {@link #open} (header and
 * index pages only; KV bytes are untouched until a chain restores). Serving happens a layer up:
 * grafted under a live {@link BlockTree} as its immutable base, the artifact's blocks join the
 * cache's key space - resume matches through them, commits dedup against them, eviction never
 * touches them. {@link PromptCache} mounts one as its catalog.
 *
 * <p>Layout (little-endian): {@code JKVF, formatVersion, modelSeed[32], blockCount, indexOffset |
 * KV blobs (64-aligned) | index: per block {key[4], parentKey[4], from, to, byteOffset, byteLen}}
 * in BFS order (parents precede children, so the tree grafts in one pass). The model seed also
 * covers the codec's blob layout: a layout change ships as a format bump, so a stale file fails
 * with a clear error instead of restoring garbage.
 *
 * <p>The cross-process lifecycle, end to end: compile once ({@link PromptCache#define} + {@link
 * PromptCache#export}, or any {@code freeze}), {@code open(path, modelSeed)} at serve start - the
 * SAME model seed, or open throws (an artifact can never serve wrong bytes) - then every request
 * resumes through the mounting tree and ingests only its tail.
 *
 * <p>One open instance is immutable and safely shared across engines/pipelines.
 */
public final class FrozenBlocks {

    static final int MAGIC = 0x46564B4A; // "JKVF"
    static final int FORMAT_VERSION = 3; // v3: self-contained blocks (ring rows + residue)
    static final int HEADER_BYTES = 64; // 4+4+32+4+8, padded
    static final int INDEX_ENTRY_BYTES = 96; // 32+32+4+4+8+8+4(crc)+4(pad)
    static final int ALIGN = 64;
    static final int COUNT_OFFSET = 40; // magic(4) + version(4) + seed(32): count, then indexOffset

    /**
     * One frozen block: an opaque self-contained blob plus its position in the key chain. {@code
     * offset} is the blob's file offset for opened artifacts; fresh (RAM) entries carry -1 until
     * {@link #append} places them.
     */
    record Entry(
            BlockTree.BlockKey key,
            BlockTree.BlockKey parentKey,
            int from,
            int to,
            long offset,
            MemorySegment mem,
            int crc) {}

    private final Path file;
    private final List<Entry> entries; // BFS order: parents precede children
    private final long kvBytes;
    private long indexOffset; // advances as append() publishes new indexes

    private FrozenBlocks(Path file, List<Entry> entries, long kvBytes, long indexOffset) {
        this.file = file;
        this.entries = entries;
        this.kvBytes = kvBytes;
        this.indexOffset = indexOffset;
    }

    Path file() {
        return file;
    }

    /**
     * Writes an EMPTY artifact for {@code modelSeed} - the birth of an accumulating catalog, so
     * every later write-back is an {@link #append} against a mounted base, never a rewrite.
     */
    public static void createEmpty(Path file, byte[] modelSeed) throws IOException {
        try (FileChannel ch =
                        FileChannel.open(
                                file,
                                StandardOpenOption.CREATE,
                                StandardOpenOption.TRUNCATE_EXISTING,
                                StandardOpenOption.READ,
                                StandardOpenOption.WRITE);
                Arena arena = Arena.ofConfined()) {
            MemorySegment map = ch.map(FileChannel.MapMode.READ_WRITE, 0, HEADER_BYTES, arena);
            map.asByteBuffer()
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .putInt(MAGIC)
                    .putInt(FORMAT_VERSION)
                    .put(seed32(modelSeed))
                    .putInt(0)
                    .putLong(HEADER_BYTES);
            map.force(); // durable before anyone appends - a half-born header bricks later boots
        }
    }

    /**
     * Maps {@code file} lazily and validates it belongs to the model identified by {@code
     * modelSeed} - throws a descriptive error when it does not. The mapping is automatic-arena: it
     * stays alive while this object (or any blob sliced from it, e.g. frozen grafts inside a
     * PromptCache) is reachable, and is unmapped by GC after.
     */
    public static FrozenBlocks open(Path file, byte[] modelSeed) throws IOException {
        MemorySegment map;
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            if (ch.size() < HEADER_BYTES) {
                // e.g. a crash between file birth and header write-back
                throw new IllegalStateException(
                        file + " is not a frozen prompt cache (truncated header)");
            }
            map = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), Arena.ofAuto());
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
            BlockTree.BlockKey key = getKey(idx);
            BlockTree.BlockKey parentKey = getKey(idx);
            int from = idx.getInt(), to = idx.getInt();
            long offset = idx.getLong(), len = idx.getLong();
            int crc = idx.getInt();
            idx.getInt(); // pad
            entries.add(new Entry(key, parentKey, from, to, offset, map.asSlice(offset, len), crc));
        }
        return new FrozenBlocks(file, entries, indexOffset - HEADER_BYTES, indexOffset);
    }

    List<Entry> entries() {
        return entries;
    }

    /** The one index-entry field order, shared by every writer ({@code open} is its reader). */
    static void putEntry(
            ByteBuffer idx,
            BlockTree.BlockKey key,
            BlockTree.BlockKey parentKey,
            int from,
            int to,
            long offset,
            long byteLen,
            int crc) {
        putKey(idx, key);
        putKey(idx, parentKey);
        idx.putInt(from).putInt(to).putLong(offset).putLong(byteLen);
        idx.putInt(crc);
        idx.putInt(0); // pad
    }

    /**
     * Appends {@code fresh} entries (mem still in RAM, BFS parents-first) to THIS artifact's file
     * without rewriting existing KV: new blobs land after the current index, a fresh full index
     * (this artifact's entries re-serialized + the new ones) lands after them, and the header flips
     * only once everything is forced - a torn append leaves the old catalog intact behind the old
     * header. Blob cost is proportional to the new blocks; the index rewrite is small (96 bytes per
     * block) but each append leaves the PREVIOUS index as dead bytes, so a long-lived catalog
     * accumulates O(appends^2) index garbage - compact by freezing to a fresh file when that ever
     * matters. Partial state never touches disk: blocks only exist complete.
     */
    void append(List<Entry> fresh) throws IOException {
        if (fresh.isEmpty()) return;
        try (FileChannel ch =
                FileChannel.open(file, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            // single-writer law, enforced: offsets below come from THIS instance's parsed view,
            // so a writer that mounted before another writer appended would overwrite its blocks
            // and orphan them - silent last-writer-wins. The advisory lock serializes writers;
            // the header re-read turns a stale view into a loud refusal instead of data loss.
            try (java.nio.channels.FileLock ignored = ch.lock()) {
                ByteBuffer head = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
                ch.read(head, COUNT_OFFSET);
                head.flip();
                int diskCount = head.getInt();
                long diskIndexOffset = head.getLong();
                if (diskCount != entries.size() || diskIndexOffset != indexOffset) {
                    throw new IOException(
                            "catalog "
                                    + file
                                    + " changed since it was mounted ("
                                    + diskCount
                                    + " blocks on disk, "
                                    + entries.size()
                                    + " mounted): another writer appended; refusing to overwrite");
                }
                appendLocked(ch, fresh);
            }
        }
    }

    private void appendLocked(FileChannel ch, List<Entry> fresh) throws IOException {
        long off = align(indexOffset + (long) entries.size() * INDEX_ENTRY_BYTES);
        long[] offsets = new long[fresh.size()];
        for (int i = 0; i < fresh.size(); i++) {
            offsets[i] = off;
            ch.write(fresh.get(i).mem().asByteBuffer(), off);
            off = align(off + fresh.get(i).mem().byteSize());
        }
        long newIndexOffset = off;
        ByteBuffer idx =
                ByteBuffer.allocate((entries.size() + fresh.size()) * INDEX_ENTRY_BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN);
        for (Entry e : entries) {
            putEntry(
                    idx,
                    e.key(),
                    e.parentKey(),
                    e.from(),
                    e.to(),
                    e.offset(),
                    e.mem().byteSize(),
                    e.crc());
        }
        for (int i = 0; i < fresh.size(); i++) {
            Entry e = fresh.get(i);
            putEntry(
                    idx,
                    e.key(),
                    e.parentKey(),
                    e.from(),
                    e.to(),
                    offsets[i],
                    e.mem().byteSize(),
                    e.crc());
        }
        idx.flip();
        ch.write(idx, newIndexOffset);
        ch.force(true); // everything durable BEFORE the header flips
        ByteBuffer flip = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        flip.putInt(entries.size() + fresh.size()).putLong(newIndexOffset).flip();
        ch.write(flip, COUNT_OFFSET);
        ch.force(false); // in-place 12-byte publish: no size change to flush
        // keep the parsed view coherent so a later append re-serializes the right index
        for (int i = 0; i < fresh.size(); i++) {
            Entry e = fresh.get(i);
            entries.add(
                    new Entry(
                            e.key(),
                            e.parentKey(),
                            e.from(),
                            e.to(),
                            offsets[i],
                            e.mem(),
                            e.crc()));
        }
        indexOffset = newIndexOffset;
    }

    /** CRC32C of a blob - the frozen-block integrity stamp (store CRCs cover only pool blobs). */
    static int crc32c(MemorySegment mem) {
        java.util.zip.CRC32C crc = new java.util.zip.CRC32C();
        crc.update(mem.asByteBuffer());
        return (int) crc.getValue();
    }

    public int blockCount() {
        return entries.size();
    }

    static byte[] seed32(byte[] seed) {
        return java.util.Arrays.copyOf(seed, 32);
    }

    static void putKey(ByteBuffer buf, BlockTree.BlockKey k) {
        buf.putLong(k.a()).putLong(k.b()).putLong(k.c()).putLong(k.d());
    }

    static BlockTree.BlockKey getKey(ByteBuffer buf) {
        return new BlockTree.BlockKey(buf.getLong(), buf.getLong(), buf.getLong(), buf.getLong());
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
