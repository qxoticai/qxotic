package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.ContentKey;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.CRC32C;

/**
 * The prompt-cache artifact FORMAT (JKVF) - this class knows files, not models: any number of
 * prompts as one content-addressed block tree, shared prefixes stored once, produced by {@link
 * BlockTree#freeze} / {@link BlockTree#appendTo} and mapped lazily by {@link #open} (header and
 * index pages only; KV bytes are untouched until a chain restores). Serving happens a layer up:
 * grafted under a live {@link BlockTree} as its immutable base, the artifact's blocks join the
 * cache's key space - resume matches through them, commits dedup against them, eviction never
 * touches them. {@link PromptCache} mounts one as its catalog.
 *
 * <p>Layout (little-endian): one immutable 4 KiB header, two checksummed commit slots on separate 4
 * KiB pages, then 64-aligned KV blobs and complete indexes. A commit names one index generation;
 * append writes and forces the data before publishing into the older slot, so a torn publication
 * falls back to the previous generation. Index entries are {@code {key[4], parentKey[4], from, to,
 * byteOffset, byteLen, crc}} in BFS order (parents precede children, so the tree grafts in one pass
 * - {@link #open} enforces the ordering, so a corrupt index fails at parse, not at graft). The
 * model seed also covers the codec's blob layout: a layout change ships as a format bump, so a
 * stale file fails with a clear error instead of restoring garbage.
 *
 * <p>The cross-process lifecycle, end to end: compile once ({@link PromptCache#define} + {@link
 * PromptCache#export}, or any {@code freeze}), {@code open(path, modelSeed)} at serve start - the
 * SAME model seed, or open throws (an artifact can never serve wrong bytes) - then every request
 * resumes through the mounting tree and ingests only its tail.
 *
 * <p>Reads are safely shared; append serializes publication and replaces its in-memory snapshots
 * only after the new generation is durable.
 */
public final class FrozenBlocks {

    private static final int MAGIC = 0x46564B4A; // "JKVF"
    private static final int FORMAT_VERSION = 4; // v4: dual checksummed commit slots
    private static final int PAGE_BYTES = 4096;
    static final int SLOT_A_OFFSET = PAGE_BYTES;
    static final int SLOT_B_OFFSET = 2 * PAGE_BYTES;
    static final int HEADER_BYTES = 3 * PAGE_BYTES;
    static final int SLOT_BYTES = 64;
    static final int INDEX_ENTRY_BYTES = 96; // 32+32+4+4+8+8+4(crc)+4(pad)
    private static final int ALIGN = 64;
    private static final int COMMIT_MAGIC = 0x54494D43; // "CMIT"
    private static final int HEADER_CRC_OFFSET = 48;
    private static final int SLOT_CRC_OFFSET = 36;
    // The seed's printable value, after the digest: u32 length + UTF-8, inside the page the
    // header CRC already spans. Display only - the digest decides - so a v4 file written before
    // it was recorded reads as zero length, and a reader that predates it never looks.
    private static final int DESCRIPTION_OFFSET = HEADER_CRC_OFFSET + 4;
    private static final int DESCRIPTION_MAX_BYTES = PAGE_BYTES - DESCRIPTION_OFFSET - 4;

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

    private record Commit(
            int slot, long generation, int blockCount, long indexOffset, int indexCrc) {

        long indexBytes() {
            return Math.multiplyExact((long) blockCount, INDEX_ENTRY_BYTES);
        }

        long committedLength() {
            return Math.addExact(indexOffset, indexBytes());
        }
    }

    private final Path file;
    private final ContentKey modelSeed;
    private volatile List<Entry> entries; // immutable BFS snapshot: parents precede children
    private volatile Set<BlockTree.BlockKey> keys;
    private volatile Commit commit;

    private FrozenBlocks(Path file, ContentKey modelSeed, List<Entry> entries, Commit commit) {
        this.file = file;
        this.modelSeed = modelSeed;
        this.entries = List.copyOf(entries);
        this.keys = keys(entries);
        this.commit = commit;
    }

    /** Whether {@code key} is already on disk in this artifact (mounted or appended by us). */
    boolean contains(BlockTree.BlockKey key) {
        return keys.contains(key);
    }

    Path file() {
        return file;
    }

    /**
     * Writes an EMPTY artifact for {@code modelSeed} - the birth of an accumulating catalog, so
     * every later write-back is an {@link #append} against a mounted base, never a rewrite.
     */
    public static void createEmpty(Path file, ContentKey modelSeed) throws IOException {
        write(file, modelSeed, List.of());
    }

    /**
     * Maps {@code file} lazily and validates it belongs to the model identified by {@code
     * modelSeed} - throws a descriptive error when it does not. The mapping is automatic-arena: it
     * stays alive while this object (or any blob sliced from it, e.g. frozen grafts inside a
     * PromptCache) is reachable, and is unmapped by GC after.
     */
    public static FrozenBlocks open(Path file, ContentKey modelSeed) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            return open(file, modelSeed, ch);
        }
    }

    private static FrozenBlocks open(Path file, ContentKey modelSeed, FileChannel ch)
            throws IOException {
        long size = ch.size();
        if (size < 8) {
            throw new IllegalStateException(
                    file + " is not a frozen prompt cache (truncated header)");
        }
        ByteBuffer prefix = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        readFully(ch, prefix, 0);
        prefix.flip();
        if (prefix.getInt() != MAGIC) {
            throw new IllegalStateException(file + " is not a frozen prompt cache (bad magic)");
        }
        int version = prefix.getInt();
        if (version != FORMAT_VERSION) {
            throw new IllegalStateException(
                    file
                            + " has frozen-cache format v"
                            + version
                            + ", this build reads v"
                            + FORMAT_VERSION
                            + "; rebuild the cache");
        }
        if (size < HEADER_BYTES) {
            throw new IllegalStateException(
                    file + " is not a frozen prompt cache (truncated header)");
        }
        ByteBuffer header = ByteBuffer.allocate(PAGE_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(ch, header, 0);
        header.flip();
        validateHeader(file, modelSeed, header);

        Commit a = readCommit(ch, 0, size);
        Commit b = readCommit(ch, 1, size);
        if (a != null && b != null && a.generation() == b.generation() && !sameCommit(a, b)) {
            throw corrupt(file, "commit slots conflict at generation " + a.generation());
        }
        Commit first = newer(a, b), second = first == a ? b : a;
        FrozenBlocks opened = openCommit(file, modelSeed, ch, first);
        if (opened != null) return opened;
        opened = openCommit(file, modelSeed, ch, second);
        if (opened != null) return opened;
        throw corrupt(file, "no valid commit slot");
    }

    private static void validateHeader(Path file, ContentKey modelSeed, ByteBuffer header) {
        if (header.getInt(8) != HEADER_BYTES || header.getInt(12) != SLOT_BYTES) {
            throw corrupt(file, "invalid header geometry");
        }
        if (header.getInt(HEADER_CRC_OFFSET) != crc32cWithZero(header, HEADER_CRC_OFFSET)) {
            throw corrupt(file, "header checksum mismatch");
        }
        byte[] stored = new byte[32];
        header.position(16);
        header.get(stored);
        byte[] seed = modelSeed.digestBytes();
        if (!Arrays.equals(stored, seed)) {
            throw new IllegalStateException(
                    "frozen cache "
                            + file
                            + " was built under a different cache identity."
                            + identityDiff(storedDescription(header), stored, modelSeed.value())
                            + "\nThe identity covers the model file, every attached companion,"
                            + " and - per modality the model projects - its decoder and plan."
                            + " Load the cache the way it was built, or rebuild it here."
                            // only a load that projects media has a decoder to pin
                            + (modelSeed.value().contains("Decoder=")
                                    ? " A JVM and a native image resolve different media"
                                            + " decoders; pin one with -Djinfer.imageDecoder /"
                                            + " -Djinfer.audioDecoder to share a media model's"
                                            + " cache between them."
                                    : ""));
        }
    }

    /**
     * Everything the file says about itself, for a reader with no model in hand: format, the
     * recorded identity and whether it matches the stored digest, both commit slots and which one
     * serves, the index, and every block - all checksums verified, since a report that skipped them
     * would be a guess. Never throws on a bad file: what could not be read is said in place.
     */
    public static String describe(Path file) throws IOException {
        StringBuilder out = new StringBuilder();
        try (var ch = FileChannel.open(file, StandardOpenOption.READ)) {
            long size = ch.size();
            out.append("frozen prompt cache ").append(file).append('\n');
            out.append(row("file", bytes(size)));
            if (size < HEADER_BYTES) {
                return out.append(row("format", "not a frozen prompt cache (truncated header)"))
                        .toString();
            }
            ByteBuffer header = ByteBuffer.allocate(PAGE_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, header, 0);
            header.flip();
            if (header.getInt(0) != MAGIC) {
                return out.append(row("format", "not a frozen prompt cache (bad magic)"))
                        .toString();
            }
            out.append(
                    row(
                            "format",
                            "JKVF v"
                                    + header.getInt(4)
                                    + (header.getInt(4) == FORMAT_VERSION
                                            ? ""
                                            : " (this build reads v" + FORMAT_VERSION + ")")
                                    + ", header "
                                    + header.getInt(8)
                                    + " bytes, commit slots "
                                    + header.getInt(12)
                                    + " bytes"));
            boolean headerOk =
                    header.getInt(HEADER_CRC_OFFSET) == crc32cWithZero(header, HEADER_CRC_OFFSET);
            out.append(row("header crc", headerOk ? "ok" : "MISMATCH - header is corrupt"));
            byte[] stored = new byte[32];
            header.get(16, stored);
            String identity = storedDescription(header);
            out.append(row("identity", identity != null ? identity : "(not recorded)"));
            String digest = HexFormat.of().formatHex(stored);
            String check = "";
            if (identity != null) {
                boolean matches = Arrays.equals(new ContentKey(identity).digestBytes(), stored);
                check =
                        identity.startsWith("sha256:")
                                ? matches
                                        ? " (the identity's own digest)"
                                        : " (does NOT match the identity)"
                                : matches
                                        ? " (sha256 of the identity line: verified)"
                                        : " (does NOT match the identity line)";
            }
            out.append(row("digest", digest + check));
            out.append('\n');

            Commit a = readCommit(ch, 0, size), b = readCommit(ch, 1, size);
            Commit live = newer(a, b);
            out.append(commitRow("commit A", a, live));
            out.append(commitRow("commit B", b, live));
            if (live == null) {
                return out.append(row("blocks", "none - no valid commit slot")).toString();
            }
            long committed = live.committedLength();
            out.append(
                    row(
                            "committed",
                            bytes(committed)
                                    + (size > committed
                                            ? "; "
                                                    + bytes(size - committed)
                                                    + " past the commit (unreached or in flight)"
                                            : "")));
            out.append('\n');
            describeBlocks(out, file, ch, live, stored);
        }
        return out.toString();
    }

    private static void describeBlocks(
            StringBuilder out, Path file, FileChannel ch, Commit live, byte[] storedDigest)
            throws IOException {
        MemorySegment map;
        try (Arena arena = Arena.ofConfined()) {
            map = ch.map(FileChannel.MapMode.READ_ONLY, 0, live.committedLength(), arena);
            boolean indexOk =
                    crc32c(map.asSlice(live.indexOffset(), live.indexBytes())) == live.indexCrc();
            out.append(row("index crc", indexOk ? "ok" : "MISMATCH - index is corrupt"));
            List<Entry> entries;
            try {
                // the chain root hangs from the digest, which is all a file carries
                ContentKey seed =
                        new ContentKey("sha256:" + HexFormat.of().formatHex(storedDigest));
                entries = parseEntries(file, seed, map, live);
            } catch (IllegalStateException e) {
                out.append(row("blocks", "unreadable: " + e.getMessage()));
                return;
            }
            long kvBytes = 0;
            int chains = 0, deepest = 0, badBlobs = 0, coveredTo = 0;
            Map<BlockTree.BlockKey, Integer> index = new HashMap<>();
            Map<BlockTree.BlockKey, Integer> depth = new HashMap<>();
            BlockTree.BlockKey root =
                    BlockTree.chainRoot(
                            new ContentKey("sha256:" + HexFormat.of().formatHex(storedDigest)));
            StringBuilder table = new StringBuilder();
            for (int i = 0; i < entries.size(); i++) {
                Entry e = entries.get(i);
                index.put(e.key(), i);
                boolean head = e.parentKey().equals(root);
                int d = head ? 1 : depth.getOrDefault(e.parentKey(), 0) + 1;
                depth.put(e.key(), d);
                if (head) chains++;
                deepest = Math.max(deepest, d);
                coveredTo = Math.max(coveredTo, e.to());
                kvBytes += e.mem().byteSize();
                boolean ok = crc32c(e.mem()) == e.crc();
                if (!ok) badBlobs++;
                table.append(
                        String.format(
                                "  %4d  %-14s %,14d  %-7s %s%n",
                                i,
                                "[" + e.from() + "," + e.to() + ")",
                                e.mem().byteSize(),
                                head ? "root" : "#" + index.get(e.parentKey()),
                                ok ? "crc ok" : "CRC MISMATCH"));
            }
            out.append(
                    row(
                            "blocks",
                            entries.size()
                                    + " ("
                                    + chains
                                    + (chains == 1 ? " chain" : " chains")
                                    + ", deepest "
                                    + deepest
                                    + "), positions [0,"
                                    + coveredTo
                                    + "), KV "
                                    + bytes(kvBytes)));
            out.append(
                    row(
                            "blob crcs",
                            badBlobs == 0
                                    ? "all " + entries.size() + " ok"
                                    : badBlobs + " MISMATCH"));
            if (!entries.isEmpty()) {
                out.append("\n     #  span                    bytes  parent  integrity\n")
                        .append(table);
            }
        }
    }

    private static String commitRow(String label, Commit c, Commit live) {
        if (c == null) return row(label, "empty or invalid");
        return row(
                label,
                "generation "
                        + c.generation()
                        + ", "
                        + c.blockCount()
                        + (c.blockCount() == 1 ? " block" : " blocks")
                        + ", index at "
                        + c.indexOffset()
                        + (c == live ? "   <- serves" : ""));
    }

    private static String row(String label, String value) {
        return String.format("  %-11s %s%n", label, value);
    }

    private static String bytes(long n) {
        String exact = String.format("%,d bytes", n);
        if (n < 1024) return exact;
        double v = n;
        String[] units = {"KiB", "MiB", "GiB", "TiB"};
        int u = -1;
        while (v >= 1024 && u < units.length - 1) {
            v /= 1024;
            u++;
        }
        return String.format("%s (%.1f %s)", exact, v, units[u]);
    }

    /** The printable seed the writer recorded, or null for a file written before that. */
    private static String storedDescription(ByteBuffer header) {
        int length = header.getInt(DESCRIPTION_OFFSET);
        if (length <= 0 || length > DESCRIPTION_MAX_BYTES) return null;
        byte[] bytes = new byte[length];
        header.get(DESCRIPTION_OFFSET + 4, bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * Both identities side by side, and - when both are printable - the first field that differs,
     * so a reader is not left comparing two long lines by eye. A field is a whitespace-separated
     * token; a quoted plan splits into several, which still points at the right one.
     */
    private static String identityDiff(String built, byte[] storedDigest, String load) {
        StringBuilder out = new StringBuilder();
        out.append("\n  built with: ")
                .append(
                        built != null
                                ? built
                                : "(not recorded; digest "
                                        + HexFormat.of().formatHex(storedDigest)
                                        + ")");
        out.append("\n  this load:  ").append(load);
        if (built != null) {
            String[] was = built.split(" "), now = load.split(" ");
            for (int i = 0; i < Math.max(was.length, now.length); i++) {
                String a = i < was.length ? was[i] : "(absent)";
                String b = i < now.length ? now[i] : "(absent)";
                if (!a.equals(b)) {
                    out.append("\n  differs at: ").append(a).append(" vs ").append(b);
                    break;
                }
            }
        }
        return out.toString();
    }

    private static FrozenBlocks openCommit(
            Path file, ContentKey modelSeed, FileChannel ch, Commit candidate) throws IOException {
        if (candidate == null) return null;
        MemorySegment map =
                ch.map(
                        FileChannel.MapMode.READ_ONLY,
                        0,
                        candidate.committedLength(),
                        Arena.ofAuto());
        if (crc32c(map.asSlice(candidate.indexOffset(), candidate.indexBytes()))
                != candidate.indexCrc()) return null;
        try {
            return new FrozenBlocks(
                    file, modelSeed, parseEntries(file, modelSeed, map, candidate), candidate);
        } catch (IllegalStateException e) {
            return null;
        }
    }

    private static List<Entry> parseEntries(
            Path file, ContentKey modelSeed, MemorySegment map, Commit commit) {
        int count = commit.blockCount();
        long indexOffset = commit.indexOffset();
        // one compact validation pass: every header/index value is ranged BEFORE it is used, so a
        // corrupt artifact fails here with one stable error, never an incidental slice/allocation
        // exception from inside the JDK. Blob CRCs stay lazy (verified once at first restore).
        ByteBuffer idx =
                map.asSlice(indexOffset, commit.indexBytes())
                        .asByteBuffer()
                        .order(ByteOrder.LITTLE_ENDIAN);
        BlockTree.BlockKey root = BlockTree.chainRoot(modelSeed);
        Map<BlockTree.BlockKey, Integer> ends = new HashMap<>();
        ends.put(root, 0);
        List<Entry> entries = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            BlockTree.BlockKey key = getKey(idx);
            BlockTree.BlockKey parentKey = getKey(idx);
            int from = idx.getInt(), to = idx.getInt();
            long offset = idx.getLong(), len = idx.getLong();
            int crc = idx.getInt();
            idx.getInt(); // pad
            if (from < 0 || to <= from) {
                throw corrupt(file, "block " + i + " has span [" + from + "," + to + ")");
            }
            if (offset < HEADER_BYTES
                    || (offset & (ALIGN - 1)) != 0
                    || len < 0
                    || len > indexOffset - offset) {
                throw corrupt(file, "block " + i + " blob lies outside the KV region");
            }
            Integer parentTo = ends.get(parentKey);
            if (parentTo == null) {
                throw corrupt(
                        file, "block " + i + " precedes its parent (index not parents-first)");
            }
            if (from != parentTo) {
                throw corrupt(file, "block " + i + " does not continue its parent");
            }
            if (ends.putIfAbsent(key, to) != null) {
                throw corrupt(file, "block " + i + " duplicates an earlier key");
            }
            entries.add(new Entry(key, parentKey, from, to, offset, map.asSlice(offset, len), crc));
        }
        return entries;
    }

    private static Commit readCommit(FileChannel ch, int slot, long fileSize) throws IOException {
        ByteBuffer bytes = ByteBuffer.allocate(SLOT_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(ch, bytes, slotOffset(slot));
        bytes.flip();
        if (bytes.getInt(0) != COMMIT_MAGIC
                || bytes.getInt(SLOT_CRC_OFFSET) != crc32cWithZero(bytes, SLOT_CRC_OFFSET)) {
            return null;
        }
        long generation = bytes.getLong(8);
        int count = bytes.getInt(16);
        long indexOffset = bytes.getLong(24);
        int indexCrc = bytes.getInt(32);
        if (generation < 0
                || count < 0
                || indexOffset < HEADER_BYTES
                || (indexOffset & (ALIGN - 1)) != 0) return null;
        Commit commit = new Commit(slot, generation, count, indexOffset, indexCrc);
        try {
            if (commit.indexBytes() > Integer.MAX_VALUE || commit.committedLength() > fileSize)
                return null;
        } catch (ArithmeticException e) {
            return null;
        }
        return commit;
    }

    private static Commit newer(Commit a, Commit b) {
        if (a == null) return b;
        if (b == null) return a;
        return a.generation() >= b.generation() ? a : b;
    }

    private static boolean sameCommit(Commit a, Commit b) {
        return a.generation() == b.generation()
                && a.blockCount() == b.blockCount()
                && a.indexOffset() == b.indexOffset()
                && a.indexCrc() == b.indexCrc();
    }

    private static int slotOffset(int slot) {
        return slot == 0 ? SLOT_A_OFFSET : SLOT_B_OFFSET;
    }

    /** Writes a complete artifact; all format serialization lives here, not in the block tree. */
    static void write(Path file, ContentKey modelSeed, List<Entry> entries) throws IOException {
        long[] offsets = new long[entries.size()];
        long off = HEADER_BYTES;
        for (int i = 0; i < entries.size(); i++) {
            offsets[i] = off;
            off = align(Math.addExact(off, entries.get(i).mem().byteSize()));
        }
        long indexOffset = off;
        List<Entry> placed = new ArrayList<>(entries.size());
        for (int i = 0; i < entries.size(); i++) {
            Entry e = entries.get(i);
            placed.add(
                    new Entry(
                            e.key(),
                            e.parentKey(),
                            e.from(),
                            e.to(),
                            offsets[i],
                            e.mem(),
                            e.crc()));
        }
        ByteBuffer index = encodeIndex(placed);
        Commit commit = new Commit(0, 0, entries.size(), indexOffset, crc32c(index));
        try (FileChannel ch =
                FileChannel.open(
                        file,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.READ,
                        StandardOpenOption.WRITE)) {
            writeFully(ch, encodeHeader(modelSeed), 0);
            for (int i = 0; i < entries.size(); i++) {
                writeFully(ch, entries.get(i).mem(), offsets[i]);
            }
            writeFully(ch, index, indexOffset);
            ch.force(true);
            writeFully(ch, encodeCommit(commit), slotOffset(commit.slot()));
            ch.force(true);
        }
    }

    private static ByteBuffer encodeHeader(ContentKey modelSeed) {
        ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        header.putInt(MAGIC)
                .putInt(FORMAT_VERSION)
                .putInt(HEADER_BYTES)
                .putInt(SLOT_BYTES)
                .put(modelSeed.digestBytes());
        byte[] description = modelSeed.value().getBytes(StandardCharsets.UTF_8);
        int recorded = Math.min(description.length, DESCRIPTION_MAX_BYTES);
        header.putInt(DESCRIPTION_OFFSET, recorded);
        header.put(DESCRIPTION_OFFSET + 4, description, 0, recorded);
        header.putInt(HEADER_CRC_OFFSET, crc32cWithZero(header, HEADER_CRC_OFFSET, PAGE_BYTES));
        header.position(0).limit(HEADER_BYTES);
        return header;
    }

    private static ByteBuffer encodeCommit(Commit commit) {
        ByteBuffer slot = ByteBuffer.allocate(SLOT_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        slot.putInt(COMMIT_MAGIC)
                .putInt(0)
                .putLong(commit.generation())
                .putInt(commit.blockCount())
                .putInt(0)
                .putLong(commit.indexOffset())
                .putInt(commit.indexCrc());
        slot.putInt(SLOT_CRC_OFFSET, crc32cWithZero(slot, SLOT_CRC_OFFSET));
        slot.position(0).limit(SLOT_BYTES);
        return slot;
    }

    /** The one error a corrupt artifact fails with - stable wording for operators to grep. */
    private static IllegalStateException corrupt(Path file, String what) {
        return new IllegalStateException(
                file + " is not a valid frozen prompt cache (" + what + ")");
    }

    List<Entry> entries() {
        return entries;
    }

    /** The one index-entry field order, shared by every writer ({@code open} is its reader). */
    private static void putEntry(
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
     * (this artifact's entries re-serialized + the new ones) lands after them, and the inactive
     * commit slot publishes it only once everything is forced. Blob cost is proportional to the new
     * blocks; the index rewrite is small (96 bytes per block) but each append leaves the PREVIOUS
     * index as dead bytes, so a long-lived catalog accumulates O(appends^2) index garbage.
     * Compaction needs no format support: mount the artifact and freeze to a FRESH file ({@link
     * PromptCache#export} is exactly that) - only live blocks are re-serialized. Partial state
     * never touches disk: blocks only exist complete.
     */
    void append(List<Entry> fresh) throws IOException {
        // ponytail: saves are rare; use per-path locks only if global write contention is measured.
        synchronized (FrozenBlocks.class) {
            appendUnderJvmLock(fresh);
        }
    }

    private void appendUnderJvmLock(List<Entry> fresh) throws IOException {
        // The index is a tree, one entry per key, and open() refuses anything else; enforce it
        // where the index is written. Entries already on disk are skipped, not rewritten: the key
        // names the same bytes.
        List<Entry> unseen = new ArrayList<>(fresh.size());
        Set<BlockTree.BlockKey> batch = new HashSet<>();
        for (Entry e : fresh) if (!keys.contains(e.key()) && batch.add(e.key())) unseen.add(e);
        fresh = unseen;
        if (fresh.isEmpty()) return;
        try (FileChannel ch =
                FileChannel.open(file, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            // The class monitor serializes this JVM and the file lock serializes processes. Offsets
            // come from THIS instance's parsed view, so selecting the disk commit turns a stale
            // view into a loud refusal instead of silent last-writer-wins.
            try (FileLock ignored = ch.lock()) {
                FrozenBlocks disk = open(file, modelSeed, ch);
                if (!sameCommit(disk.commit, commit)) {
                    throw new IOException(
                            "catalog "
                                    + file
                                    + " changed since it was mounted ("
                                    + disk.commit.blockCount()
                                    + " blocks on disk, "
                                    + entries.size()
                                    + " mounted): another writer appended; refusing to overwrite");
                }
                appendLocked(ch, fresh, disk.commit);
            }
        }
    }

    private void appendLocked(FileChannel ch, List<Entry> fresh, Commit current)
            throws IOException {
        long off = align(current.committedLength());
        long[] offsets = new long[fresh.size()];
        List<Entry> nextEntries = new ArrayList<>(Math.addExact(entries.size(), fresh.size()));
        nextEntries.addAll(entries);
        for (int i = 0; i < fresh.size(); i++) {
            Entry e = fresh.get(i);
            offsets[i] = off;
            nextEntries.add(
                    new Entry(e.key(), e.parentKey(), e.from(), e.to(), off, e.mem(), e.crc()));
            off = align(Math.addExact(off, e.mem().byteSize()));
        }
        long newIndexOffset = off;
        ByteBuffer idx = encodeIndex(nextEntries);
        Commit nextCommit =
                new Commit(
                        1 - current.slot(),
                        Math.incrementExact(current.generation()),
                        nextEntries.size(),
                        newIndexOffset,
                        crc32c(idx));
        nextEntries = List.copyOf(nextEntries);
        Set<BlockTree.BlockKey> nextKeys = keys(nextEntries);
        for (int i = 0; i < fresh.size(); i++) {
            writeFully(ch, fresh.get(i).mem(), offsets[i]);
        }
        writeFully(ch, idx, newIndexOffset);
        ch.force(true); // blobs and index durable before publication
        writeFully(ch, encodeCommit(nextCommit), slotOffset(nextCommit.slot()));
        ch.force(true);
        entries = nextEntries;
        keys = nextKeys;
        commit = nextCommit;
    }

    /**
     * FileChannel.write is not all-or-nothing: loop until the buffer drains - a truncated blob or
     * index would otherwise be published by the commit slot, bricking that generation.
     */
    private static void readFully(FileChannel ch, ByteBuffer buf, long pos) throws IOException {
        while (buf.hasRemaining()) {
            int read = ch.read(buf, pos);
            if (read < 0) throw new IOException("catalog ends at byte " + pos);
            if (read == 0) throw new IOException("catalog read made no progress at byte " + pos);
            pos += read;
        }
    }

    private static void writeFully(FileChannel ch, ByteBuffer buf, long pos) throws IOException {
        while (buf.hasRemaining()) {
            int written = ch.write(buf, pos);
            if (written == 0)
                throw new IOException("catalog write made no progress at byte " + pos);
            pos += written;
        }
    }

    /** A blob to the channel at {@code pos}: ByteBuffer views are int-sized, a blob is not. */
    private static void writeFully(FileChannel ch, MemorySegment mem, long pos) throws IOException {
        for (long off = 0; off < mem.byteSize(); off += BUFFER_VIEW) {
            long len = Math.min(BUFFER_VIEW, mem.byteSize() - off);
            writeFully(ch, mem.asSlice(off, len).asByteBuffer(), Math.addExact(pos, off));
        }
    }

    /**
     * The largest ByteBuffer view taken over a blob (a define-only prefill block can pass 2 GiB).
     */
    private static final long BUFFER_VIEW = 1L << 30;

    /** CRC32C of a blob - the frozen-block integrity stamp (store CRCs cover only pool blobs). */
    static int crc32c(MemorySegment mem) {
        CRC32C crc = new CRC32C();
        for (long off = 0; off < mem.byteSize(); off += BUFFER_VIEW) {
            long len = Math.min(BUFFER_VIEW, mem.byteSize() - off);
            crc.update(mem.asSlice(off, len).asByteBuffer());
        }
        return (int) crc.getValue();
    }

    private static int crc32c(ByteBuffer bytes) {
        CRC32C crc = new CRC32C();
        crc.update(bytes.duplicate());
        return (int) crc.getValue();
    }

    private static int crc32cWithZero(ByteBuffer bytes, int crcOffset) {
        return crc32cWithZero(bytes, crcOffset, bytes.limit());
    }

    private static int crc32cWithZero(ByteBuffer bytes, int crcOffset, int length) {
        CRC32C crc = new CRC32C();
        ByteBuffer before = bytes.duplicate();
        before.position(0).limit(crcOffset);
        crc.update(before);
        crc.update(new byte[Integer.BYTES]);
        ByteBuffer after = bytes.duplicate();
        after.position(crcOffset + Integer.BYTES).limit(length);
        crc.update(after);
        return (int) crc.getValue();
    }

    private static int indexBytes(int count) {
        return Math.toIntExact(Math.multiplyExact((long) count, INDEX_ENTRY_BYTES));
    }

    private static ByteBuffer encodeIndex(List<Entry> entries) {
        ByteBuffer index =
                ByteBuffer.allocate(indexBytes(entries.size())).order(ByteOrder.LITTLE_ENDIAN);
        for (Entry e : entries) {
            putEntry(
                    index,
                    e.key(),
                    e.parentKey(),
                    e.from(),
                    e.to(),
                    e.offset(),
                    e.mem().byteSize(),
                    e.crc());
        }
        return index.flip();
    }

    private static Set<BlockTree.BlockKey> keys(List<Entry> entries) {
        Set<BlockTree.BlockKey> keys = new HashSet<>(entries.size());
        for (Entry e : entries) keys.add(e.key());
        return keys;
    }

    public int blockCount() {
        return entries.size();
    }

    private static void putKey(ByteBuffer buf, BlockTree.BlockKey k) {
        buf.putLong(k.a()).putLong(k.b()).putLong(k.c()).putLong(k.d());
    }

    private static BlockTree.BlockKey getKey(ByteBuffer buf) {
        return new BlockTree.BlockKey(buf.getLong(), buf.getLong(), buf.getLong(), buf.getLong());
    }

    private static long align(long offset) {
        return Math.addExact(offset, ALIGN - 1) & -ALIGN;
    }

    @Override
    public String toString() {
        return "FrozenBlocks["
                + entries.size()
                + " blocks, "
                + ((commit.committedLength() - HEADER_BYTES) >> 20)
                + "MB, "
                + file
                + "]";
    }
}
