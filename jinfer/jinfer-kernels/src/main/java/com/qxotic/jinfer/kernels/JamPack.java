// Load-time weight packing: re-shuffle quantized weight tensors into jam's packed in-memory
// layout (jam.h JAM_PACK_ABI 1) and swap each view's dtype to JamPacked. One page-aligned slab
// holds every packed tensor; on Apple silicon that same single copy is what Metal reads
// zero-copy (unified memory), so "share with the GPU" needs nothing. The canonical mmap pages
// are simply never touched again - READ_ONLY file-backed pages are the first thing every OS
// evicts under pressure, so "one copy" holds without any explicit drop. Pure Java throughout.
package com.qxotic.jinfer.kernels;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.Memories;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Map;

final class JamPack {

    private JamPack() {}

    private static final boolean ENABLED =
            Boolean.parseBoolean(System.getProperty("jinfer.pack", "true"));

    /** Row-read outside matmul (embedding lookup; covers the tied-LM-head case). Never packed. */
    private static boolean rowRead(String name) {
        return name.equals("token_embd.weight");
    }

    private record Job(
            String name, MemoryView<MemorySegment> view, DataType dt, int rows, int k, long off,
            long bytes) {
        long groupBytes() {
            return bytes / (rows / 4); // uniform by layout construction (jam.h JAM_PACK_ABI)
        }
    }

    /**
     * The packed-size policy ({@link MatMul#nativePackSize}'s shape). A seam, not a feature: the
     * policy lives in libjam but the layout mechanics are pure Java, and the test classpath
     * excludes jam-native - injecting the spec formula is the only way to unit-test the mechanics.
     */
    @FunctionalInterface
    interface PackPolicy {
        long size(DataType dt, int rows, int k);
    }

    /**
     * Pack every tensor the jam backend asks for ({@link JAM#packSize} is the policy - it returns
     * 0 on hardware without packed kernels, for unpackable dtypes, and for unsupported shapes), in
     * place in {@code views}. No-op without jam or with {@code -Djinfer.pack=false}. The slab lives
     * in {@code arena} - the same lifetime as the weights it replaces.
     */
    static Map<String, MemoryView<MemorySegment>> apply(
            Map<String, MemoryView<MemorySegment>> views, Arena arena) {
        if (!ENABLED) return views;
        return apply(views, arena, MatMul::nativePackSize);
    }

    static Map<String, MemoryView<MemorySegment>> apply(
            Map<String, MemoryView<MemorySegment>> views, Arena arena, PackPolicy policy) {
        var jobs = new ArrayList<Job>();
        long total = 0;
        for (var e : views.entrySet()) {
            MemoryView<MemorySegment> v = e.getValue();
            DataType dt = v.dataType();
            // capability, not policy: the layouts packGroup below can produce. Whether a tensor
            // SHOULD pack is jam's call alone (nativePackSize == 0 keeps it canonical).
            if (dt != DataType.Q4_0 && dt != DataType.Q4_K
                    && dt != DataType.Q5_K && dt != DataType.Q6_K) continue;
            if (rowRead(e.getKey())) continue;
            long[] dims = v.shape().toArray(); // physical: innermost counts BLOCKS
            if (dims.length < 2) continue;
            long k = dims[dims.length - 1] * dt.elementsPerBlock();
            long rows = 1;
            for (int i = 0; i < dims.length - 1; i++) rows *= dims[i];
            // jam gates total rows % 4; a 3D (expert-sliced) tensor additionally needs each
            // matrix's rows on a group boundary so expert offsets stay layout-exact.
            if (dims[dims.length - 2] % 4 != 0) continue;
            if (rows > Integer.MAX_VALUE || k > Integer.MAX_VALUE) continue;
            long bytes = policy.size(dt, (int) rows, (int) k);
            // The layout is rows/4 equal-stride groups; a size that doesn't divide is some other
            // ABI's answer - keep canonical rather than truncate into a corrupt layout.
            if (bytes == 0 || bytes % (rows / 4) != 0) continue;
            total = (total + 63) & ~63L; // each tensor starts cache-line aligned
            jobs.add(new Job(e.getKey(), v, dt, (int) rows, (int) k, total, bytes));
            total += bytes;
        }
        if (jobs.isEmpty()) return views;

        // A PRIVATE (copy-on-write) mapping of an empty temp file - Java's portable spelling of
        // an anonymous mmap. Not Arena.allocate: model-scale slabs (a 26B model packs ~15 GB)
        // blow the JVM direct-memory limit, while mapped memory is exempt, page-aligned (all
        // Metal's zero-copy wrap needs), and portable incl. Windows. PRIVATE, not READ_WRITE:
        // written pages become anonymous copies that are NEVER flushed back - a shared mapping
        // would have the kernel lazily writing the whole throwaway slab to disk. The file itself
        // stays 0 bytes; it exists only because FileChannel.map needs one. The mapping rides the
        // arena's lifetime; the file is unlinked immediately (POSIX) or on exit (Windows).
        MemorySegment slab = mappedSlab(total, arena);
        Memory<MemorySegment> slabMemory = Memories.of(slab);
        try (var ignored = Timer.log("Pack " + jobs.size() + " weight tensors")) {
            for (Job j : jobs) {
                MemorySegment src = j.view.memory().base();
                long srcBase = j.view.byteOffset();
                long srcRowBytes =
                        (long) j.k / j.dt.elementsPerBlock() * j.dt.byteSize();
                long gb = j.groupBytes();
                Parallel.forLoop(
                        j.rows / 4,
                        g -> packGroup(j.dt, src, srcBase, srcRowBytes, slab, j.off, gb, g, j.k));
                JamPacked packed = JamPacked.of(j.dt, j.k, gb / 4);
                Shape logical = j.dt.logicalShape(j.view.shape());
                views.put(
                        j.name,
                        MemoryView.of(
                                slabMemory, j.off, packed,
                                Layout.rowMajor(packed.physicalShape(logical))));
            }
        }
        return views;
    }

    // ---- one 4-row group -> the jam.h JAM_PACK_ABI 1 sections ----

    private static void packGroup(
            DataType dt, MemorySegment src, long srcBase, long srcRowBytes,
            MemorySegment dst, long dstBase, long gb, int g, int k) {
        final int nb = k / 32, sb = k / 256;
        final long go = dstBase + (long) g * gb;
        for (int r = 0; r < 4; r++) {
            final long w = srcBase + ((long) g * 4 + r) * srcRowBytes;
            if (dt == DataType.Q4_0) {
                long so = go + (long) nb * 64;
                for (int b = 0; b < nb; b++) {
                    MemorySegment.copy(src, w + (long) b * 18 + 2, dst, go + (long) b * 64 + r * 16L, 16);
                    dst.set(JAVA_FLOAT_UNALIGNED, so + ((long) b * 4 + r) * 4, f16(src, w + (long) b * 18));
                }
            } else if (dt == DataType.Q6_K) {
                long sco = go + (long) nb * 128, ddo = sco + (long) nb * 8;
                for (int B = 0; B < sb; B++) {
                    long s = w + (long) B * 210;
                    dst.set(JAVA_FLOAT_UNALIGNED, ddo + ((long) B * 4 + r) * 4, f16(src, s + 208));
                    for (int hh = 0; hh < 2; hh++)
                        for (int gg = 0; gg < 4; gg++) {
                            int blk = B * 8 + hh * 4 + gg;
                            long ql = s + hh * 64 + (gg & 1) * 32, qh = s + 128 + hh * 32;
                            long d = go + (long) blk * 128 + r * 32L;
                            for (int j = 0; j < 32; j++) {
                                int lo = gg < 2 ? (u8(src, ql + j) & 0xF) : (u8(src, ql + j) >> 4);
                                int hi = ((u8(src, qh + j) >> (2 * gg)) & 3) << 4;
                                dst.set(JAVA_BYTE, d + j, (byte) ((lo | hi) - 32));
                            }
                            dst.set(JAVA_BYTE, sco + (long) blk * 8 + r, src.get(JAVA_BYTE, s + 192 + hh * 8 + gg * 2));
                            dst.set(JAVA_BYTE, sco + (long) blk * 8 + 4 + r, src.get(JAVA_BYTE, s + 192 + hh * 8 + gg * 2 + 1));
                        }
                }
            } else { // Q4_K / Q5_K
                boolean q5 = dt == DataType.Q5_K;
                int pb = q5 ? 128 : 64, bytes = q5 ? 176 : 144;
                long smo = go + (long) nb * pb, ddo = smo + (long) nb * 8;
                int[] sc = new int[8], mn = new int[8];
                for (int B = 0; B < sb; B++) {
                    long s = w + (long) B * bytes;
                    dst.set(JAVA_FLOAT_UNALIGNED, ddo + ((long) B * 8 + r) * 4, f16(src, s));
                    dst.set(JAVA_FLOAT_UNALIGNED, ddo + ((long) B * 8 + 4 + r) * 4, f16(src, s + 2));
                    scalesMins(src, s + 4, sc, mn);
                    long qh = s + 16, qs = s + (q5 ? 48 : 16);
                    for (int gg = 0; gg < 4; gg++) {
                        int bl = B * 8 + 2 * gg, bh = bl + 1;
                        long dl = go + (long) bl * pb + r * (long) (pb / 4);
                        long dh = go + (long) bh * pb + r * (long) (pb / 4);
                        if (q5) {
                            for (int j = 0; j < 32; j++) {
                                int b = u8(src, qs + gg * 32 + j), h = u8(src, qh + j);
                                dst.set(JAVA_BYTE, dl + j, (byte) ((b & 0xF) | (((h >> (2 * gg)) & 1) << 4)));
                                dst.set(JAVA_BYTE, dh + j, (byte) ((b >> 4) | (((h >> (2 * gg + 1)) & 1) << 4)));
                            }
                        } else {
                            for (int j = 0; j < 16; j++) { // elem e low nibble, e+16 high (Q4_0 order)
                                int alo = u8(src, qs + gg * 32 + j), ahi = u8(src, qs + gg * 32 + 16 + j);
                                dst.set(JAVA_BYTE, dl + j, (byte) ((alo & 0xF) | ((ahi & 0xF) << 4)));
                                dst.set(JAVA_BYTE, dh + j, (byte) ((alo >> 4) | ((ahi >> 4) << 4)));
                            }
                        }
                        dst.set(JAVA_BYTE, smo + (long) bl * 8 + r, (byte) sc[2 * gg]);
                        dst.set(JAVA_BYTE, smo + (long) bl * 8 + 4 + r, (byte) mn[2 * gg]);
                        dst.set(JAVA_BYTE, smo + (long) bh * 8 + r, (byte) sc[2 * gg + 1]);
                        dst.set(JAVA_BYTE, smo + (long) bh * 8 + 4 + r, (byte) mn[2 * gg + 1]);
                    }
                }
            }
        }
    }

    /** The 8 6-bit (scale, min) pairs of a Q4_K/Q5_K super-block (ggml get_scale_min_k4). */
    private static void scalesMins(MemorySegment src, long off, int[] sc, int[] mn) {
        for (int t = 0; t < 4; t++) {
            int a = u8(src, off + t), b = u8(src, off + t + 4), c = u8(src, off + t + 8);
            sc[t] = a & 63;
            mn[t] = b & 63;
            sc[t + 4] = (c & 0xF) | ((a >> 6) << 4);
            mn[t + 4] = (c >> 4) | ((b >> 6) << 4);
        }
    }

    private static int u8(MemorySegment s, long off) {
        return s.get(JAVA_BYTE, off) & 0xFF;
    }

    private static float f16(MemorySegment s, long off) {
        return Float.float16ToFloat(s.get(JAVA_SHORT_UNALIGNED, off));
    }

    private static MemorySegment mappedSlab(long bytes, Arena arena) {
        try {
            Path file = Files.createTempFile("jinfer-pack-", ".bin");
            try (FileChannel ch =
                    FileChannel.open(
                            file, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
                MemorySegment slab = ch.map(FileChannel.MapMode.PRIVATE, 0, bytes, arena);
                try {
                    Files.delete(file); // POSIX: unlink now, pages live until the arena closes
                } catch (IOException windowsKeepsMappedFiles) {
                    file.toFile().deleteOnExit();
                }
                return slab;
            }
        } catch (IOException e) {
            throw new OutOfMemoryError("jam pack slab: cannot map " + bytes + " B: " + e);
        }
    }
}
