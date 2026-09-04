package com.qxotic.jinfer.kernels;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memories;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * {@link JamPack} against two independent executable specs: a canonical-GGUF dequantizer (ggml's
 * dequantize_row_*) and a packed-layout dequantizer written from the jam.h {@code JAM_PACK_ABI 1}
 * layout comment. Every element must match BIT-EXACTLY - the spec's own claim ("values are exactly
 * the canonical dequant; the transform only reorders bytes and widens scales") - over random
 * payloads with per-block varied scales, so any byte landing in the wrong line, lane, row, group,
 * or section slot fails. The size policy is injected (the spec formula): the native library is
 * excluded from this classpath, and the policy is its half of the contract anyway - these tests own
 * the pure-Java mechanics.
 */
class JamPackTest {

    private final Arena arena = Arena.ofAuto();

    /** jam.h JAM_PACK_ABI 1 group sizes (jam_pack_group_bytes). */
    private static long groupBytes(DataType dt, int k) {
        long nb = k / 32, sb = k / 256;
        if (dt == DataType.Q4_0) return nb * 80;
        if (dt == DataType.MXFP4) return nb * 68;
        if (dt == DataType.Q4_K) return nb * 72 + sb * 32;
        if (dt == DataType.Q5_K) return nb * 136 + sb * 32;
        if (dt == DataType.Q6_K) return nb * 136 + sb * 16;
        return 0;
    }

    private static final JamPack.PackPolicy SPEC = (dt, rows, k) -> (rows / 4) * groupBytes(dt, k);

    // ---- bit-exact layout parity, one test per dtype ----

    @Test
    void q4_0PackedBitExact() {
        packedBitExact(DataType.Q4_0);
    }

    @Test
    void mxfp4PackedBitExact() {
        packedBitExact(DataType.MXFP4);
    }

    @Test
    void q4_kPackedBitExact() {
        packedBitExact(DataType.Q4_K);
    }

    @Test
    void q5_kPackedBitExact() {
        packedBitExact(DataType.Q5_K);
    }

    @Test
    void q6_kPackedBitExact() {
        packedBitExact(DataType.Q6_K);
    }

    /**
     * Negative control: the parity walk itself must have teeth. A single flipped slab byte fails it
     * - so a future refactor can never leave both readers vacuously agreeing.
     */
    @Test
    void parityWalkDetectsSingleByteCorruption() {
        int m = 8, k = 512;
        MemorySegment canon = canonical(DataType.Q6_K, m, k, 99);
        var views = new LinkedHashMap<String, MemoryView<MemorySegment>>();
        views.put("w", Views.wrap(canon, DataType.Q6_K, Shape.flat(m, 2)));
        MemoryView<MemorySegment> v = JamPack.apply(views, arena, SPEC).get("w");
        MemorySegment slab = v.memory().base();
        long victim = v.byteOffset() + 5; // a payload byte in group 0
        slab.set(JAVA_BYTE, victim, (byte) (slab.get(JAVA_BYTE, victim) ^ 0x41));
        assertThrows(
                AssertionError.class,
                () -> assertParity(canon, 0, v, DataType.Q6_K, m, k),
                "parity walk must detect a corrupted packed byte");
    }

    private void packedBitExact(DataType dt) {
        int m = 8, k = 512; // 2 groups, 16 32-blocks/row, 2 super-blocks/row
        MemorySegment canon = canonical(dt, m, k, dt.name().hashCode());
        var views = new LinkedHashMap<String, MemoryView<MemorySegment>>();
        views.put("blk.0.w", Views.wrap(canon, dt, Shape.flat(m, k / dt.elementsPerBlock())));
        MemoryView<MemorySegment> v = JamPack.apply(views, arena, SPEC).get("blk.0.w");

        JamPacked packed = assertInstanceOf(JamPacked.class, v.dataType());
        assertSame(dt, packed.base());
        assertEquals(k, packed.elementsPerBlock());
        assertEquals(groupBytes(dt, k) / 4, packed.byteSize(), "uniform row stride");
        assertArrayEquals(
                new long[] {m, k}, packed.logicalShape(v.shape()).toArray(), "shape roundtrip");
        assertParity(canon, 0, v, dt, m, k);
    }

    /** A 3D (expert-sliced) tensor: flattened rows pack, group addressing exact per expert. */
    @Test
    void expertTensorPacksPerGroup() {
        int experts = 2, mPer = 4, k = 256;
        MemorySegment canon = canonical(DataType.Q6_K, experts * mPer, k, 7);
        var views = new LinkedHashMap<String, MemoryView<MemorySegment>>();
        views.put("ffn.exps", Views.wrap(canon, DataType.Q6_K, Shape.flat(experts, mPer, 1)));
        MemoryView<MemorySegment> v = JamPack.apply(views, arena, SPEC).get("ffn.exps");
        assertInstanceOf(JamPacked.class, v.dataType());
        assertParity(canon, 0, v, DataType.Q6_K, experts * mPer, k);
    }

    /** A source view at a nonzero byte offset (a tensor mid-mmap) packs from the right bytes. */
    @Test
    void sourceViewOffsetRespected() {
        int m = 4, k = 256, pad = 128;
        MemorySegment canon = canonical(DataType.Q4_K, m, k, 11);
        MemorySegment padded = arena.allocate(pad + canon.byteSize(), 64);
        MemorySegment.copy(canon, 0, padded, pad, canon.byteSize());
        var views = new LinkedHashMap<String, MemoryView<MemorySegment>>();
        views.put(
                "blk.0.w",
                MemoryView.of(
                        Memories.of(padded),
                        pad,
                        DataType.Q4_K,
                        Layout.rowMajor(Shape.flat(m, k / 256))));
        MemoryView<MemorySegment> v = JamPack.apply(views, arena, SPEC).get("blk.0.w");
        assertInstanceOf(JamPacked.class, v.dataType());
        assertParity(padded, pad, v, DataType.Q4_K, m, k);
    }

    // ---- selection and planning ----

    /** Capability gates: what packGroup cannot produce is never offered to the policy. */
    @Test
    void ineligibleTensorsKeepCanonicalViews() {
        var views = new LinkedHashMap<String, MemoryView<MemorySegment>>();
        var embd = Views.wrap(canonical(DataType.Q4_K, 8, 256, 1), DataType.Q4_K, Shape.flat(8, 1));
        var flat = Views.wrap(canonical(DataType.Q4_K, 8, 256, 2), DataType.Q4_K, Shape.flat(8));
        var odd = Views.wrap(canonical(DataType.Q4_K, 6, 256, 3), DataType.Q4_K, Shape.flat(6, 1));
        var q8 = Views.wrap(Oracles.q8(arena, 8, 256, 4), DataType.Q8_0, Shape.flat(8, 8));
        views.put("token_embd.weight", embd); // row-read (embedding lookup)
        views.put(
                "per_layer_token_embd.weight",
                embd); // row-read (Gemma 4 per-layer embedding lookup)
        views.put("flat", flat); // rank 1
        views.put("odd", odd); // rows % 4 != 0
        views.put("q8", q8); // no packed layout for the dtype
        Map<String, MemoryView<MemorySegment>> out = JamPack.apply(views, arena, SPEC);
        assertSame(embd, out.get("token_embd.weight"));
        assertSame(embd, out.get("per_layer_token_embd.weight"));
        assertSame(flat, out.get("flat"));
        assertSame(odd, out.get("odd"));
        assertSame(q8, out.get("q8"));
    }

    /** The policy owns the decision: 0 keeps canonical; a non-uniform size is not this ABI. */
    @Test
    void policyDeclineAndAbiDriftKeepCanonical() {
        var view =
                Views.wrap(canonical(DataType.Q4_0, 8, 512, 5), DataType.Q4_0, Shape.flat(8, 16));
        var views = new LinkedHashMap<String, MemoryView<MemorySegment>>();
        views.put("w", view);
        assertSame(view, JamPack.apply(views, arena, (dt, rows, k) -> 0).get("w"));
        assertSame(
                view,
                JamPack.apply(views, arena, (dt, rows, k) -> SPEC.size(dt, rows, k) + 1).get("w"));
    }

    /** All packed tensors share one slab; each starts cache-line aligned. */
    @Test
    void slabSharedAndTensorsCacheLineAligned() {
        int m = 4, k = 32; // Q4_0 group = 80 bytes: exercises the 64B round-up between tensors
        var views = new LinkedHashMap<String, MemoryView<MemorySegment>>();
        MemorySegment a = canonical(DataType.Q4_0, m, k, 21),
                b = canonical(DataType.Q4_0, m, k, 22);
        views.put("a", Views.wrap(a, DataType.Q4_0, Shape.flat(m, 1)));
        views.put("b", Views.wrap(b, DataType.Q4_0, Shape.flat(m, 1)));
        Map<String, MemoryView<MemorySegment>> out = JamPack.apply(views, arena, SPEC);
        MemoryView<MemorySegment> va = out.get("a"), vb = out.get("b");
        assertSame(va.memory(), vb.memory(), "one slab for all packed tensors");
        assertEquals(0, va.byteOffset() % 64);
        assertEquals(0, vb.byteOffset() % 64);
        assertTrue(vb.byteOffset() >= va.byteOffset() + 80, "tensors must not overlap");
        assertParity(a, 0, va, DataType.Q4_0, m, k);
        assertParity(b, 0, vb, DataType.Q4_0, m, k);
    }

    /** The slab's temp file is only a mapping token: unlinked before apply returns (POSIX). */
    @Test
    void slabTempFileUnlinked() throws Exception {
        Assumptions.assumeTrue(
                !System.getProperty("os.name", "").startsWith("Windows"), "POSIX unlink semantics");
        Path tmp = Path.of(System.getProperty("java.io.tmpdir"));
        long before = packTempFiles(tmp);
        var views = new LinkedHashMap<String, MemoryView<MemorySegment>>();
        views.put(
                "w",
                Views.wrap(canonical(DataType.Q4_0, 8, 512, 6), DataType.Q4_0, Shape.flat(8, 16)));
        assertInstanceOf(JamPacked.class, JamPack.apply(views, arena, SPEC).get("w").dataType());
        assertEquals(before, packTempFiles(tmp), "slab temp file must not survive apply");
    }

    private static long packTempFiles(Path tmp) throws Exception {
        try (var files = Files.list(tmp)) {
            return files.filter(p -> p.getFileName().toString().startsWith("jinfer-pack-")).count();
        }
    }

    // ---- test data: random payloads, per-block varied (finite, positive) f16 scales ----

    private MemorySegment canonical(DataType dt, int m, int k, long seed) {
        Random rng = new Random(seed);
        int bs = (int) dt.byteSize();
        long blocks = (long) m * k / dt.elementsPerBlock();
        MemorySegment s = arena.allocate(blocks * bs, 64);
        for (long i = 0; i < s.byteSize(); i++) s.set(JAVA_BYTE, i, (byte) rng.nextInt(256));
        for (long b = 0; b < blocks; b++) {
            long bo = b * bs;
            short d = Float.floatToFloat16((rng.nextFloat() + 0.5f) * 0.01f);
            short dmin = Float.floatToFloat16((rng.nextFloat() + 0.5f) * 0.005f);
            if (dt == DataType.MXFP4) {
                s.set(JAVA_BYTE, bo, (byte) (120 + rng.nextInt(8)));
            } else if (dt == DataType.Q6_K) {
                s.set(JAVA_SHORT_UNALIGNED, bo + 208, d);
            } else {
                s.set(JAVA_SHORT_UNALIGNED, bo, d);
                if (dt != DataType.Q4_0) s.set(JAVA_SHORT_UNALIGNED, bo + 2, dmin);
            }
        }
        return s;
    }

    // ---- the parity walk ----

    private static void assertParity(
            MemorySegment canon,
            long canonBase,
            MemoryView<MemorySegment> packedView,
            DataType dt,
            int m,
            int k) {
        MemorySegment slab = packedView.memory().base();
        long base = packedView.byteOffset();
        for (int row = 0; row < m; row++) {
            for (int e = 0; e < k; e++) {
                float expected = canonicalValue(canon, canonBase, dt, k, row, e);
                float actual = packedValue(slab, base, dt, k, row, e);
                if (Float.floatToIntBits(expected) != Float.floatToIntBits(actual)) {
                    assertEquals(expected, actual, dt + " row " + row + " elem " + e);
                }
            }
        }
    }

    // ---- canonical reader: ggml dequantize_row_{q4_0,q4_K,q5_K,q6_K}, element at a time ----

    private static float canonicalValue(
            MemorySegment s, long tensorBase, DataType dt, int k, int row, int e) {
        long rowBase = tensorBase + (long) row * (k / dt.elementsPerBlock()) * dt.byteSize();
        if (dt == DataType.Q4_0) {
            long bo = rowBase + (e / 32) * 18L;
            int j = e % 32;
            int q = j < 16 ? u8(s, bo + 2 + j) & 0xF : u8(s, bo + 2 + j - 16) >> 4;
            return f16(s, bo) * (q - 8);
        }
        if (dt == DataType.MXFP4) {
            long bo = rowBase + (e / 32) * 17L;
            int j = e % 32;
            int q = j < 16 ? u8(s, bo + 1 + j) & 0xF : u8(s, bo + 1 + j - 16) >> 4;
            return Math.scalb(new float[] {0, .5f, 1, 1.5f, 2, 3, 4, 6}[q & 7], u8(s, bo) - 127)
                    * (q < 8 ? 1 : -1);
        }
        long bo = rowBase + (e / 256) * dt.byteSize();
        int ee = e % 256;
        if (dt == DataType.Q6_K) {
            int hh = ee / 128, c = ee % 128 / 32, l = ee % 32;
            int ql = u8(s, bo + 64 * hh + (c == 1 || c == 3 ? 32 : 0) + l);
            int lo = c < 2 ? ql & 0xF : ql >> 4;
            int q = (lo | ((u8(s, bo + 128 + 32 * hh + l) >> (2 * c)) & 3) << 4) - 32;
            int sc = s.get(JAVA_BYTE, bo + 192 + ee / 16); // signed
            return f16(s, bo + 208) * (sc * q);
        }
        boolean q5 = dt == DataType.Q5_K;
        int gg = ee / 64, h = ee % 64 / 32, l = ee % 32, is = 2 * gg + h;
        int qb = u8(s, bo + (q5 ? 48 : 16) + gg * 32 + l);
        int q = h == 0 ? qb & 0xF : qb >> 4;
        if (q5) q |= ((u8(s, bo + 16 + l) >> (2 * gg + h)) & 1) << 4;
        int sc, mn; // get_scale_min_k4
        if (is < 4) {
            sc = u8(s, bo + 4 + is) & 63;
            mn = u8(s, bo + 8 + is) & 63;
        } else {
            sc = (u8(s, bo + 8 + is) & 0xF) | (u8(s, bo + is) >> 6) << 4;
            mn = (u8(s, bo + 8 + is) >> 4) | (u8(s, bo + 4 + is) >> 6) << 4;
        }
        return f16(s, bo) * (sc * q) - f16(s, bo + 2) * mn;
    }

    // ---- packed reader: the jam.h JAM_PACK_ABI 1 layout comment, element at a time ----

    private static float packedValue(
            MemorySegment s, long tensorBase, DataType dt, int k, int row, int e) {
        int g = row / 4, r = row % 4, b = e / 32, j = e % 32, B = e / 256;
        long nb = k / 32, sb = k / 256;
        long go = tensorBase + g * groupBytes(dt, k);
        if (dt == DataType.Q4_0) {
            long line = go + 64L * b + 16L * r;
            int q = j < 16 ? u8(s, line + j) & 0xF : u8(s, line + j - 16) >> 4;
            return f32(s, go + 64 * nb + 16L * b + 4L * r) * (q - 8);
        }
        if (dt == DataType.MXFP4) {
            long block = go + 68L * b;
            long qb = block + 4 + (j & 15) / 4 * 16L + r * 4L + (j & 3);
            int q = j < 16 ? u8(s, qb) & 0xF : u8(s, qb) >> 4;
            return Math.scalb(
                            new float[] {0, .5f, 1, 1.5f, 2, 3, 4, 6}[q & 7],
                            u8(s, block + r) - 127)
                    * (q < 8 ? 1 : -1);
        }
        if (dt == DataType.Q4_K) {
            long line = go + 64L * b + 16L * r, smo = go + 64 * nb, ddo = go + 72 * nb;
            int q = j < 16 ? u8(s, line + j) & 0xF : u8(s, line + j - 16) >> 4;
            int sc = u8(s, smo + 8L * b + r), mn = u8(s, smo + 8L * b + 4 + r);
            return f32(s, ddo + 32L * B + 4L * r) * (sc * q)
                    - f32(s, ddo + 32L * B + 16 + 4L * r) * mn;
        }
        if (dt == DataType.Q5_K) {
            long smo = go + 128 * nb, ddo = go + 136 * nb;
            int q = u8(s, go + 128L * b + 32L * r + j); // q5 stored 0..31
            int sc = u8(s, smo + 8L * b + r), mn = u8(s, smo + 8L * b + 4 + r);
            return f32(s, ddo + 32L * B + 4L * r) * (sc * q)
                    - f32(s, ddo + 32L * B + 16 + 4L * r) * mn;
        }
        long sco = go + 128 * nb, ddo = sco + 8 * nb;
        int q = s.get(JAVA_BYTE, go + 128L * b + 32L * r + j); // stored q6 - 32, signed
        int sc = s.get(JAVA_BYTE, sco + 8L * b + (j < 16 ? r : 4 + r)); // signed
        return f32(s, ddo + 16L * B + 4L * r) * (sc * q);
    }

    private static int u8(MemorySegment s, long off) {
        return s.get(JAVA_BYTE, off) & 0xFF;
    }

    private static float f16(MemorySegment s, long off) {
        return Float.float16ToFloat(s.get(JAVA_SHORT_UNALIGNED, off));
    }

    private static float f32(MemorySegment s, long off) {
        return s.get(JAVA_FLOAT_UNALIGNED, off);
    }
}
