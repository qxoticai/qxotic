package com.qxotic.jam.scalar;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;

import com.oracle.svm.shared.AlwaysInline;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * A row decoder: a span of one weight row, starting on a block boundary, to F32. The span's raw
 * bytes are staged into a Java array with one bulk copy and decoded from there, so the element
 * loops are plain array code: no per-element {@code MemorySegment} accessor, whose inlining is the
 * first thing C2 drops once a caller grows large (measured: 8x slower decode inside the gemm than
 * in isolation). Values mirror jam's native reference (jam_ref.h) exactly.
 *
 * <p>One instance per worker: it owns the staging buffers.
 */
final class Decode {

    private byte[] raw = new byte[0];
    private short[] raw16 = new short[0];

    /**
     * Decodes {@code count} elements of row {@code row} of {@code w} from element {@code from} into
     * {@code out[at .. at+count)}, then zeroes {@code out[at+count .. at+pad)}. {@code from} and
     * {@code count} are multiples of the dtype's block size.
     */
    void row(Weight w, int row, int from, int count, int pad, float[] out, int at) {
        MemorySegment seg = w.seg();
        long src = w.row(row) + w.type().rowBytes(from);
        switch (w.type()) {
            case F32 -> MemorySegment.copy(seg, JAVA_FLOAT_UNALIGNED, src, out, at, count);
            case F16 -> f16(stage16(seg, src, count), count, out, at);
            case BF16 -> bf16(stage16(seg, src, count), count, out, at);
            case Q8_0 -> q8_0(stage(w, src, count), count, out, at);
            case Q4_0 -> q4_0(stage(w, src, count), count, out, at);
            case Q4_K -> q4_K(stage(w, src, count), count, out, at);
            case Q5_K -> q5_K(stage(w, src, count), count, out, at);
            case Q6_K -> q6_K(stage(w, src, count), count, out, at);
            case MXFP4 -> mxfp4(stage(w, src, count), count, out, at);
            case NVFP4 -> nvfp4(stage(w, src, count), count, out, at);
            case Q1_0 -> q1_0(stage(w, src, count), count, out, at);
        }
        Arrays.fill(out, at + count, at + pad, 0f);
    }

    private byte[] stage(Weight w, long src, int count) {
        int bytes = (int) w.type().rowBytes(count);
        if (raw.length < bytes) raw = new byte[bytes];
        MemorySegment.copy(w.seg(), JAVA_BYTE, src, raw, 0, bytes);
        return raw;
    }

    private short[] stage16(MemorySegment seg, long src, int count) {
        if (raw16.length < count) raw16 = new short[count];
        MemorySegment.copy(seg, JAVA_SHORT_UNALIGNED, src, raw16, 0, count);
        return raw16;
    }

    static void f16(short[] h, int count, float[] out, int at) {
        for (int i = 0; i < count; i++) out[at + i] = Float.float16ToFloat(h[i]);
    }

    static void bf16(short[] h, int count, float[] out, int at) {
        for (int i = 0; i < count; i++) out[at + i] = Float.intBitsToFloat(h[i] << 16);
    }

    /** Block (34 B): f16 d, int8 qs[32]. */
    static void q8_0(byte[] q, int count, float[] out, int at) {
        for (int o = at, p = 0, end = at + count; o < end; o += 32, p += 34) {
            float d = f16(q, p);
            int s = p + 2;
            for (int e = 0; e < 32; e++) out[o + e] = q[s + e] * d;
        }
    }

    /** Block (18 B): f16 d, nibbles qs[16]; low nibbles are elements 0..15; v = d(q-8). */
    static void q4_0(byte[] q, int count, float[] out, int at) {
        for (int o = at, p = 0, end = at + count; o < end; o += 32, p += 18) {
            float d = f16(q, p);
            int s = p + 2, hi = o + 16;
            for (int e = 0; e < 16; e++) {
                int b = q[s + e] & 0xFF;
                out[o + e] = ((b & 0xF) - 8) * d;
                out[hi + e] = ((b >> 4) - 8) * d;
            }
        }
    }

    /**
     * Super-block (144 B, 256 elements): f16 d, f16 dmin, packed 6-bit scales/mins[12], nibbles
     * qs[128]. Four 64-element groups of 32 bytes; low nibbles are the group's first 32 elements
     * (scale pair 2g), high nibbles the last 32 (pair 2g+1); v = d sc q - dmin mn.
     */
    static void q4_K(byte[] q, int count, float[] out, int at) {
        for (int o = at, p = 0, end = at + count; o < end; o += 256, p += 144) {
            float d = f16(q, p), dmin = f16(q, p + 2);
            for (int g = 0; g < 4; g++) {
                int sl = scaleMin(q, p + 4, 2 * g), sh = scaleMin(q, p + 4, 2 * g + 1);
                float dl = d * (sl >> 8), ml = dmin * (sl & 0xFF);
                float dh = d * (sh >> 8), mh = dmin * (sh & 0xFF);
                int s = p + 16 + 32 * g, lo = o + 64 * g, hi = lo + 32;
                for (int e = 0; e < 32; e++) {
                    int b = q[s + e] & 0xFF;
                    out[lo + e] = (b & 0xF) * dl - ml;
                    out[hi + e] = (b >> 4) * dh - mh;
                }
            }
        }
    }

    /**
     * Super-block (176 B): as Q4_K plus qh[32] carrying the fifth bit - byte e holds bit 2g for the
     * group's low half and bit 2g+1 for its high half.
     */
    static void q5_K(byte[] q, int count, float[] out, int at) {
        for (int o = at, p = 0, end = at + count; o < end; o += 256, p += 176) {
            float d = f16(q, p), dmin = f16(q, p + 2);
            for (int g = 0; g < 4; g++) {
                int sl = scaleMin(q, p + 4, 2 * g), sh = scaleMin(q, p + 4, 2 * g + 1);
                float dl = d * (sl >> 8), ml = dmin * (sl & 0xFF);
                float dh = d * (sh >> 8), mh = dmin * (sh & 0xFF);
                int qh = p + 16, s = p + 48 + 32 * g, lo = o + 64 * g, hi = lo + 32, shift = 2 * g;
                for (int e = 0; e < 32; e++) {
                    int b = q[s + e] & 0xFF;
                    int h = (q[qh + e] & 0xFF) >> shift;
                    out[lo + e] = ((b & 0xF) | ((h & 1) << 4)) * dl - ml;
                    out[hi + e] = ((b >> 4) | ((h & 2) << 3)) * dh - mh;
                }
            }
        }
    }

    /**
     * Super-block (210 B): ql[128] nibbles, qh[64] 2-bit highs, int8 scales[16], f16 d. Two
     * 128-element halves; within a half, bytes {@code e} and {@code 32+e} of ql hold the low
     * nibbles of elements {@code e, 32+e} and the high nibbles of {@code 64+e, 96+e}, and byte
     * {@code e} of qh holds those four elements' high bit pairs, lowest pair first. Scales cover 16
     * elements each; v = d sc (q - 32). One pass reads each byte once.
     */
    static void q6_K(byte[] q, int count, float[] out, int at) {
        for (int o = at, p = 0, end = at + count; o < end; o += 256, p += 210) {
            float d = f16(q, p + 208);
            for (int h = 0; h < 2; h++) {
                int ql = p + 64 * h, qh = p + 128 + 32 * h, sc = p + 192 + 8 * h;
                int base = o + 128 * h;
                for (int half = 0; half < 2; half++) {
                    float s0 = d * q[sc + half], s1 = d * q[sc + 2 + half];
                    float s2 = d * q[sc + 4 + half], s3 = d * q[sc + 6 + half];
                    for (int e = 16 * half, last = e + 16; e < last; e++) {
                        int a = q[ql + e] & 0xFF, b = q[ql + 32 + e] & 0xFF, hi = q[qh + e] & 0xFF;
                        out[base + e] = (((a & 0xF) | ((hi & 3) << 4)) - 32) * s0;
                        out[base + 32 + e] = (((b & 0xF) | ((hi & 12) << 2)) - 32) * s1;
                        out[base + 64 + e] = (((a >> 4) | (hi & 48)) - 32) * s2;
                        out[base + 96 + e] = (((b >> 4) | ((hi & 192) >> 2)) - 32) * s3;
                    }
                }
            }
        }
    }

    /** Block (17 B): e8m0 e, fp4 qs[16]; low nibbles first; v = 0.5 2^(e-127) kv[q]. */
    static void mxfp4(byte[] q, int count, float[] out, int at) {
        for (int o = at, p = 0, end = at + count; o < end; o += 32, p += 17) {
            int e8 = q[p] & 0xFF;
            float dhalf = 0.5f * Float.intBitsToFloat(e8 == 0 ? 0x00400000 : e8 << 23);
            int s = p + 1, hi = o + 16;
            for (int e = 0; e < 16; e++) {
                int b = q[s + e] & 0xFF;
                out[o + e] = FP4[b & 0xF] * dhalf;
                out[hi + e] = FP4[b >> 4] * dhalf;
            }
        }
    }

    /** Block (36 B, 64 elements): ue4m3 d[4], fp4 qs[32]; per 16 elements one scale, 8 bytes. */
    static void nvfp4(byte[] q, int count, float[] out, int at) {
        for (int o = at, p = 0, end = at + count; o < end; o += 64, p += 36) {
            for (int g = 0; g < 4; g++) {
                float scale = ue4m3(q[p + g] & 0xFF);
                int s = p + 4 + 8 * g, lo = o + 16 * g, hi = lo + 8;
                for (int e = 0; e < 8; e++) {
                    int b = q[s + e] & 0xFF;
                    out[lo + e] = FP4[b & 0xF] * scale;
                    out[hi + e] = FP4[b >> 4] * scale;
                }
            }
        }
    }

    /** Block (18 B, 128 elements): f16 d, sign bits LSB-first; v = bit ? d : -d. */
    static void q1_0(byte[] q, int count, float[] out, int at) {
        for (int o = at, p = 0, end = at + count; o < end; o += 128, p += 18) {
            float d = f16(q, p);
            for (int e = 0; e < 16; e++) {
                int bits = q[p + 2 + e], to = o + 8 * e;
                for (int i = 0; i < 8; i++) out[to + i] = (((bits >> i) & 1) * 2 - 1) * d;
            }
        }
    }

    /** Little-endian f16 at {@code p}. */
    @AlwaysInline("one call per block inside every decoder")
    private static float f16(byte[] q, int p) {
        return Float.float16ToFloat((short) ((q[p] & 0xFF) | (q[p + 1] << 8)));
    }

    /** E2M1 FP4 code to signed magnitude (jam_ref kv[16]). */
    private static final float[] FP4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    /**
     * GGML get_scale_min_k4: the 6-bit scale and min of sub-block {@code j} from the packed 12
     * bytes at {@code base}, as {@code (scale << 8) | min}.
     */
    @AlwaysInline("eight calls per k-quant super-block")
    private static int scaleMin(byte[] q, int base, int j) {
        if (j < 4) return ((q[base + j] & 63) << 8) | (q[base + j + 4] & 63);
        int b = q[base + j + 4] & 0xFF;
        int sc = (b & 0xF) | (((q[base + j - 4] & 0xFF) >> 6) << 4);
        int mn = (b >> 4) | (((q[base + j] & 0xFF) >> 6) << 4);
        return (sc << 8) | mn;
    }

    /** NVFP4 per-16 scale: UE4M3 code to float (ggml_ue4m3_to_fp32, bit 7 ignored). */
    @AlwaysInline("one call per 16 elements")
    private static float ue4m3(int x) {
        if (x == 0 || x == 0x7F) return 0f;
        int e = (x >> 3) & 0xF, m = x & 0x7;
        return e != 0 ? Math.scalb(1f + m / 8f, e - 7) : Math.scalb((float) m, -9);
    }
}
