package com.qxotic.jam;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;

/**
 * Pure-Java reference {@link JAM}: a {@code dot()}-based matmul that decodes the quantized weight on the
 * fly, for any shape (prefill or decode). No native code, no Vector API — the portable floor and the
 * correctness reference every other backend is checked against. Activations and result are F32.
 *
 * <p>Offsets are BYTE offsets into the operand segments; {@code ldw/lda/ldr} are ELEMENT row strides
 * (the native convention). The weight is read block-by-block through {@link GGMLType}'s geometry.
 *
 * <p>Decodes every jam weight dtype: {@code F32 F16 BF16 Q4_0 Q8_0}, the k-quants {@code Q4_K/Q5_K/Q6_K},
 * and FP4 {@code MXFP4/NVFP4} — the dequant mirrors jam's native reference (jam_ref.h).
 */
public final class ScalarJAM implements JAM {

    @Override
    public int mm(MemorySegment w, long wOff, int wt, int ldw,
                  MemorySegment a, long aOff, int at, int lda,
                  MemorySegment r, long rOff, int rt, int ldr,
                  int m, int n, int k) {
        if (at != F32 || rt != F32) return EUNSUPPORTED;
        GGMLType t = GGMLType.byCode(wt);
        if (t == null || !decodable(t)) return EUNSUPPORTED;

        // C[token j][feature i] = Σ_l W[i][l] · A[j][l], written token-major: r[j·ldr + i].
        java.util.stream.IntStream.range(0, n * m).parallel().forEach(idx -> {
            int j = idx / m, i = idx - j * m;
            long wRow = wOff + t.rowBytes((long) i * ldw);            // byte base of weight row i
            long aRow = aOff + (long) j * lda * Float.BYTES;          // byte base of activation row j
            double sum = 0;
            for (int l = 0; l < k; l++) {
                sum += (double) decode(w, wRow, t, l) * a.get(JAVA_FLOAT_UNALIGNED, aRow + (long) l * Float.BYTES);
            }
            r.set(JAVA_FLOAT_UNALIGNED, rOff + ((long) j * ldr + i) * Float.BYTES, (float) sum);
        });
        return OK;
    }

    private static boolean decodable(GGMLType t) {
        return switch (t) {
            case F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4 -> true;
            default -> false;
        };
    }

    /** Decode element {@code l} of a weight row whose block-0 starts at byte {@code rowBase}. */
    private static float decode(MemorySegment w, long rowBase, GGMLType t, int l) {
        switch (t) {
            case F32:
                return w.get(JAVA_FLOAT_UNALIGNED, rowBase + (long) l * Float.BYTES);
            case F16:
                return Float.float16ToFloat(w.get(JAVA_SHORT_UNALIGNED, rowBase + (long) l * 2));
            case BF16:
                return Float.intBitsToFloat((w.get(JAVA_SHORT_UNALIGNED, rowBase + (long) l * 2) & 0xFFFF) << 16);
            case Q8_0: {                                              // block: f16 d, int8 qs[32]
                long blk = rowBase + (long) (l >> 5) * 34;
                float d = Float.float16ToFloat(w.get(JAVA_SHORT_UNALIGNED, blk));
                return d * w.get(JAVA_BYTE, blk + 2 + (l & 31));
            }
            case Q4_0: {                                              // block: f16 d, nibble qs[16]; v = d·(q-8)
                long blk = rowBase + (long) (l >> 5) * 18;
                float d = Float.float16ToFloat(w.get(JAVA_SHORT_UNALIGNED, blk));
                int within = l & 31;
                int b = w.get(JAVA_BYTE, blk + 2 + (within & 15)) & 0xFF;
                int q = within < 16 ? (b & 0xF) : (b >> 4);
                return d * (q - 8);
            }
            case Q4_K: {                                             // 144B: f16 d, f16 dmin, scales[12], qs[128]
                long blk = rowBase + (long) (l / 256) * 144;
                int within = l % 256;
                float d = Float.float16ToFloat(w.get(JAVA_SHORT_UNALIGNED, blk));
                float dmin = Float.float16ToFloat(w.get(JAVA_SHORT_UNALIGNED, blk + 2));
                int g = within / 64, half = (within % 64) / 32, e = within % 32, scIdx = g * 2 + half;
                long sm = scaleMinK4(w, blk + 4, scIdx);
                int qb = w.get(JAVA_BYTE, blk + 16 + (long) g * 32 + e) & 0xFF;
                int nib = half == 0 ? (qb & 0xF) : (qb >> 4);
                return d * (int) (sm >> 8) * nib - dmin * (int) (sm & 0xFF);
            }
            case Q5_K: {                                             // 176B: d, dmin, scales[12], qh[32], qs[128]
                long blk = rowBase + (long) (l / 256) * 176;
                int within = l % 256;
                float d = Float.float16ToFloat(w.get(JAVA_SHORT_UNALIGNED, blk));
                float dmin = Float.float16ToFloat(w.get(JAVA_SHORT_UNALIGNED, blk + 2));
                int g = within / 64, half = (within % 64) / 32, e = within % 32, scIdx = g * 2 + half;
                long sm = scaleMinK4(w, blk + 4, scIdx);
                int qs = w.get(JAVA_BYTE, blk + 48 + (long) g * 32 + e) & 0xFF;
                int qh = w.get(JAVA_BYTE, blk + 16 + e) & 0xFF;            // 5th bit: one byte per e, bit 2g / 2g+1
                int q5 = half == 0 ? ((qs & 0xF) | (((qh >> (2 * g)) & 1) << 4))
                                   : ((qs >> 4)  | (((qh >> (2 * g + 1)) & 1) << 4));
                return d * (int) (sm >> 8) * q5 - dmin * (int) (sm & 0xFF);
            }
            case Q6_K: {                                             // 210B: ql[128], qh[64], s8 scales[16], f16 d
                long blk = rowBase + (long) (l / 256) * 210;
                int within = l % 256;
                int h = within / 128, w128 = within % 128, j = w128 / 32, ll = w128 % 32;
                float d = Float.float16ToFloat(w.get(JAVA_SHORT_UNALIGNED, blk + 208));
                int qlIdx = h * 64 + ((j == 1 || j == 3) ? 32 + ll : ll);
                int qlByte = w.get(JAVA_BYTE, blk + qlIdx) & 0xFF;
                int qlnib = j < 2 ? (qlByte & 0xF) : (qlByte >> 4);
                int qhByte = w.get(JAVA_BYTE, blk + 128 + (long) h * 32 + ll) & 0xFF;
                int qv = qlnib | (((qhByte >> (2 * j)) & 3) << 4);
                int sc = w.get(JAVA_BYTE, blk + 192 + h * 8 + j * 2 + ll / 16);   // signed int8 scale
                return d * sc * (qv - 32);
            }
            case MXFP4: {                                            // 17B: e8m0 e, fp4 qs[16]; 32 elems
                long blk = rowBase + (long) (l / 32) * 17;
                int within = l % 32;
                float dhalf = mxfp4Dhalf(w.get(JAVA_BYTE, blk) & 0xFF);
                int qb = w.get(JAVA_BYTE, blk + 1 + (within & 15)) & 0xFF;
                int nib = within < 16 ? (qb & 0xF) : (qb >> 4);
                return dhalf * FP4_KV[nib];
            }
            case NVFP4: {                                            // 36B: ue4m3 d[4], fp4 qs[32]; 64 elems, per-16 scale
                long blk = rowBase + (long) (l / 64) * 36;
                int within = l % 64, s = within / 16, w16 = within % 16;
                float scale = ue4m3ToFloat(w.get(JAVA_BYTE, blk + s) & 0xFF);
                int qb = w.get(JAVA_BYTE, blk + 4 + (long) s * 8 + (w16 & 7)) & 0xFF;
                int nib = w16 < 8 ? (qb & 0xF) : (qb >> 4);
                return FP4_KV[nib] * scale;
            }
            default:
                throw new IllegalStateException(t.toString());       // decodable() guards this
        }
    }

    /** E2M1 FP4 code -> signed magnitude, shared by MXFP4 + NVFP4 (== jam_ref kv[16]). */
    private static final int[] FP4_KV = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    /** GGML get_scale_min_k4: unpack the j-th 6-bit scale + min from the packed scales[12] at {@code base};
     *  returned as {@code (sc << 8) | mn} (both 0..63). Mirrors the packing in jam_ref make_q4k/make_q5k. */
    private static long scaleMinK4(MemorySegment w, long base, int j) {
        int sc, mn;
        if (j < 4) {
            sc = (w.get(JAVA_BYTE, base + j) & 0xFF) & 63;
            mn = (w.get(JAVA_BYTE, base + j + 4) & 0xFF) & 63;
        } else {
            int bj4 = w.get(JAVA_BYTE, base + j + 4) & 0xFF;
            sc = (bj4 & 0xF) | (((w.get(JAVA_BYTE, base + j - 4) & 0xFF) >> 6) << 4);
            mn = (bj4 >> 4) | (((w.get(JAVA_BYTE, base + j)     & 0xFF) >> 6) << 4);
        }
        return ((long) sc << 8) | mn;
    }

    /** MXFP4 e8m0 scale code -> {@code 0.5·2^(e-127)} (== jam_mxfp4_dhalf). */
    private static float mxfp4Dhalf(int e) {
        int bits = (e == 0) ? 0x00400000 : (e << 23);
        return 0.5f * Float.intBitsToFloat(bits);
    }

    /** NVFP4 per-16 scale: UE4M3 code -> float, INCLUDING ggml_ue4m3_to_fp32's x0.5
     *  (kvalues_mxfp4 are 2x the E2M1 values; the halved scale compensates). */
    private static float ue4m3ToFloat(int x) {
        if (x == 0 || x == 0x7F) return 0f;
        int e = (x >> 3) & 0xF, m = x & 0x7;
        return 0.5f * (e != 0 ? Math.scalb(1f + m / 8f, e - 7) : Math.scalb((float) m, -9));
    }
}
