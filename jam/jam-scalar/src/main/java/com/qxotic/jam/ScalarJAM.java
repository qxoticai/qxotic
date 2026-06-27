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
 * <p>Implemented dtypes: {@code F32 F16 BF16 Q4_0 Q8_0}. The k-quants ({@code Q4_K/Q5_K/Q6_K}) and FP4
 * ({@code MXFP4/NVFP4}) decode is the next step — they currently return {@link #EUNSUPPORTED}.
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
            case F32, F16, BF16, Q4_0, Q8_0 -> true;
            default -> false;   // Q4_K/Q5_K/Q6_K/MXFP4/NVFP4 decode: TODO
        };
    }

    /** Decode element {@code l} of a weight row whose block-0 starts at byte {@code rowBase}. */
    private static float decode(MemorySegment w, long rowBase, GGMLType t, int l) {
        switch (t) {
            case F32:
                return w.get(JAVA_FLOAT_UNALIGNED, rowBase + (long) l * Float.BYTES);
            case F16:
                return f16ToFloat(w.get(JAVA_SHORT_UNALIGNED, rowBase + (long) l * 2));
            case BF16:
                return Float.intBitsToFloat((w.get(JAVA_SHORT_UNALIGNED, rowBase + (long) l * 2) & 0xFFFF) << 16);
            case Q8_0: {                                              // block: f16 d, int8 qs[32]
                long blk = rowBase + (long) (l >> 5) * 34;
                float d = f16ToFloat(w.get(JAVA_SHORT_UNALIGNED, blk));
                return d * w.get(JAVA_BYTE, blk + 2 + (l & 31));
            }
            case Q4_0: {                                              // block: f16 d, nibble qs[16]; v = d·(q-8)
                long blk = rowBase + (long) (l >> 5) * 18;
                float d = f16ToFloat(w.get(JAVA_SHORT_UNALIGNED, blk));
                int within = l & 31;
                int b = w.get(JAVA_BYTE, blk + 2 + (within & 15)) & 0xFF;
                int q = within < 16 ? (b & 0xF) : (b >> 4);
                return d * (q - 8);
            }
            default:
                throw new IllegalStateException(t.toString());       // decodable() guards this
        }
    }

    /** IEEE half -> float. */
    static float f16ToFloat(short h) {
        int x = h & 0xFFFF, sign = (x & 0x8000) << 16, exp = (x >> 10) & 0x1F, man = x & 0x3FF;
        int bits;
        if (exp == 0) {
            if (man == 0) bits = sign;                               // +/- 0
            else { int e = 113; while ((man & 0x400) == 0) { man <<= 1; e--; } man &= 0x3FF; bits = sign | (e << 23) | (man << 13); }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000 | (man << 13);                  // inf / nan
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
        }
        return Float.intBitsToFloat(bits);
    }
}
