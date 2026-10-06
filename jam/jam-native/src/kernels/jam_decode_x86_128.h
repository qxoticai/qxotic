/* 128-bit x86 (SSE3) weight DECODER + reduce, for the jam_gemm_q128 engine. A *true* SSE3 floor: no SSSE3
 * (pmaddubsw/pabsb/psignb) and no F16C - so weights decode to two __m128i int8 halves plus a SOFTWARE-
 * converted float scale, and the engine does sign-extend + madd. One decoder per quant (Q8_0, Q4_0, MXFP4). */
#ifndef JAM_DECODE_X86_128_H
#define JAM_DECODE_X86_128_H

#include <pmmintrin.h>   /* SSE3 (_mm_hadd_ps); pulls in SSE/SSE2 */
#include <stdint.h>
#include "jam_mxfp4.h"   /* jam_mxfp4_blk, jam_mxfp4_dhalf, JAM_MXFP4_CODES (pure C) */
#include "jam_fp16.h"    /* jam_half2float (shared software fp16->fp32; SSE3 has no F16C) */

typedef struct __attribute__((packed)) { uint16_t d; int8_t qs[32]; } jam_q8_blk;   /* Q8_0: 34 bytes */
typedef struct __attribute__((packed)) { uint16_t d; uint8_t qs[16]; } jam_q4_0_blk; /* Q4_0: 18 bytes */

/* signed int8 -> int16 (low / high 8 lanes): interleave each byte with its sign mask; dot of 16+16 int8
 * via madd. Shared by the q128 engine and the SSE3 K-quant kernels. Not unpack(x,x) + srai 8: clang reads
 * that as a sext whose low bytes are dead and unpacks into whatever register is free, a false dependency
 * that chains every dot of the tile into one serial path (2.5x slower than gcc's copy + unpack). */
#define JAM_SEXT_LO(x) _mm_unpacklo_epi8((x), _mm_cmpgt_epi8(_mm_setzero_si128(), (x)))
#define JAM_SEXT_HI(x) _mm_unpackhi_epi8((x), _mm_cmpgt_epi8(_mm_setzero_si128(), (x)))
#define JAM_DOT16(w, a) _mm_add_epi32(_mm_madd_epi16(JAM_SEXT_LO(w), JAM_SEXT_LO(a)), \
                                      _mm_madd_epi16(JAM_SEXT_HI(w), JAM_SEXT_HI(a)))

/* Q8_0: weights are already int8 (two 16-byte halves); scale is the fp16 block delta. */
static inline void jam_decode_q8_0_128(const void* blk, __m128i* wlo, __m128i* whi, float* dW) {
    const jam_q8_blk* w = (const jam_q8_blk*) blk;
    *wlo = _mm_loadu_si128((const __m128i*) w->qs);
    *whi = _mm_loadu_si128((const __m128i*) (w->qs + 16));
    *dW  = jam_half2float(w->d);
}

/* Q4_0: value = d·(nibble-8). Nibble decode is pure arithmetic (and/shift/sub) - no SSSE3 pshufb needed:
 * low nibbles are elements 0..15, high nibbles 16..31. */
static inline void jam_decode_q4_0_128(const void* blk, __m128i* wlo, __m128i* whi, float* dW) {
    const jam_q4_0_blk* w = (const jam_q4_0_blk*) blk;
    const __m128i m4 = _mm_set1_epi8(0x0F), e8 = _mm_set1_epi8(8);
    __m128i qs = _mm_loadu_si128((const __m128i*) w->qs);
    *wlo = _mm_sub_epi8(_mm_and_si128(qs, m4), e8);                          /* elements 0..15 */
    *whi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qs, 4), m4), e8);       /* elements 16..31 */
    *dW  = jam_half2float(w->d);
}

/* Q5_0: value = d·(q-16), q = nibble | 5th bit (bit j of qh for element j, bit j+16 for element j+16). SSE3 has
 * no pshufb, so the bit bytes are spread with unpacks: [b0 x8 | b1 x8] for elements 0..15, [b2 x8 | b3 x8]
 * for 16..31; and+cmpeq then tests bit j%8 of each (a scalar per-element spread ran at half the speed). */
typedef struct __attribute__((packed)) { uint16_t d; uint32_t qh; uint8_t qs[16]; } jam_q5_0_blk; /* 22 bytes */
static inline void jam_decode_q5_0_128(const void* blk, __m128i* wlo, __m128i* whi, float* dW) {
    const jam_q5_0_blk* w = (const jam_q5_0_blk*) blk;
    const __m128i m4 = _mm_set1_epi8(0x0F), e16 = _mm_set1_epi8(16), b4 = _mm_set1_epi8(0x10);
    const __m128i bit = _mm_set1_epi64x((long long) 0x8040201008040201ull);   /* byte j: 1 << (j % 8) */
    uint32_t qh; __builtin_memcpy(&qh, &w->qh, 4);
    __m128i h = _mm_cvtsi32_si128((int) qh);
    h = _mm_unpacklo_epi8(h, h);
    h = _mm_unpacklo_epi16(h, h);                                    /* b0 x4, b1 x4, b2 x4, b3 x4 */
    __m128i hl = _mm_unpacklo_epi32(h, h), hh = _mm_unpackhi_epi32(h, h);
    hl = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hl, bit), bit), b4);
    hh = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hh, bit), bit), b4);
    __m128i qs = _mm_loadu_si128((const __m128i*) w->qs);
    *wlo = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(qs, m4), hl), e16);
    *whi = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(qs, 4), m4), hh), e16);
    *dW  = jam_half2float(w->d);
}

/* MXFP4: nibble -> int8 code (FP4 value ×2, JAM_MXFP4_CODES); the ×½ folds into the scale (jam_mxfp4_dhalf).
 * True SSE3 has no pshufb for the LUT, so the code is computed: magnitude m = nibble & 7 maps to
 * {0,1,2,3,4,6,8,12} = m + max(m-4, 0) + (m == 7 ? 2 : 0), then bit 3 negates. Exact, and replaces a
 * scalar per-nibble table walk that clang ran 30% slower than gcc. */
static inline __m128i jam_mxfp4_codes_128(__m128i nib) {
    const __m128i m7 = _mm_set1_epi8(7), c4 = _mm_set1_epi8(4), c2 = _mm_set1_epi8(2), c8 = _mm_set1_epi8(8);
    __m128i m = _mm_and_si128(nib, m7);
    __m128i v = _mm_add_epi8(_mm_add_epi8(m, _mm_subs_epu8(m, c4)), _mm_and_si128(_mm_cmpeq_epi8(m, m7), c2));
    __m128i neg = _mm_cmpeq_epi8(_mm_and_si128(nib, c8), c8);      /* 0xFF where the sign bit is set */
    return _mm_sub_epi8(_mm_xor_si128(v, neg), neg);
}
static inline void jam_decode_mxfp4_128(const void* blk, __m128i* wlo, __m128i* whi, float* dW) {
    const jam_mxfp4_blk* w = (const jam_mxfp4_blk*) blk;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i qs = _mm_loadu_si128((const __m128i*) w->qs);
    *wlo = jam_mxfp4_codes_128(_mm_and_si128(qs, m4));                    /* elements 0..15 */
    *whi = jam_mxfp4_codes_128(_mm_and_si128(_mm_srli_epi16(qs, 4), m4)); /* elements 16..31 */
    *dW  = jam_mxfp4_dhalf(w->e);
}

/* horizontal sum of 4 packed floats via SSE3 haddps (two rounds). */
static inline float jam_hsum4_sse3(__m128 v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

#endif /* JAM_DECODE_X86_128_H */
