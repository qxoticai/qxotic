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

/* signed int8 -> int16 (low / high 8 lanes): duplicate each byte then arithmetic-shift right 8 to
 * sign-extend; dot of 16+16 int8 via madd. Shared by the q128 engine and the SSE3 K-quant kernels. */
#define JAM_SEXT_LO(x) _mm_srai_epi16(_mm_unpacklo_epi8((x), (x)), 8)
#define JAM_SEXT_HI(x) _mm_srai_epi16(_mm_unpackhi_epi8((x), (x)), 8)
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

/* MXFP4: nibble -> int8 code (FP4 value ×2); the ×½ folds into the scale (jam_mxfp4_dhalf). The LUT lookup
 * is SSSE3 pshufb on the wider kernels - true SSE3 has none, so decode the 32 nibbles SCALARLY here (done
 * ONCE per weight block, amortized across the engine's 4-column tile), then the SSE int8 dot does the rest. */
static inline void jam_decode_mxfp4_128(const void* blk, __m128i* wlo, __m128i* whi, float* dW) {
    const jam_mxfp4_blk* w = (const jam_mxfp4_blk*) blk;
    static const int8_t lut[16] = { JAM_MXFP4_CODES };
    int8_t lo[16], hi[16];
    for (int j = 0; j < 16; ++j) { uint8_t b = w->qs[j]; lo[j] = lut[b & 0x0F]; hi[j] = lut[b >> 4]; }
    *wlo = _mm_loadu_si128((const __m128i*) lo);   /* elements 0..15 */
    *whi = _mm_loadu_si128((const __m128i*) hi);   /* elements 16..31 */
    *dW  = jam_mxfp4_dhalf(w->e);
}

/* horizontal sum of 4 packed floats via SSE3 haddps (two rounds). */
static inline float jam_hsum4_sse3(__m128 v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

#endif /* JAM_DECODE_X86_128_H */
