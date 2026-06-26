/* 256-bit x86 weight DECODERS (per quant) + shared reduce, for the jam_gemm_q256 engine.
 * A decoder turns one weight block into 32 int8 weight values (__m256i) plus a float block scale; the
 * engine then does abs/sign + the per-ISA int8 dot. One decoder per quant — reused across every ISA. */
#ifndef JAM_DECODE_X86_256_H
#define JAM_DECODE_X86_256_H

#include <immintrin.h>
#include <stdint.h>
#include "jam_mxfp4.h"

typedef struct __attribute__((packed)) { uint16_t d; int8_t qs[32]; } jam_q8_blk;   /* Q8_0: 34 bytes */

/* Q8_0: weights are already int8; scale is the fp16 block delta. */
static inline void jam_decode_q8_0_256(const void* blk, __m256i* wq, float* dW) {
    const jam_q8_blk* w = (const jam_q8_blk*) blk;
    *wq = _mm256_loadu_si256((const __m256i*) w->qs);
    *dW = _cvtsh_ss(w->d);
}

/* Q4_0: { fp16 d; nibble qs[16] } = 18 bytes. value = d·(nibble-8); decode nibble->int8 (nibble-8). */
typedef struct __attribute__((packed)) { uint16_t d; uint8_t qs[16]; } jam_q4_0_blk;
static inline void jam_decode_q4_0_256(const void* blk, __m256i* wq, float* dW) {
    const jam_q4_0_blk* w = (const jam_q4_0_blk*) blk;
    const __m128i m4 = _mm_set1_epi8(0x0F), e8 = _mm_set1_epi8(8);
    __m128i qs = _mm_loadu_si128((const __m128i*) w->qs);
    __m128i lo = _mm_sub_epi8(_mm_and_si128(qs, m4), e8);                 /* elements 0..15 */
    __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qs, 4), m4), e8); /* elements 16..31 */
    *wq = _mm256_set_m128i(hi, lo);
    *dW = _cvtsh_ss(w->d);
}

/* MXFP4: decode FP4 nibbles -> int8 (value×2) via one shuffle; scale folds in the ×½. qs[j] low nibble
 * is element j, high nibble element j+16 -> lo|hi halves match the int8 element order 0..31. */
static inline void jam_decode_mxfp4_256(const void* blk, __m256i* wq, float* dW) {
    const jam_mxfp4_blk* w = (const jam_mxfp4_blk*) blk;
    const __m128i lut = _mm_setr_epi8(JAM_MXFP4_CODES);
    const __m128i m4  = _mm_set1_epi8(0x0F);
    __m128i qs = _mm_loadu_si128((const __m128i*) w->qs);
    __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(qs, m4));
    __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(qs, 4), m4));
    *wq = _mm256_set_m128i(hi, lo);
    *dW = jam_mxfp4_dhalf(w->e);
}

static inline float jam_hsum8_256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
    return _mm_cvtss_f32(s);
}

#endif /* JAM_DECODE_X86_256_H */
