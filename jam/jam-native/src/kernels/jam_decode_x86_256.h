/* 256-bit x86 weight DECODERS (per quant) + shared reduce, for the jam_gemm_q256 engine.
 * A decoder turns one weight block into 32 int8 weight values (__m256i) plus its float block scale,
 * broadcast; the engine then does abs/sign + the per-ISA int8 dot. One decoder per quant - reused across
 * every ISA. */
#ifndef JAM_DECODE_X86_256_H
#define JAM_DECODE_X86_256_H

#include <immintrin.h>
#include <stdint.h>
#include "jam_mxfp4.h"

typedef struct __attribute__((packed)) { uint16_t d; int8_t qs[32]; } jam_q8_blk;   /* Q8_0: 34 bytes */

/* fp16 scale -> broadcast f32, converted as a vector. A scalar _cvtsh_ss lets clang load the half with a
 * merging vpinsrw into whatever register is free - in the decode loop that was the FMA accumulator, so
 * every block waited on the previous one (avx_vnni Q8_0 decode 3x slower than gcc). */
static inline __m256 jam_h2f_splat256(uint16_t h) { return _mm256_cvtph_ps(_mm_set1_epi16((short) h)); }

/* Q8_0: weights are already int8; scale is the fp16 block delta. */
static inline void jam_decode_q8_0_256(const void* blk, __m256i* wq, __m256* dW) {
    const jam_q8_blk* w = (const jam_q8_blk*) blk;
    *wq = _mm256_loadu_si256((const __m256i*) w->qs);
    *dW = jam_h2f_splat256(w->d);
}

/* The nibble quants decode both 16-element halves in ONE 256-bit pass: [qs | qs >> 4], then every mask,
 * shuffle and offset once on the ymm. Two 128-bit pipelines joined at the end cost gcc ~20% in decode;
 * clang merged them itself. */
static inline __m256i jam_nibbles_256(const uint8_t* qs) {   /* low nibbles = elements 0..15, high = 16..31 */
    const __m128i v = _mm_loadu_si128((const __m128i*) qs);
    return _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(v, 4), v), _mm256_set1_epi8(0x0F));
}

/* Q4_0: { fp16 d; nibble qs[16] } = 18 bytes. value = d·(nibble-8); decode nibble->int8 (nibble-8). */
typedef struct __attribute__((packed)) { uint16_t d; uint8_t qs[16]; } jam_q4_0_blk;
static inline void jam_decode_q4_0_256(const void* blk, __m256i* wq, __m256* dW) {
    const jam_q4_0_blk* w = (const jam_q4_0_blk*) blk;
    *wq = _mm256_sub_epi8(jam_nibbles_256(w->qs), _mm256_set1_epi8(8));
    *dW = jam_h2f_splat256(w->d);
}

/* Q5_0: { fp16 d; uint32 qh; nibble qs[16] } = 22 bytes. value = d·(q-16), q = nibble | (5th bit from qh):
 * bit j of qh is element j's high bit. Lane 0 holds qh, lane 1 qh >> 16, so in each 128-bit lane bit j
 * becomes byte j in three ops: pshufb copies bit-byte j/8 into byte j, and+cmpeq tests bit j%8 there, and
 * lands it as 0x10. */
typedef struct __attribute__((packed)) { uint16_t d; uint32_t qh; uint8_t qs[16]; } jam_q5_0_blk;
static inline void jam_decode_q5_0_256(const void* blk, __m256i* wq, __m256* dW) {
    const jam_q5_0_blk* w = (const jam_q5_0_blk*) blk;
    const __m256i shuf = _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1);
    const __m256i mask = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    uint32_t qh; __builtin_memcpy(&qh, &w->qh, 4);
    const __m128i h = _mm_cvtsi32_si128((int) qh);
    __m256i b = _mm256_shuffle_epi8(_mm256_set_m128i(_mm_srli_epi32(h, 16), h), shuf);
    __m256i hi = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(b, mask), mask), _mm256_set1_epi8(0x10));
    *wq = _mm256_sub_epi8(_mm256_or_si256(jam_nibbles_256(w->qs), hi), _mm256_set1_epi8(16));
    *dW = jam_h2f_splat256(w->d);
}

/* MXFP4: decode FP4 nibbles -> int8 (value×2) via one shuffle; scale folds in the ×½. qs[j] low nibble
 * is element j, high nibble element j+16 -> lo|hi halves match the int8 element order 0..31. */
static inline void jam_decode_mxfp4_256(const void* blk, __m256i* wq, __m256* dW) {
    const jam_mxfp4_blk* w = (const jam_mxfp4_blk*) blk;
    const __m256i lut = _mm256_setr_epi8(JAM_MXFP4_CODES, JAM_MXFP4_CODES);
    *wq = _mm256_shuffle_epi8(lut, jam_nibbles_256(w->qs));
    *dW = _mm256_set1_ps(jam_mxfp4_dhalf(w->e));
}

static inline float jam_hsum8_256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
    return _mm_cvtss_f32(s);
}

#endif /* JAM_DECODE_X86_256_H */
