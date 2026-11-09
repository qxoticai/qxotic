/* The low-bit codes formats: Q2_0 and the PrismML fork's PQ2_0 and PTQ1_0. A block is a scale d and
 * one small integer code per element; value = code * d. Every kernel decodes a block to s8 codes, then
 * runs the int8 dot.
 *
 * Q2_0 (GGML_TYPE_Q2_0 = 42): { fp16 d; uint8_t qs[16]; } = 18 bytes, 64 elements; PQ2_0 (142): the
 *   same at 128 elements, { fp16 d; uint8_t qs[32]; } = 34 bytes. 2-bit codes LSB-first (byte = elem/4,
 *   shift = 2 * (elem%4)); value = (code - 1) * d, so codes 0..3 are -1, 0, +1, +2.
 * PTQ1_0 (143): { uint8_t qs[24]; uint8_t qh[2]; fp16 d; } = 28 bytes, 128 elements: TQ1_0's base-3
 *   packing. In a stage of c bytes (16 and 8 bytes of qs with 5 trits, then qh's 2 bytes with 4), element
 *   n*c + m is trit n of byte m, ((b * 3^n) & 255) * 3 >> 8; value = (trit - 1) * d. */
#ifndef JAM_LOWBIT_H
#define JAM_LOWBIT_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "jam.h"
#include "jam_fp16.h"

#define JAM_Q2_0_QK      64
#define JAM_Q2_0_BYTES   18
#define JAM_PQ2_0_QK     128
#define JAM_PQ2_0_BYTES  34
#define JAM_PTQ1_0_QK    128
#define JAM_PTQ1_0_BYTES 28
#define JAM_PTQ1_0_D     26    /* the scale sits after qs and qh */

/* Block bytes -> s8 codes. The row kernels decode each row once, but decode (n == 1) pays this per
 * token, so NEON gets SIMD versions: a table lookup puts byte e/4 in lane e and a per-lane shift brings
 * its code down; trits use NEON's byte multiply, -1 - (q >= 86) - (q >= 171) with compare masks of -1. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
static inline void jam_q2_codes(const uint8_t* blk, int qk, int8_t* out) {
    static const uint8_t lanes[2][16] = {{0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3},
                                         {4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7}};
    static const int8_t shift[16] = {0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6};
    const uint8x16_t lo = vld1q_u8(lanes[0]), hi = vld1q_u8(lanes[1]);
    const int8x16_t sh = vld1q_s8(shift);
    for (int j = 0; j < qk; j += 32) {                         /* 8 code bytes -> 32 codes */
        const uint8x8_t b = vld1_u8(blk + 2 + j / 4);
        const uint8x16_t src = vcombine_u8(b, b);
        vst1q_s8(out + j, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(src, lo), sh), vdupq_n_u8(3))), vdupq_n_s8(1)));
        vst1q_s8(out + j + 16, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(src, hi), sh), vdupq_n_u8(3))), vdupq_n_s8(1)));
    }
}
static inline int8x16_t jam_trits16(uint8x16_t b, uint8_t pow) {
    const uint8x16_t q = vmulq_u8(b, vdupq_n_u8(pow));
    const int8x16_t ge1 = vreinterpretq_s8_u8(vcgeq_u8(q, vdupq_n_u8(86)));
    const int8x16_t ge2 = vreinterpretq_s8_u8(vcgeq_u8(q, vdupq_n_u8(171)));
    return vsubq_s8(vsubq_s8(vdupq_n_s8(-1), ge1), ge2);
}
static inline void jam_ptq1_0_codes(const uint8_t* blk, int8_t* out) {
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    const uint8x16_t s16 = vld1q_u8(blk);                      /* stage of 16 bytes: elements 0..79 */
    const uint8x8_t s8 = vld1_u8(blk + 16);                    /* stage of 8 bytes: elements 80..119 */
    for (int n = 0; n < 5; n++) {
        vst1q_s8(out + n * 16, jam_trits16(s16, pow3[n]));
        vst1_s8(out + 80 + n * 8, vget_low_s8(jam_trits16(vcombine_u8(s8, s8), pow3[n])));
    }
    for (int n = 0; n < 4; n++)                                /* qh: elements 120..127 */
        for (int h = 0; h < 2; h++) {
            const uint8_t q = (uint8_t) (blk[24 + h] * pow3[n]);
            out[120 + n * 2 + h] = (int8_t) (((q * 3) >> 8) - 1);
        }
}
#else
static inline void jam_q2_codes(const uint8_t* blk, int qk, int8_t* out) {
    for (int j = 0; j < qk; j++) out[j] = (int8_t) (((blk[2 + j / 4] >> 2 * (j % 4)) & 3) - 1);
}
static inline void jam_ptq1_0_codes(const uint8_t* blk, int8_t* out) {
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    static const int stage[3][4] = { { 0, 16, 5, 0 }, { 16, 8, 5, 80 }, { 24, 2, 4, 120 } };  /* byte, c, trits, elem */
    for (int s = 0; s < 3; s++)
        for (int n = 0; n < stage[s][2]; n++)
            for (int m = 0; m < stage[s][1]; m++) {
                const uint8_t q = (uint8_t) (blk[stage[s][0] + m] * pow3[n]);
                out[stage[s][3] + n * stage[s][1] + m] = (int8_t) (((q * 3) >> 8) - 1);
            }
}
#endif

/* Per dtype (JAM_Q2_0, JAM_PQ2_0 or JAM_PTQ1_0): elements and bytes per block. */
static inline int jam_lowbit_qk(jam_dtype dt) { return dt == JAM_Q2_0 ? JAM_Q2_0_QK : JAM_PQ2_0_QK; }
static inline int jam_lowbit_bytes(jam_dtype dt) {
    return dt == JAM_Q2_0 ? JAM_Q2_0_BYTES : dt == JAM_PQ2_0 ? JAM_PQ2_0_BYTES : JAM_PTQ1_0_BYTES;
}

/* The row kernels' code buffer: per thread, grown on demand and reused, never freed (a malloc per call
 * paged in fresh memory every time: 2x slower decode on macOS). */
static inline int8_t* jam_lowbit_scratch(size_t bytes) {
    static _Thread_local int8_t* buf; static _Thread_local size_t cap;
    if (bytes > cap) { free(buf); buf = (int8_t*) malloc(bytes); cap = buf ? bytes : 0; }
    return buf;
}

/* A row of nblk blocks -> its k s8 codes and one scale per 32 elements (a block's scale repeats). */
static inline void jam_lowbit_row(jam_dtype dt, const uint8_t* w, int nblk, int8_t* codes, float* d32) {
    const int qk = jam_lowbit_qk(dt), bytes = jam_lowbit_bytes(dt);
    for (int b = 0; b < nblk; b++, w += bytes, codes += qk) {
        if (dt == JAM_PTQ1_0) jam_ptq1_0_codes(w, codes);
        else                  jam_q2_codes(w, qk, codes);
        const float d = jam_half_at(dt == JAM_PTQ1_0 ? w + JAM_PTQ1_0_D : w);
        for (int s = 0; s < qk / 32; s++) *d32++ = d;
    }
}

#if defined(__AVX2__)
#include <immintrin.h>
/* The 32 s8 codes (-1..2) of the 8 code bytes at q: byte e/4 goes to lane e, then bit 2(e%4) and the
 * bit above it become the code's 1s and 2s. No per-byte variable shift on x86, so two compares. */
static inline __m256i jam_q2_codes32(const uint8_t* q) {
    long long bytes; memcpy(&bytes, q, 8);
    const __m256i src = _mm256_shuffle_epi8(_mm256_set1_epi64x(bytes),
        _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7));
    const __m256i lo = _mm256_set1_epi32(0x40100401), hi = _mm256_set1_epi32((int) 0x80200802);
    const __m256i ones = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(src, lo), lo), _mm256_set1_epi8(1));
    const __m256i twos = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(src, hi), hi), _mm256_set1_epi8(2));
    return _mm256_sub_epi8(_mm256_or_si256(ones, twos), _mm256_set1_epi8(1));
}
#endif

#endif /* JAM_LOWBIT_H */
