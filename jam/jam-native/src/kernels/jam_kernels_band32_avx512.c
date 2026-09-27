/* The 32x4 VNNI band: every quantized prefill (n >= JAM_VNNI_MIN_SEQ) on AVX-512-VNNI.
 *
 * One tile, eight formats. A band is 32 weight rows, repacked per call into two 16-row groups whose
 * vpdpbusd lanes are the rows. The tile sweeps 4 activation columns at a time and every activation
 * broadcast feeds both groups: one broadcast per two dots.
 *
 * Why this shape. On Zen 5 the vector unit issues 4 pipe slots per cycle: a 512-bit vpdpbusd takes 2,
 * any other 512-bit op 2, a zmm load 1, a memory broadcast half. A tile is judged by the slots it
 * spends per dot beyond the dot itself (single core, fraction of the vpdpbusd peak):
 *   16 rows x 4 columns, one broadcast per dot ................ 0.52
 *   32 rows x 4 columns, one broadcast per two dots ........... 0.90-0.95
 *   sub-block scale in float (cvt, mul, fmadd, fnmadd) ........ 0.62-0.70
 *   sub-block scale in int (vpmulld, vpaddd) .................. 0.80-0.87
 * Measured and rejected: k-chunking the band into L1 (flat: the tile is pipe-bound, not load-bound)
 * and EVEX {1to16} broadcasts (the load unit serializes them, half the speed).
 * A band's fixed cost, streaming and repacking its weights, is worth about 10 columns of dots
 * (single core, k = 4096), so the tile only pays off from JAM_VNNI_MIN_SEQ columns up. Hence the
 * repack's shape: rows decode one at a time, everything after that runs 16 rows wide, in registers.
 *
 * Math. Activations are s8 codes q with a float scale xd, stored biased (u8 = q + 128) because they
 * are vpdpbusd's unsigned operand; weights are s8 codes w. Every dot acc = sum w (q + 128) therefore
 * carries 128 sum w, a constant of the weight row that the repack precomputes.
 *   K-quants   one xd per 256 elements (llama.cpp's Q8_K), integer sub-block scales sc and mins mn:
 *              xd (d (sum_sub sc acc_sub - K) - dmin sum_sub mn qsum_sub),   K = 128 sum_sub sc sum w
 *   32-blocks  one xd per 32 elements (llama.cpp's Q8_0 activations), one float scale d per block:
 *              sum_b xd_b (d_b acc_b - C_b),                                 C_b = 128 d_b sum w
 *
 * Repack: one blob per 16-row group per 256 elements.
 *   W     [64][16][4] s8   lanes = rows, a byte group = 4 consecutive k
 *   S     K-quants: sc [nsub][16] i32      32-blocks: d [8][16] f32, then C [8][16] f32
 *         (Q4_K, Q5_K: the second half of S holds the mins as i32, the source of MN)
 *   MN    [4][16][2] i16   min pairs (2p, 2p + 1)                           Q4_K, Q5_K
 *   K     [16] i32
 *   D, DMIN  [16] f32 each
 * A group short of 16 rows is padded with zero rows and stored under a mask.
 *
 * Phase 1 leaves in the job: xq = u8 [n][k], dx = f32 [n][k / 256] or [n][k / 32], and for the
 * K-quants xsum = i16 [n][k / 32], the per-32 sums of the s8 codes. */
#include "jam_internal.h"
#include "jam_kquant.h"
#include "jam_mxfp4.h"
#include "jam_q1_0.h"
#include <immintrin.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

enum { W_OFF = 0, S_OFF = 4096, MN_OFF = 5120, K_OFF = 5376, D_OFF = 5440, DMIN_OFF = 5504 };
_Static_assert(DMIN_OFF + 64 == JAM_BAND32_BLOB, "blob layout");

typedef enum {
    KIND_K32,   /* Q4_K, Q5_K: 8 sub-blocks of 32, integer scales and mins */
    KIND_K16,   /* Q6_K: 16 sub-blocks of 16, integer scales */
    KIND_F32    /* Q8_0, Q4_0, Q5_0, MXFP4, Q1_0: 8 blocks of 32, float scales */
} band_kind;

#define INLINE static inline __attribute__((always_inline))
#define ALIGN64 __attribute__((aligned(64)))

INLINE float h2f(const uint8_t* p) { uint16_t h; memcpy(&h, p, 2); return _cvtsh_ss(h); }

/* ================= phase 1: activations -> biased s8 codes ================= */

/* One 32-block at scale 1 / inv; returns the sum of its s8 codes. */
INLINE int quant32(const float* x, float inv, uint8_t* xq) {
    const __m512 v = _mm512_set1_ps(inv);
    __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x), v));
    __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + 16), v));
    __m256i q = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm512_cvtsepi32_epi8(q0)),
                                        _mm512_cvtsepi32_epi8(q1), 1);
    _mm256_storeu_si256((__m256i*) xq, _mm256_xor_si256(q, _mm256_set1_epi8((char) 0x80)));
    return _mm512_reduce_add_epi32(_mm512_add_epi32(q0, q1));
}

INLINE float absmax(const float* x, int n) {                  /* n a multiple of 16 */
    __m512 m = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) m = _mm512_max_ps(m, _mm512_abs_ps(_mm512_loadu_ps(x + i)));
    return _mm512_reduce_max_ps(m);
}

/* K-quants: one scale per 256 elements, plus the per-32 code sums for the min term. */
void jam_band32_quant256_avx512(void* job, int s0, int s1, int tid) {
    (void) tid;
    const jam_band_job* J = (const jam_band_job*) job;
    const int k = J->dim1, sblocks = k / JAM_QKK;
    for (int s = s0; s < s1; s++) {
        const float* x = J->rhs + (size_t) s * J->rhs_stride;
        uint8_t* xq = (uint8_t*) J->xq + (size_t) s * k;
        float* xd = J->dx + (size_t) s * sblocks;
        int16_t* xs = (int16_t*) J->xsum + (size_t) s * sblocks * 8;
        for (int sb = 0; sb < sblocks; sb++, x += JAM_QKK, xq += JAM_QKK, xs += 8) {
            float amax = absmax(x, JAM_QKK);
            xd[sb] = amax / 127.0f;
            for (int b = 0; b < 8; b++)
                xs[b] = (int16_t) quant32(x + b * JAM_QK, amax > 0.0f ? 127.0f / amax : 0.0f, xq + b * JAM_QK);
        }
    }
}

/* 32-block quants: one scale per 32 elements. */
void jam_band32_quant32_avx512(void* job, int s0, int s1, int tid) {
    (void) tid;
    const jam_band_job* J = (const jam_band_job*) job;
    const int k = J->dim1, nb = k / JAM_QK;
    for (int s = s0; s < s1; s++) {
        const float* x = J->rhs + (size_t) s * J->rhs_stride;
        uint8_t* xq = (uint8_t*) J->xq + (size_t) s * k;
        float* xd = J->dx + (size_t) s * nb;
        for (int b = 0; b < nb; b++, x += JAM_QK, xq += JAM_QK) {
            float amax = absmax(x, JAM_QK);
            xd[b] = amax / 127.0f;
            quant32(x, amax > 0.0f ? 127.0f / amax : 0.0f, xq);
        }
    }
}

/* ================= decode: one row, one 256-element stretch -> s8 codes + raw scales ================= */

/* w = the row's first byte, sb = which 256-element stretch, nbs = its 32-blocks (8, fewer on a ragged
 * last stretch of a 32-block quant). Writes nbs * 32 codes and the row's scales as 16 lanes:
 *   Q4_K, Q5_K   i32 sc[8] then mn[8]     Q6_K   i32 sc[16]     32-blocks   f32 d[8]
 * The K-quants also return the block's d (and dmin). */
typedef void (*band_decode)(const uint8_t* w, int sb, int nbs, uint8_t* codes, void* scales,
                            float* d, float* dmin);

/* 32 packed bytes -> 32 low nibbles, 32 high nibbles */
INLINE void nibbles64(const uint8_t* q, __m256i* lo, __m256i* hi) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256i v = _mm256_loadu_si256((const __m256i*) q);
    *lo = _mm256_and_si256(v, m4);
    *hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
}
/* 16 packed bytes -> [16 low nibbles | 16 high nibbles] */
INLINE __m256i nibbles32(const uint8_t* q) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i v = _mm_loadu_si128((const __m128i*) q);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_and_si128(v, m4)),
                                   _mm_and_si128(_mm_srli_epi16(v, 4), m4), 1);
}
/* the bits `mask` of every byte of qh at bit `shift`, moved up to bit 4 (above a nibble) */
INLINE __m256i plane(__m256i qh, int shift, int mask) {
    return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, shift), _mm256_set1_epi8((char) mask)), 4);
}
INLINE void put(uint8_t* codes, __m256i v) { _mm256_store_si256((__m256i*) codes, v); }

INLINE void k32_header(const uint8_t* w, void* scales, float* d, float* dmin) {   /* Q4_K / Q5_K */
    uint8_t scmn[16];
    jam_q4k_scales_mins(w + 4, scmn, scmn + 8);
    _mm512_store_si512(scales, _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) scmn)));
    *d = h2f(w); *dmin = h2f(w + 2);
}

INLINE void decode_q4k(const uint8_t* w, int sb, int nbs, uint8_t* codes, void* scales, float* d, float* dmin) {
    (void) nbs;
    w += (size_t) sb * JAM_Q4K_BYTES;
    k32_header(w, scales, d, dmin);
    for (int g = 0; g < 4; g++) {                 /* byte i of chunk g: sub-block 2g low, 2g + 1 high */
        __m256i lo, hi;
        nibbles64(w + 16 + g * 32, &lo, &hi);
        put(codes + g * 64, lo); put(codes + g * 64 + 32, hi);
    }
}

INLINE void decode_q5k(const uint8_t* w, int sb, int nbs, uint8_t* codes, void* scales, float* d, float* dmin) {
    (void) nbs;
    w += (size_t) sb * JAM_Q5K_BYTES;
    k32_header(w, scales, d, dmin);
    const __m256i qh = _mm256_loadu_si256((const __m256i*) (w + 16));   /* bit s of qh[i]: sub-block s */
    for (int g = 0; g < 4; g++) {
        __m256i lo, hi;
        nibbles64(w + 48 + g * 32, &lo, &hi);
        put(codes + g * 64,      _mm256_or_si256(lo, plane(qh, 2 * g, 1)));
        put(codes + g * 64 + 32, _mm256_or_si256(hi, plane(qh, 2 * g + 1, 1)));
    }
}

INLINE void decode_q6k(const uint8_t* w, int sb, int nbs, uint8_t* codes, void* scales, float* d, float* dmin) {
    (void) nbs; (void) dmin;
    w += (size_t) sb * JAM_Q6K_BYTES;              /* ql[128] qh[64] scales[16] d */
    _mm512_store_si512(scales, _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (w + 192))));
    *d = h2f(w + 208);
    const __m256i b32 = _mm256_set1_epi8(32);
    for (int h = 0; h < 2; h++) {                  /* a half: 4 planes of 32 codes */
        const __m256i qh = _mm256_loadu_si256((const __m256i*) (w + 128 + h * 32));
        __m256i q[4];
        nibbles64(w + h * 64,      &q[0], &q[2]);
        nibbles64(w + h * 64 + 32, &q[1], &q[3]);
        for (int j = 0; j < 4; j++)
            put(codes + h * 128 + j * 32, _mm256_sub_epi8(_mm256_or_si256(q[j], plane(qh, 2 * j, 3)), b32));
    }
}

INLINE void decode_q8_0(const uint8_t* w, int sb, int nbs, uint8_t* codes, void* scales, float* d, float* dmin) {
    (void) d; (void) dmin;
    w += (size_t) sb * 8 * JAM_Q8_0_BYTES;
    for (int b = 0; b < nbs; b++, w += JAM_Q8_0_BYTES) {
        ((float*) scales)[b] = h2f(w);
        put(codes + b * 32, _mm256_loadu_si256((const __m256i*) (w + 2)));
    }
}

INLINE void decode_q4_0(const uint8_t* w, int sb, int nbs, uint8_t* codes, void* scales, float* d, float* dmin) {
    (void) d; (void) dmin;
    w += (size_t) sb * 8 * JAM_Q4_0_BYTES;
    for (int b = 0; b < nbs; b++, w += JAM_Q4_0_BYTES) {
        ((float*) scales)[b] = h2f(w);
        put(codes + b * 32, _mm256_sub_epi8(nibbles32(w + 2), _mm256_set1_epi8(8)));
    }
}

INLINE void decode_q5_0(const uint8_t* w, int sb, int nbs, uint8_t* codes, void* scales, float* d, float* dmin) {
    (void) d; (void) dmin;
    w += (size_t) sb * 8 * JAM_Q5_0_BYTES;
    for (int b = 0; b < nbs; b++, w += JAM_Q5_0_BYTES) {
        ((float*) scales)[b] = h2f(w);
        __m256i bit4 = _mm256_and_si256(_mm256_movm_epi8((__mmask32) jam_load32(w + 2)), _mm256_set1_epi8(0x10));
        put(codes + b * 32, _mm256_sub_epi8(_mm256_or_si256(nibbles32(w + 6), bit4), _mm256_set1_epi8(16)));
    }
}

INLINE void decode_mxfp4(const uint8_t* w, int sb, int nbs, uint8_t* codes, void* scales, float* d, float* dmin) {
    (void) d; (void) dmin;
    static const int8_t lut[16] = { JAM_MXFP4_CODES };
    const __m256i t = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) lut));
    w += (size_t) sb * 8 * JAM_MXFP4_BYTES;
    for (int b = 0; b < nbs; b++, w += JAM_MXFP4_BYTES) {
        ((float*) scales)[b] = jam_mxfp4_dhalf(w[0]);
        put(codes + b * 32, _mm256_shuffle_epi8(t, nibbles32(w + 1)));
    }
}

INLINE void decode_q1_0(const uint8_t* w, int sb, int nbs, uint8_t* codes, void* scales, float* d, float* dmin) {
    (void) d; (void) dmin;
    for (int b = 0; b < nbs; b++) {                /* a 128-element block spans four 32-blocks */
        const int b32 = sb * 8 + b;
        const uint8_t* blk = w + (size_t) (b32 >> 2) * JAM_Q1_0_BYTES;
        ((float*) scales)[b] = h2f(blk);
        __m256i set = _mm256_movm_epi8((__mmask32) jam_load32(blk + 2 + (b32 & 3) * 4));
        put(codes + b * 32, _mm256_or_si256(_mm256_xor_si256(set, _mm256_set1_epi8(-1)), _mm256_set1_epi8(1)));
    }                                              /* bit set: +1, clear: -1 */
}

/* ================= repack: up to 16 rows -> blobs ================= */

/* 16 rows x 16 dwords (row r at src + r * stride) -> 16 dwords x 16 rows (dword g at dst + g * 64).
 * Four vpermt2d butterfly stages: stage s merges rows 2^s apart; v[i] ends as dword order[i]. */
INLINE void transpose16(const uint8_t* src, int stride, uint8_t* dst) {
    static const int32_t merge[4][2][16] = {
        {{ 0, 16,  1, 17,  2, 18,  3, 19,  4, 20,  5, 21,  6, 22,  7, 23 },
         { 8, 24,  9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31 }},
        {{ 0,  1, 16, 17,  4,  5, 20, 21,  8,  9, 24, 25, 12, 13, 28, 29 },
         { 2,  3, 18, 19,  6,  7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31 }},
        {{ 0,  1,  2,  3, 16, 17, 18, 19,  4,  5,  6,  7, 20, 21, 22, 23 },
         { 8,  9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31 }},
        {{ 0,  1,  2,  3,  4,  5,  6,  7, 16, 17, 18, 19, 20, 21, 22, 23 },
         { 8,  9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 }},
    };
    static const int order[16] = { 0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15 };
    __m512i v[16];                                 /* fully unrolled, so the 16 rows live in registers */
    #pragma GCC unroll 16
    for (int r = 0; r < 16; r++) v[r] = _mm512_load_si512((const void*) (src + r * stride));
    #pragma GCC unroll 4
    for (int s = 0; s < 4; s++) {
        const int span = 1 << s;
        const __m512i lo = _mm512_loadu_si512((const void*) merge[s][0]);
        const __m512i hi = _mm512_loadu_si512((const void*) merge[s][1]);
        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            if (i & span) continue;
            __m512i a = v[i], b = v[i + span];
            v[i]        = _mm512_permutex2var_epi32(a, lo, b);
            v[i + span] = _mm512_permutex2var_epi32(a, hi, b);
        }
    }
    #pragma GCC unroll 16
    for (int i = 0; i < 16; i++) _mm512_store_si512((void*) (dst + order[i] * 64), v[i]);
}

/* Rows decode one at a time into row-major codes and scales; transposes make both 16 rows wide,
 * and the sums and corrections are computed on the 16 rows at once. */
INLINE void repack16(const uint8_t* w, int64_t w_stride, int nrows, int nb, uint8_t* blob,
                     const band_kind kind, band_decode decode) {
    const int nsub = kind == KIND_K16 ? 16 : 8, ni = 64 / nsub;   /* 4-k groups per sub-block */
    const __m512i ones = _mm512_set1_epi8(1);
    ALIGN64 uint8_t codes[16 * JAM_QKK] = { 0 };   /* what no row writes stays zero: the padding rows */
    ALIGN64 uint8_t scales[16 * 64] = { 0 };       /* of a short group, the blocks past a ragged k */
    for (int sb = 0; sb * 8 < nb; sb++, blob += JAM_BAND32_BLOB) {
        const int nbs = nb - sb * 8 < 8 ? nb - sb * 8 : 8;
        memset(blob + D_OFF, 0, 128);              /* d, dmin */
        for (int r = 0; r < nrows; r++)
            decode(w + r * w_stride, sb, nbs, codes + r * JAM_QKK, scales + r * 64,
                   (float*) (blob + D_OFF) + r, (float*) (blob + DMIN_OFF) + r);
        for (int c = 0; c * 2 < nbs; c++)          /* 64 codes = 16 dwords per row per transpose */
            transpose16(codes + c * 64, JAM_QKK, blob + W_OFF + c * 1024);
        transpose16(scales, 64, blob + S_OFF);     /* S = lane 0..15 of every row; Q4_K/Q5_K: 8..15 = mn */

        __m512i bias = _mm512_setzero_si512();
        for (int sub = 0; sub < (kind == KIND_F32 ? nbs : nsub); sub++) {
            __m512i sum = _mm512_setzero_si512();  /* the codes' sum per row: every byte against 1 */
            for (int i = sub * ni; i < (sub + 1) * ni; i++)
                sum = _mm512_dpbusd_epi32(sum, ones, _mm512_load_si512((const void*) (blob + W_OFF + i * 64)));
            if (kind == KIND_F32) {
                __m512 c128 = _mm512_mul_ps(_mm512_load_ps(blob + S_OFF + sub * 64), _mm512_set1_ps(128.0f));
                _mm512_store_ps(blob + S_OFF + 512 + sub * 64, _mm512_mul_ps(c128, _mm512_cvtepi32_ps(sum)));
            } else {
                __m512i sc = _mm512_load_si512((const void*) (blob + S_OFF + sub * 64));
                bias = _mm512_add_epi32(bias, _mm512_mullo_epi32(sc, sum));
            }
        }
        if (kind == KIND_F32) continue;
        _mm512_store_si512((void*) (blob + K_OFF), _mm512_slli_epi32(bias, 7));
        if (kind == KIND_K16) continue;
        for (int p = 0; p < 4; p++) {              /* mn (2p, 2p + 1) -> one s16 pair per lane */
            __m512i lo = _mm512_load_si512((const void*) (blob + S_OFF + (8 + 2 * p) * 64));
            __m512i hi = _mm512_load_si512((const void*) (blob + S_OFF + (9 + 2 * p) * 64));
            _mm512_store_si512((void*) (blob + MN_OFF + p * 64), _mm512_or_si512(lo, _mm512_slli_epi32(hi, 16)));
        }
    }
}

/* ================= the tile: ng groups of 16 rows x nc columns, all of k ================= */

INLINE void tile(const int ng, const int nc, const band_kind kind,
                 const uint8_t* b0, const uint8_t* b1, int nb,
                 const uint8_t* xq, int64_t k, const float* xd, const int16_t* xs,
                 float* out, int64_t ldc, __mmask16 rows0, __mmask16 rows1) {
    const int nsub = kind == KIND_K16 ? 16 : 8, ni = 64 / nsub;
    const int64_t xd_stride = kind == KIND_F32 ? nb : nb / 8;
    __m512 f[2][4];
    for (int g = 0; g < 2; g++) for (int c = 0; c < 4; c++) f[g][c] = _mm512_setzero_ps();

    for (int sb = 0; sb * 8 < nb; sb++, b0 += JAM_BAND32_BLOB, b1 += JAM_BAND32_BLOB) {
        const int subs = kind == KIND_F32 && nb - sb * 8 < 8 ? nb - sb * 8 : nsub;
        const uint8_t* x = xq + sb * JAM_QKK;
        __m512i s1[2][4];
        for (int g = 0; g < 2; g++) for (int c = 0; c < 4; c++) s1[g][c] = _mm512_setzero_si512();

        for (int sub = 0; sub < subs; sub++) {
            __m512i acc[2][4];
            for (int g = 0; g < 2; g++) for (int c = 0; c < 4; c++) acc[g][c] = _mm512_setzero_si512();
            for (int i = sub * ni; i < (sub + 1) * ni; i++) {
                __m512i w0 = _mm512_load_si512((const void*) (b0 + W_OFF + i * 64));
                __m512i w1 = ng > 1 ? _mm512_load_si512((const void*) (b1 + W_OFF + i * 64)) : w0;
                #pragma GCC unroll 4
                for (int c = 0; c < nc; c++) {
                    __m512i a = _mm512_set1_epi32(jam_load32(x + c * k + i * 4));
                    acc[0][c] = _mm512_dpbusd_epi32(acc[0][c], a, w0);
                    if (ng > 1) acc[1][c] = _mm512_dpbusd_epi32(acc[1][c], a, w1);
                }
            }
            for (int g = 0; g < ng; g++) {
                const uint8_t* s = (g ? b1 : b0) + S_OFF + sub * 64;
                if (kind == KIND_F32) {            /* f += xd (d acc - C) */
                    __m512 d = _mm512_load_ps(s), cw = _mm512_load_ps(s + 512);
                    #pragma GCC unroll 4
                    for (int c = 0; c < nc; c++) {
                        __m512 xdv = _mm512_set1_ps(xd[c * xd_stride + sb * 8 + sub]);
                        f[g][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[g][c]), _mm512_mul_ps(d, xdv), f[g][c]);
                        f[g][c] = _mm512_fnmadd_ps(cw, xdv, f[g][c]);
                    }
                } else {                           /* s1 += sc acc, exact */
                    __m512i sc = _mm512_load_si512((const void*) s);
                    #pragma GCC unroll 4
                    for (int c = 0; c < nc; c++)
                        s1[g][c] = _mm512_add_epi32(s1[g][c], _mm512_mullo_epi32(acc[g][c], sc));
                }
            }
        }
        if (kind == KIND_F32) continue;

        for (int g = 0; g < ng; g++) {             /* once per 256: f += xd (d (s1 - K) - dmin s2) */
            const uint8_t* b = g ? b1 : b0;
            const __m512i kk = _mm512_load_si512((const void*) (b + K_OFF));
            const __m512 d = _mm512_load_ps(b + D_OFF);
            #pragma GCC unroll 4
            for (int c = 0; c < nc; c++) {
                __m512 v = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_sub_epi32(s1[g][c], kk)));
                if (kind == KIND_K32) {            /* s2 = sum_sub mn qsum: 4 pairs of s16 x s16 */
                    const int16_t* qsum = xs + c * nb + sb * 8;
                    __m512i s2 = _mm512_setzero_si512();
                    for (int p = 0; p < 4; p++)
                        s2 = _mm512_dpwssd_epi32(s2, _mm512_load_si512((const void*) (b + MN_OFF + p * 64)),
                                                 _mm512_set1_epi32(jam_load32(qsum + p * 2)));
                    v = _mm512_fnmadd_ps(_mm512_load_ps(b + DMIN_OFF), _mm512_cvtepi32_ps(s2), v);
                }
                f[g][c] = _mm512_fmadd_ps(v, _mm512_set1_ps(xd[c * xd_stride + sb]), f[g][c]);
            }
        }
    }
    for (int g = 0; g < ng; g++)
        for (int c = 0; c < nc; c++) _mm512_mask_storeu_ps(out + c * ldc + g * 16, g ? rows1 : rows0, f[g][c]);
}

/* ================= the band: claim a 32-row tile, repack it, sweep the columns ================= */

INLINE void band(const jam_band_job* J, int tid, const band_kind kind, band_decode decode) {
    const int m = J->dim0, k = J->dim1, n = J->seq, nb = k / JAM_QK;
    const int64_t ldc = J->out_stride;
    const int64_t xd_stride = kind == KIND_F32 ? nb : nb / 8;
    uint8_t* b0 = J->repack[tid].qs;
    uint8_t* b1 = b0 + (size_t) ((nb + 7) / 8) * JAM_BAND32_BLOB;
    /* Tiles are claimed from a shared counter, not taken from the caller's static slice: a worker on
     * an SMT sibling or the slower CCD then takes fewer tiles instead of stalling the call. */
    const int ntiles = (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND;
    for (int t; (t = atomic_fetch_add_explicit(&((jam_band_job*) J)->next_tile, 1, memory_order_relaxed)) < ntiles; ) {
        const int row = t * JAM_VNNI_BAND, rows = m - row < JAM_VNNI_BAND ? m - row : JAM_VNNI_BAND;
        const int ng = (rows + 15) / 16;
        const __mmask16 rows0 = (__mmask16) ((1u << (rows < 16 ? rows : 16)) - 1);
        const __mmask16 rows1 = (__mmask16) ((1u << (rows < 16 ? 0 : rows - 16)) - 1);
        for (int g = 0; g < ng; g++)
            repack16(J->w + (int64_t) (row + g * 16) * J->w_stride, J->w_stride,
                     rows - g * 16 < 16 ? rows - g * 16 : 16, nb, g ? b1 : b0, kind, decode);
        for (int c = 0; c < n; c += 4) {
            const int nc = n - c < 4 ? n - c : 4;
            #define TILE(NG, NC) tile(NG, NC, kind, b0, b1, nb, (const uint8_t*) J->xq + (size_t) c * k, k, \
                                      J->dx + (size_t) c * xd_stride, (const int16_t*) J->xsum + (size_t) c * nb, \
                                      J->out + c * ldc + row, ldc, rows0, rows1)
            if (nc == 4) { if (ng == 2) TILE(2, 4); else TILE(1, 4); }   /* constants: accumulators in registers */
            else         { if (ng == 2) TILE(2, nc); else TILE(1, nc); } /* the column tail, < 4 */
            #undef TILE
        }
    }
}

#define BAND(NAME, KIND, DECODE) \
    void NAME(void* job, int t0, int t1, int tid) { \
        (void) t0; (void) t1; \
        band((const jam_band_job*) job, tid, KIND, DECODE); \
    }
BAND(jam_q4k_band32_avx512,   KIND_K32, decode_q4k)
BAND(jam_q5k_band32_avx512,   KIND_K32, decode_q5k)
BAND(jam_q6k_band32_avx512,   KIND_K16, decode_q6k)
BAND(jam_q8_0_band32_avx512,  KIND_F32, decode_q8_0)
BAND(jam_q4_0_band32_avx512,  KIND_F32, decode_q4_0)
BAND(jam_q5_0_band32_avx512,  KIND_F32, decode_q5_0)
BAND(jam_mxfp4_band32_avx512, KIND_F32, decode_mxfp4)
BAND(jam_q1_0_band32_avx512,  KIND_F32, decode_q1_0)
