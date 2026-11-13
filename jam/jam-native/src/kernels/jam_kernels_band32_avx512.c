/* The 32x6 VNNI band: quantized prefill on AVX-512-VNNI, the 32-block quants from JAM_BAND32_MIN_SEQ
 * columns, the K-quants from JAM_VNNI_MIN_SEQ.
 *
 * One tile, eleven formats. A band is 32 weight rows, repacked per call into two 16-row groups whose
 * vpdpbusd lanes are the rows. The tile sweeps 6 activation columns at a time and every activation
 * broadcast feeds both groups: one broadcast per two dots, one weight load per six.
 *
 * Why this shape. On Zen 5 the vector unit issues 4 pipe slots per cycle: a 512-bit vpdpbusd takes 2,
 * any other 512-bit op 2, a zmm load 1, a memory broadcast half. A tile is judged by the slots it
 * spends per dot beyond the dot itself (single core, fraction of the vpdpbusd peak):
 *   16 rows x 4 columns, one broadcast per dot ................ 0.52
 *   32 rows x 4 columns, one broadcast per two dots ........... 0.90-0.95
 *   32 rows x 6 columns, a weight load per 6 dots, not 4 ...... measured +5 to +7% over x 4; x 8 spills
 *   sub-block scale in float (cvt, mul, fmadd, fnmadd) ........ 0.62-0.70
 *   sub-block scale in int (vpmulld, vpaddd) .................. 0.80-0.87
 * Measured and rejected: k-chunking the band into L1 (flat: the tile is pipe-bound, not load-bound)
 * and EVEX {1to16} broadcasts (the load unit serializes them, half the speed).
 * A band's fixed cost, streaming and repacking its weights, is worth about 7 columns of dots
 * (single core, k = 2048), and a short call pays it in full. Hence the repack's shape: 16 bytes of
 * each of 16 rows load straight into four registers, one row per lane group, and everything after
 * that, the format's decode included, runs 16 rows wide.
 *
 * Math. Activations are s8 codes q with a float scale xd, stored biased (u8 = q + 128) because they
 * are vpdpbusd's unsigned operand; weights are s8 codes w. Every dot sum w (q + 128) therefore
 * carries 128 sum w, a constant of the weight row that the repack precomputes as an integer, and the
 * accumulators start from its negative: what reaches the floats is the exact sum w q.
 *   K-quants   one xd per 256 elements (llama.cpp's Q8_K), integer sub-block scales sc and mins mn:
 *              xd (d s1 - dmin sum_sub mn qsum_sub),   s1 = sum_sub sc acc_sub - K,   K = 128 sum_sub sc sum w
 *   32-blocks  one xd per 32 elements (llama.cpp's Q8_0 activations), one float scale d per block:
 *              sum_b xd_b d_b acc_b,                   acc_b = sum w (q + 128) + B_b,   B_b = -128 sum w
 *
 * Repack: one blob per 16-row group per 256 elements.
 *   W     [64][16][4] s8   lanes = rows, a byte group = 4 consecutive k
 *   S     K-quants: sc [nsub][16] i32      32-blocks: d [8][16] f32
 *   S2    32-blocks: B [8][16] i32
 *   MN    [4][16][2] i16   min pairs (2p, 2p + 1)                           Q4_K, Q5_K
 *   K     [16] i32         -K                                               K-quants
 *   D, DMIN  [16] f32 each
 * A group short of 16 rows is padded with zero rows and stored under a mask.
 *
 * Phase 1 leaves in the job: xq = u8 [n][k], dx = f32 [n][k / 256] or [n][k / 32], and for the
 * K-quants xsum = i16 [n][k / 32], the per-32 sums of the s8 codes. */
#include "jam_internal.h"
#include "jam_kquant.h"
#include "jam_mxfp4.h"
#include "jam_lowbit.h"
#include "jam_q1_0.h"
#include <float.h>
#include <immintrin.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

enum {
    W_OFF = 0, W_BYTES = 1024,   /* per 64 codes of a row: 16 dwords x 16 rows */
    S_OFF = 4096, S2_OFF = 4608, /* S2: the second half of S (the 32-blocks' B) */
    MN_OFF = 5120, K_OFF = 5376, D_OFF = 5440, DMIN_OFF = 5504
};
_Static_assert(DMIN_OFF + 64 == JAM_BAND32_BLOB, "blob layout");

typedef enum {
    KIND_K32,   /* Q4_K, Q5_K: 8 sub-blocks of 32, integer scales and mins */
    KIND_K16,   /* Q6_K: 16 sub-blocks of 16, integer scales */
    KIND_F32    /* Q8_0, Q4_0, Q5_0, MXFP4, Q1_0 and the low-bit codes: 8 blocks of 32, float scales */
} band_kind;

#define NCMAX 6   /* columns per tile: 12 dot and 12 scale accumulators in registers, the float sums beside them */
_Static_assert(NCMAX == 6, "the band's tile switch names 4, 5 and 6 columns");
#define INLINE static inline __attribute__((always_inline))
#define ALIGN64 __attribute__((aligned(64)))

/* ================= phase 1: activations -> biased s8 codes ================= */

/* One 32-block at scale 1 / inv; returns the sum of its s8 codes, taken off the stored biased bytes
 * (sad against zero): nothing to overflow, even where an inf input made the int32 codes INT_MIN. */
INLINE int quant32(const float* x, float inv, uint8_t* xq) {
    const __m512 v = _mm512_set1_ps(inv);
    __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x), v));
    __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + 16), v));
    __m256i q = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm512_cvtsepi32_epi8(q0)),
                                        _mm512_cvtsepi32_epi8(q1), 1);
    const __m256i u = _mm256_xor_si256(q, _mm256_set1_epi8((char) 0x80));
    _mm256_storeu_si256((__m256i*) xq, u);
    const __m256i s4 = _mm256_sad_epu8(u, _mm256_setzero_si256());   /* 4 x 64-bit partial sums */
    __m128i s = _mm_add_epi64(_mm256_castsi256_si128(s4), _mm256_extracti128_si256(s4, 1));
    s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
    return _mm_cvtsi128_si32(s) - 32 * 128;
}

/* An amax below FLT_MIN reads as 0: 127 / amax would overflow, so the stretch quantizes to zeros. */
INLINE float absmax(const float* x, int n) {                  /* n a multiple of 16 */
    __m512 m = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) m = _mm512_max_ps(m, _mm512_abs_ps(_mm512_loadu_ps(x + i)));
    const float amax = _mm512_reduce_max_ps(m);
    return amax < FLT_MIN ? 0.0f : amax;
}

/* what maps [-amax, amax] onto the s8 codes; an all-zero stretch quantizes to zeros */
INLINE float to_codes(float amax) { return amax > 0.0f ? 127.0f / amax : 0.0f; }

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
            const float amax = absmax(x, JAM_QKK), inv = to_codes(amax);
            xd[sb] = amax / 127.0f;
            for (int b = 0; b < 8; b++)
                xs[b] = (int16_t) quant32(x + b * JAM_QK, inv, xq + b * JAM_QK);
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
            const float amax = absmax(x, JAM_QK);
            xd[b] = amax / 127.0f;
            quant32(x, to_codes(amax), xq);
        }
    }
}

/* ================= repack: 16 rows -> blobs, 16 rows wide from the first load ================= */

/* The 16 rows of a group at one 256-element stretch: row r starts at p[r / 4] + (r % 4) * stride. */
typedef struct { const uint8_t* p[4]; int64_t stride; } rows16;

/* The 16 bytes at `off` of every row, transposed: v[j] holds dword j of the chunk, lanes = rows.
 * Lane L of t[q] is loaded with row 4L + q, so a 4x4 transpose inside each 128-bit lane is the whole
 * job: 16 loads and 8 in-lane unpacks per 16 rows x 16 bytes, no shuffle across lanes, and nothing
 * between the weight bytes and the registers. */
typedef struct { __m512i v[4]; } quad;
INLINE quad chunk(const rows16 R, int64_t off) {
    __m512i t[4];
    #pragma GCC unroll 4
    for (int q = 0; q < 4; q++) {
        const int64_t o = q * R.stride + off;
        t[q] = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*) (R.p[0] + o)));
        t[q] = _mm512_inserti32x4(t[q], _mm_loadu_si128((const __m128i*) (R.p[1] + o)), 1);
        t[q] = _mm512_inserti32x4(t[q], _mm_loadu_si128((const __m128i*) (R.p[2] + o)), 2);
        t[q] = _mm512_inserti32x4(t[q], _mm_loadu_si128((const __m128i*) (R.p[3] + o)), 3);
    }
    const __m512i a = _mm512_unpacklo_epi32(t[0], t[1]), b = _mm512_unpackhi_epi32(t[0], t[1]);
    const __m512i c = _mm512_unpacklo_epi32(t[2], t[3]), d = _mm512_unpackhi_epi32(t[2], t[3]);
    return (quad) {{ _mm512_unpacklo_epi64(a, c), _mm512_unpackhi_epi64(a, c),
                     _mm512_unpacklo_epi64(b, d), _mm512_unpackhi_epi64(b, d) }};
}

/* A format's repack: the stretch's nbs 32-blocks (8, fewer on a ragged last stretch of a 32-block
 * quant) -> the blob's codes W and its raw scales:
 *   Q4_K, Q5_K   S = sc [8], MN, D, DMIN     Q6_K   S = sc [16], D     32-blocks   S = d [nbs]
 * Every chunk it reads lies inside the stretch: nothing is read past a row's last block. */
typedef void (*band_repack)(const rows16 R, int nbs, uint8_t* blob);
#define REPACK(NAME) INLINE void NAME(const rows16 R, int nbs, uint8_t* blob)

#define X32(v) _mm512_set1_epi32((int) (v))
INLINE __m512i and32(__m512i a, uint32_t m) { return _mm512_and_si512(a, X32(m)); }
INLINE __m512i nib_lo(__m512i v) { return and32(v, 0x0f0f0f0f); }
INLINE __m512i nib_hi(__m512i v) { return and32(_mm512_srli_epi32(v, 4), 0x0f0f0f0f); }
/* the fp16 in the low / high word of every lane */
INLINE __m512 half_lo(__m512i v) { return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(v)); }
INLINE __m512 half_hi(__m512i v) { return half_lo(_mm512_srli_epi32(v, 16)); }
INLINE void put_w(uint8_t* blob, int i, __m512i v) { _mm512_store_si512((void*) (blob + W_OFF + i * 64), v); }
INLINE void put_s(uint8_t* blob, int i, __m512i v) { _mm512_store_si512((void*) (blob + S_OFF + i * 64), v); }
INLINE void put_d(uint8_t* blob, int i, __m512 v)  { _mm512_store_ps(blob + S_OFF + i * 64, v); }

/* table[n] = a dword whose byte i is `one` where bit i of n is set, else `zero` (vpermd reads the low
 * 4 bits of an index, so a shifted bit field indexes it without a mask) */
#define SPREAD1(n, one, zero) \
    (int32_t) ((uint32_t) ((n) & 1 ? (one) : (zero))       | (uint32_t) ((n) & 2 ? (one) : (zero)) << 8 | \
               (uint32_t) ((n) & 4 ? (one) : (zero)) << 16 | (uint32_t) ((n) & 8 ? (one) : (zero)) << 24)
#define SPREAD(one, zero) { \
    SPREAD1(0, one, zero),  SPREAD1(1, one, zero),  SPREAD1(2, one, zero),  SPREAD1(3, one, zero),  \
    SPREAD1(4, one, zero),  SPREAD1(5, one, zero),  SPREAD1(6, one, zero),  SPREAD1(7, one, zero),  \
    SPREAD1(8, one, zero),  SPREAD1(9, one, zero),  SPREAD1(10, one, zero), SPREAD1(11, one, zero), \
    SPREAD1(12, one, zero), SPREAD1(13, one, zero), SPREAD1(14, one, zero), SPREAD1(15, one, zero) }
INLINE __m512i spread(const int32_t table[16], __m512i bits, int shift) {
    return _mm512_permutexvar_epi32(_mm512_srli_epi32(bits, shift), _mm512_load_si512((const void*) table));
}

/* Q4_K / Q5_K header {fp16 d, dmin; 12 bytes of 6-bit scales and mins}: llama.cpp's three-mask word
 * trick (jam_q4k_scales_mins), on 16 rows. */
INLINE void k32_header(const rows16 R, uint8_t* blob) {
    const quad h = chunk(R, 0);                    /* d dmin | bytes 0-3 | 4-7 | 8-11 */
    const __m512i sc[2] = { and32(h.v[1], 0x3f3f3f3f),
        _mm512_or_si512(nib_lo(h.v[3]), _mm512_slli_epi32(and32(_mm512_srli_epi32(h.v[1], 6), 0x03030303), 4)) };
    const __m512i mn[2] = { and32(h.v[2], 0x3f3f3f3f),
        _mm512_or_si512(nib_hi(h.v[3]), _mm512_slli_epi32(and32(_mm512_srli_epi32(h.v[2], 6), 0x03030303), 4)) };
    for (int s = 0; s < 8; s++) put_s(blob, s, and32(_mm512_srli_epi32(sc[s >> 2], 8 * (s & 3)), 0xFF));
    for (int p = 0; p < 4; p++) {                  /* mn (2p, 2p + 1) -> one s16 pair per lane */
        const __m512i pair = _mm512_srli_epi32(mn[p >> 1], 16 * (p & 1));
        _mm512_store_si512((void*) (blob + MN_OFF + p * 64),
                           _mm512_or_si512(and32(pair, 0xFF), _mm512_slli_epi32(and32(pair, 0xFF00), 8)));
    }
    _mm512_store_ps(blob + D_OFF, half_lo(h.v[0]));
    _mm512_store_ps(blob + DMIN_OFF, half_hi(h.v[0]));
}

REPACK(repack_q4k) {                               /* d dmin scales[12] qs[128] */
    (void) nbs;
    k32_header(R, blob);
    for (int c = 0; c < 8; c++) {                  /* byte i of pair g = c / 2: sub-block 2g low, 2g + 1 high */
        const quad q = chunk(R, 16 + c * 16);
        for (int j = 0; j < 4; j++) {
            const int i = (c >> 1) * 16 + (c & 1) * 4 + j;
            put_w(blob, i, nib_lo(q.v[j]));
            put_w(blob, i + 8, nib_hi(q.v[j]));
        }
    }
}

REPACK(repack_q5k) {                               /* d dmin scales[12] qh[32] qs[128] */
    (void) nbs;
    k32_header(R, blob);
    const quad qh[2] = { chunk(R, 16), chunk(R, 32) };   /* bit s of qh[i]: bit 4 of element i of sub-block s */
    for (int c = 0; c < 8; c++) {
        const quad q = chunk(R, 48 + c * 16);
        const int g = c >> 1;
        for (int j = 0; j < 4; j++) {
            const int i = g * 16 + (c & 1) * 4 + j;
            const __m512i h = qh[c & 1].v[j];
            const __m512i lo4 = _mm512_slli_epi32(and32(_mm512_srli_epi32(h, 2 * g), 0x01010101), 4);
            const __m512i hi4 = _mm512_slli_epi32(and32(_mm512_srli_epi32(h, 2 * g + 1), 0x01010101), 4);
            put_w(blob, i, _mm512_or_si512(nib_lo(q.v[j]), lo4));
            put_w(blob, i + 8, _mm512_or_si512(nib_hi(q.v[j]), hi4));
        }
    }
}

/* Q6_K: ql[128] qh[64] scales[16] d. Element e of half h = e / 128 reads nibble plane j = e / 32 % 4
 * (ql low nibbles for j < 2, high for j >= 2) and bits 2j, 2j + 1 of qh[h * 32 + e % 32]. */
REPACK(repack_q6k) {
    (void) nbs;
    const __m512i b32 = _mm512_set1_epi8(32);
    const quad sc = chunk(R, 192);
    for (int s = 0; s < 16; s++)                   /* s8 scales, one byte each */
        put_s(blob, s, _mm512_srai_epi32(_mm512_slli_epi32(sc.v[s >> 2], 24 - 8 * (s & 3)), 24));
    _mm512_store_ps(blob + D_OFF, half_hi(chunk(R, 194).v[3]));   /* bytes 208, 209 */
    const quad qh[4] = { chunk(R, 128), chunk(R, 144), chunk(R, 160), chunk(R, 176) };
    for (int c = 0; c < 8; c++) {                  /* ql[16c ..]: half c / 4, planes jl and jl + 2 */
        const quad q = chunk(R, c * 16);
        const int h = c >> 2, jl = (c >> 1) & 1;
        for (int j = 0; j < 4; j++) {
            const int i = h * 32 + jl * 8 + (c & 1) * 4 + j;
            const __m512i bits = qh[h * 2 + (c & 1)].v[j];
            const __m512i lo2 = _mm512_slli_epi32(and32(_mm512_srli_epi32(bits, 2 * jl), 0x03030303), 4);
            const __m512i hi2 = _mm512_slli_epi32(and32(_mm512_srli_epi32(bits, 2 * jl + 4), 0x03030303), 4);
            put_w(blob, i, _mm512_sub_epi8(_mm512_or_si512(nib_lo(q.v[j]), lo2), b32));
            put_w(blob, i + 16, _mm512_sub_epi8(_mm512_or_si512(nib_hi(q.v[j]), hi2), b32));
        }
    }
}

REPACK(repack_q8_0) {                              /* per 32: fp16 d, qs[32] */
    for (int b = 0; b < nbs; b++) {
        const int64_t at = (int64_t) b * JAM_Q8_0_BYTES;
        put_d(blob, b, half_lo(chunk(R, at).v[0]));
        const quad q0 = chunk(R, at + 2), q1 = chunk(R, at + 18);
        for (int j = 0; j < 4; j++) { put_w(blob, b * 8 + j, q0.v[j]); put_w(blob, b * 8 + 4 + j, q1.v[j]); }
    }
}

REPACK(repack_q4_0) {                              /* per 32: fp16 d, qs[16]; element e low, e + 16 high */
    const __m512i b8 = _mm512_set1_epi8(8);
    for (int b = 0; b < nbs; b++) {
        const int64_t at = (int64_t) b * JAM_Q4_0_BYTES;
        put_d(blob, b, half_lo(chunk(R, at).v[0]));
        const quad q = chunk(R, at + 2);
        for (int j = 0; j < 4; j++) {
            put_w(blob, b * 8 + j, _mm512_sub_epi8(nib_lo(q.v[j]), b8));
            put_w(blob, b * 8 + 4 + j, _mm512_sub_epi8(nib_hi(q.v[j]), b8));
        }
    }
}

REPACK(repack_q5_0) {                              /* per 32: fp16 d, u32 qh (bit e: element e), qs[16] */
    static const int32_t bit4[16] ALIGN64 = SPREAD(0x10, 0);
    const __m512i b16 = _mm512_set1_epi8(16);
    for (int b = 0; b < nbs; b++) {
        const int64_t at = (int64_t) b * JAM_Q5_0_BYTES;
        const quad h = chunk(R, at);               /* d, qh's low word | qh's high word, ... */
        put_d(blob, b, half_lo(h.v[0]));
        const __m512i qh = _mm512_or_si512(_mm512_srli_epi32(h.v[0], 16), _mm512_slli_epi32(h.v[1], 16));
        const quad q = chunk(R, at + 6);
        for (int j = 0; j < 4; j++) {
            const __m512i lo = _mm512_or_si512(nib_lo(q.v[j]), spread(bit4, qh, 4 * j));
            const __m512i hi = _mm512_or_si512(nib_hi(q.v[j]), spread(bit4, qh, 16 + 4 * j));
            put_w(blob, b * 8 + j, _mm512_sub_epi8(lo, b16));
            put_w(blob, b * 8 + 4 + j, _mm512_sub_epi8(hi, b16));
        }
    }
}

REPACK(repack_mxfp4) {                             /* per 32: E8M0 e, qs[16] of FP4 codes */
    static const int8_t codes[16] = { JAM_MXFP4_CODES };
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) codes));
    for (int b = 0; b < nbs; b++) {
        const int64_t at = (int64_t) b * JAM_MXFP4_BYTES;
        /* jam_mxfp4_dhalf on 16 rows: 2^(e - 128), e = 0 and 1 the subnormals */
        const __m512i e = and32(chunk(R, at).v[0], 0xFF);
        const __m512i normal = _mm512_slli_epi32(_mm512_sub_epi32(e, X32(1)), 23);
        const __m512i tiny = _mm512_slli_epi32(_mm512_add_epi32(e, X32(1)), 21);
        put_d(blob, b, _mm512_castsi512_ps(_mm512_mask_mov_epi32(normal, _mm512_cmplt_epu32_mask(e, X32(2)), tiny)));
        const quad q = chunk(R, at + 1);
        for (int j = 0; j < 4; j++) {
            put_w(blob, b * 8 + j, _mm512_shuffle_epi8(lut, nib_lo(q.v[j])));
            put_w(blob, b * 8 + 4 + j, _mm512_shuffle_epi8(lut, nib_hi(q.v[j])));
        }
    }
}

REPACK(repack_q1_0) {                              /* per 128: fp16 d, 128 sign bits (set: +1, clear: -1) */
    static const int32_t sign[16] ALIGN64 = SPREAD(0x01, 0xFF);
    for (int b = 0; b < nbs; b += 4) {
        const int64_t at = (int64_t) (b / 4) * JAM_Q1_0_BYTES;
        const __m512 d = half_lo(chunk(R, at).v[0]);
        const quad q = chunk(R, at + 2);           /* v[j]: the 32 bits of 32-block b + j */
        for (int j = 0; j < 4; j++) {
            put_d(blob, b + j, d);
            for (int g = 0; g < 8; g++) put_w(blob, (b + j) * 8 + g, spread(sign, q.v[j], 4 * g));
        }
    }
}

/* Q2_0 / PQ2_0: fp16 d, then one byte per 4 elements, 2-bit codes LSB-first, value = code - 1. Byte t
 * of a lane is copied to its 4 bytes; bit 2e and the bit above it are then element e's 1s and 2s. */
INLINE void q2_repack(const rows16 R, int nbs, uint8_t* blob, const int qk) {
    /* shuffle indices: every byte reads the first byte of its dword (+ t below: byte t) */
    const __m512i byte0 = _mm512_set4_epi32(0x0c0c0c0c, 0x08080808, 0x04040404, 0);
    const int per = qk / 32, bytes = 2 + qk / 4;   /* 32-blocks and bytes per block */
    const __m512i minus1 = _mm512_set1_epi8(-1), one = _mm512_set1_epi8(1), two = _mm512_set1_epi8(2);
    for (int b = 0; b < nbs; b += per) {
        const int64_t at = (int64_t) (b / per) * bytes;
        const __m512 d = half_lo(chunk(R, at).v[0]);
        for (int s = 0; s < per; s++) put_d(blob, b + s, d);
        for (int c = 0; c < per / 2; c++) {        /* 16 code bytes: 16 k-groups, two 32-blocks */
            const quad q = chunk(R, at + 2 + c * 16);
            for (int j = 0; j < 4; j++)
                for (int t = 0; t < 4; t++) {
                    const __m512i src =
                        _mm512_shuffle_epi8(q.v[j], _mm512_add_epi8(byte0, _mm512_set1_epi8((char) t)));
                    __m512i code =
                        _mm512_mask_add_epi8(minus1, _mm512_test_epi8_mask(src, X32(0x40100401)), minus1, one);
                    code = _mm512_mask_add_epi8(code, _mm512_test_epi8_mask(src, X32(0x80200802)), code, two);
                    put_w(blob, b * 8 + c * 16 + j * 4 + t, code);
                }
        }
    }
}
REPACK(repack_q2_0)  { q2_repack(R, nbs, blob, JAM_Q2_0_QK); }
REPACK(repack_pq2_0) { q2_repack(R, nbs, blob, JAM_PQ2_0_QK); }

/* The trits of every byte b at the powers given per byte of a 16-bit half (mul = both halves of a
 * lane): q = b * 3^n mod 256, trit - 1 = (q >= 86) + (q >= 171) - 1 (jam_ptq1_0_codes: q * 3 >> 8). */
INLINE __m512i trits(__m512i v, uint32_t mul) {
    const __m512i m = X32(mul), low = _mm512_set1_epi16(0x00FF);
    const __m512i even = _mm512_and_si512(_mm512_mullo_epi16(_mm512_and_si512(v, low), m), low);
    const __m512i odd = _mm512_slli_epi16(_mm512_mullo_epi16(_mm512_srli_epi16(v, 8), m), 8);
    const __m512i q = _mm512_or_si512(even, odd);
    const __m512i ge1 = _mm512_movm_epi8(_mm512_cmpge_epu8_mask(q, _mm512_set1_epi8(86)));
    const __m512i ge2 = _mm512_movm_epi8(_mm512_cmpge_epu8_mask(q, _mm512_set1_epi8((char) 171)));
    return _mm512_sub_epi8(_mm512_sub_epi8(_mm512_set1_epi8(-1), ge1), ge2);
}

/* PTQ1_0, per 128: qs[24] qh[2] fp16 d, base-3 digits. In a stage of c bytes, element n c + m is
 * trit n of byte m: 16 bytes x 5 trits, 8 bytes x 5 trits, then qh's 2 bytes x 4 trits. */
REPACK(repack_ptq1_0) {
    static const uint32_t pow3[5] = { 1, 3, 9, 27, 81 };
    for (int b = 0; b < nbs; b += 4) {
        const int64_t at = (int64_t) (b / 4) * JAM_PTQ1_0_BYTES;
        const quad s16 = chunk(R, at), rest = chunk(R, at + 12);   /* rest: -, qs[16..19], qs[20..23], qh d */
        for (int j = 0; j < 4; j++) put_d(blob, b + j, half_hi(rest.v[3]));
        for (int n = 0; n < 5; n++) {
            for (int j = 0; j < 4; j++) put_w(blob, b * 8 + n * 4 + j, trits(s16.v[j], pow3[n] * 0x00010001u));
            for (int j = 0; j < 2; j++)
                put_w(blob, b * 8 + 20 + n * 2 + j, trits(rest.v[1 + j], pow3[n] * 0x00010001u));
        }
        /* elements 120..127: trits 0, 1 of (qh[0], qh[1]), then trits 2, 3 */
        const __m512i qh2 = _mm512_or_si512(and32(rest.v[3], 0xFFFF), _mm512_slli_epi32(rest.v[3], 16));
        put_w(blob, b * 8 + 30, trits(qh2, 0x00030001));
        put_w(blob, b * 8 + 31, trits(qh2, 0x001B0009));
    }
}

/* A short group's rows are copied at this stride: the widest stretch of any format, 8 Q8_0 blocks. */
enum { PAD_STRIDE = 8 * JAM_Q8_0_BYTES };

INLINE void repack16(const uint8_t* w, int64_t w_stride, int nrows, int nb, uint8_t* blob,
                     const band_kind kind, band_repack repack, const int bytes256) {
    const int nsub = kind == KIND_K16 ? 16 : 8, ni = 64 / nsub;   /* 4-k groups per sub-block */
    const __m512i ones = _mm512_set1_epi8(1), zero = _mm512_setzero_si512();
    for (int sb = 0; sb * 8 < nb; sb++, blob += JAM_BAND32_BLOB, w += bytes256) {
        const int nbs = nb - sb * 8 < 8 ? nb - sb * 8 : 8;
        rows16 R = { { w, w + 4 * w_stride, w + 8 * w_stride, w + 12 * w_stride }, w_stride };
        ALIGN64 uint8_t pad[16 * PAD_STRIDE];
        if (nrows < 16) {                          /* a short group: its rows over zero rows, masked at the store */
            const size_t bytes = (size_t) nbs * bytes256 / 8;
            memset(pad, 0, sizeof pad);
            for (int r = 0; r < nrows; r++) memcpy(pad + r * PAD_STRIDE, w + r * w_stride, bytes);
            R = (rows16) { { pad, pad + 4 * PAD_STRIDE, pad + 8 * PAD_STRIDE, pad + 12 * PAD_STRIDE }, PAD_STRIDE };
        }
        repack(R, nbs, blob);

        /* Every dot acc = sum w (q + 128) carries 128 sum w: the codes' sum per row, every byte against 1. */
        __m512i bias = zero;
        for (int sub = 0; sub < (kind == KIND_F32 ? nbs : nsub); sub++) {
            __m512i sum = zero;
            for (int i = sub * ni; i < (sub + 1) * ni; i++)
                sum = _mm512_dpbusd_epi32(sum, ones, _mm512_load_si512((const void*) (blob + W_OFF + i * 64)));
            if (kind == KIND_F32) {                /* B = -128 sum w, the block's dots start from it */
                const __m512i b = _mm512_slli_epi32(_mm512_sub_epi32(zero, sum), 7);
                _mm512_store_si512((void*) (blob + S2_OFF + sub * 64), b);
            } else {
                const __m512i sc = _mm512_load_si512((const void*) (blob + S_OFF + sub * 64));
                bias = _mm512_add_epi32(bias, _mm512_mullo_epi32(sc, sum));
            }
        }
        if (kind != KIND_F32)                      /* -K = -128 sum_sub sc sum w, where s1 starts */
            _mm512_store_si512((void*) (blob + K_OFF), _mm512_slli_epi32(_mm512_sub_epi32(zero, bias), 7));
    }
}

/* ================= the tile: ng groups of 16 rows x nc columns, all of k ================= */

INLINE void tile(const int ng, const int nc, const band_kind kind,
                 const uint8_t* b0, const uint8_t* b1, int nb,
                 const uint8_t* xq, int64_t k, const float* xd, const int16_t* xs,
                 float* out, int64_t ldc, __mmask16 rows0, __mmask16 rows1) {
    const int nsub = kind == KIND_K16 ? 16 : 8, ni = 64 / nsub;
    const int64_t xd_stride = kind == KIND_F32 ? nb : nb / 8;
    __m512 f[2][NCMAX];
    for (int g = 0; g < 2; g++) for (int c = 0; c < NCMAX; c++) f[g][c] = _mm512_setzero_ps();

    for (int sb = 0; sb * 8 < nb; sb++, b0 += JAM_BAND32_BLOB, b1 += JAM_BAND32_BLOB) {
        const int subs = kind == KIND_F32 && nb - sb * 8 < 8 ? nb - sb * 8 : nsub;
        const uint8_t* x = xq + sb * JAM_QKK;
        __m512i s1[2][NCMAX];
        for (int g = 0; g < 2; g++) {              /* K-quants: s1 starts at -K */
            const __m512i start = kind != KIND_F32 && g < ng
                ? _mm512_load_si512((const void*) ((g ? b1 : b0) + K_OFF)) : _mm512_setzero_si512();
            for (int c = 0; c < NCMAX; c++) s1[g][c] = start;
        }

        for (int sub = 0; sub < subs; sub++) {
            __m512i acc[2][NCMAX];
            for (int g = 0; g < 2; g++) {          /* 32-blocks: the dots start at B, so acc ends as sum w q */
                const __m512i start = kind == KIND_F32 && g < ng
                    ? _mm512_load_si512((const void*) ((g ? b1 : b0) + S2_OFF + sub * 64)) : _mm512_setzero_si512();
                for (int c = 0; c < NCMAX; c++) acc[g][c] = start;
            }
            for (int i = sub * ni; i < (sub + 1) * ni; i++) {
                __m512i w0 = _mm512_load_si512((const void*) (b0 + W_OFF + i * 64));
                __m512i w1 = ng > 1 ? _mm512_load_si512((const void*) (b1 + W_OFF + i * 64)) : w0;
                #pragma GCC unroll 6
                for (int c = 0; c < nc; c++) {
                    __m512i a = _mm512_set1_epi32(jam_load32(x + c * k + i * 4));
                    acc[0][c] = _mm512_dpbusd_epi32(acc[0][c], a, w0);
                    if (ng > 1) acc[1][c] = _mm512_dpbusd_epi32(acc[1][c], a, w1);
                }
            }
            for (int g = 0; g < ng; g++) {
                const uint8_t* b = g ? b1 : b0;
                if (kind == KIND_F32) {            /* f += xd d acc */
                    const __m512 d = _mm512_load_ps(b + S_OFF + sub * 64);
                    #pragma GCC unroll 6
                    for (int c = 0; c < nc; c++) {
                        const __m512 xdv = _mm512_set1_ps(xd[c * xd_stride + sb * 8 + sub]);
                        f[g][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[g][c]), _mm512_mul_ps(d, xdv), f[g][c]);
                    }
                } else {                           /* s1 += sc acc, exact */
                    const __m512i sc = _mm512_load_si512((const void*) (b + S_OFF + sub * 64));
                    #pragma GCC unroll 6
                    for (int c = 0; c < nc; c++)
                        s1[g][c] = _mm512_add_epi32(s1[g][c], _mm512_mullo_epi32(acc[g][c], sc));
                }
            }
        }
        if (kind == KIND_F32) continue;

        for (int g = 0; g < ng; g++) {             /* once per 256: f += xd (d s1 - dmin s2) */
            const uint8_t* b = g ? b1 : b0;
            const __m512 d = _mm512_load_ps(b + D_OFF);
            #pragma GCC unroll 6
            for (int c = 0; c < nc; c++) {
                __m512 v = _mm512_mul_ps(d, _mm512_cvtepi32_ps(s1[g][c]));
                if (kind == KIND_K32) {            /* s2 = sum_sub mn qsum: 4 pairs of s16 x s16 */
                    const int16_t* qsum = xs + c * nb + sb * 8;
                    __m512i s2 = _mm512_setzero_si512();
                    for (int p = 0; p < 4; p++) {
                        const __m512i mn = _mm512_load_si512((const void*) (b + MN_OFF + p * 64));
                        s2 = _mm512_dpwssd_epi32(s2, mn, _mm512_set1_epi32(jam_load32(qsum + p * 2)));
                    }
                    v = _mm512_fnmadd_ps(_mm512_load_ps(b + DMIN_OFF), _mm512_cvtepi32_ps(s2), v);
                }
                f[g][c] = _mm512_fmadd_ps(v, _mm512_set1_ps(xd[c * xd_stride + sb]), f[g][c]);
            }
        }
    }
    for (int g = 0; g < ng; g++)
        for (int c = 0; c < nc; c++)
            _mm512_mask_storeu_ps(out + c * ldc + g * 16, g ? rows1 : rows0, f[g][c]);
}

/* ================= the band: claim a 32-row tile, repack it, sweep the columns ================= */

/* Reads one byte per cache line of nrows contiguous rows, in address order. A short prompt is
 * memory-bound and the repack walks its rows in lockstep: 16 interleaved streams, which hardware
 * prefetchers follow badly. Touched first, the group arrives as one stream (n = 16: +4% to +10%).
 * ponytail: a touch pass, not a row-major repack; that would save the second read. */
INLINE void touch_rows(const uint8_t* w, int64_t w_stride, int nrows, int row_bytes) {
    unsigned seen = 0;
    for (int r = 0; r < nrows; r++)
        for (int o = 0; o < row_bytes; o += 64) seen += w[r * w_stride + o];
    __asm__ volatile("" : : "r"(seen));            /* the reads are the point: keep them */
}

INLINE void band(jam_band_job* J, int tid, const band_kind kind, band_repack repack,
                 const int bytes256) {             /* weight bytes per 256 elements */
    const int m = J->dim0, k = J->dim1, n = J->seq, nb = k / JAM_QK;
    const int row_bytes = nb * bytes256 / 8;
    const int64_t ldc = J->out_stride;
    const int64_t xd_stride = kind == KIND_F32 ? nb : nb / 8;
    const uint8_t* xq = (const uint8_t*) J->xq;
    const int16_t* xs = (const int16_t*) J->xsum;
    uint8_t* b0 = J->repack[tid].qs;
    uint8_t* b1 = b0 + (size_t) ((nb + 7) / 8) * JAM_BAND32_BLOB;
    /* Tiles are claimed from a shared counter, not taken from the caller's static slice: a worker on
     * an SMT sibling or the slower CCD then takes fewer tiles instead of stalling the call. */
    const int ntiles = (m + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND;
    for (int t; (t = atomic_fetch_add_explicit(&J->next_tile, 1, memory_order_relaxed)) < ntiles; ) {
        const int row = t * JAM_VNNI_BAND, rows = m - row < JAM_VNNI_BAND ? m - row : JAM_VNNI_BAND;
        const int ng = (rows + 15) / 16;
        const __mmask16 rows0 = (__mmask16) ((1u << (rows < 16 ? rows : 16)) - 1);
        const __mmask16 rows1 = (__mmask16) ((1u << (rows < 16 ? 0 : rows - 16)) - 1);
        for (int g = 0; g < ng; g++) {
            const uint8_t* w = J->w + (int64_t) (row + g * 16) * J->w_stride;
            const int nrows = rows - g * 16 < 16 ? rows - g * 16 : 16;
            touch_rows(w, J->w_stride, nrows, row_bytes);
            repack16(w, J->w_stride, nrows, nb, g ? b1 : b0, kind, repack, bytes256);
        }
        /* ng and nc as constants keep the tile's accumulators in registers. The last 7 to 11 columns
         * go as two tiles of 4 to 6, so only a call of 3 columns or fewer runs the variable form. */
        #define TILE(NG, NC) tile(NG, NC, kind, b0, b1, nb, xq + (size_t) c * k, k, \
                                  J->dx + (size_t) c * xd_stride, xs + (size_t) c * nb, \
                                  J->out + c * ldc + row, ldc, rows0, rows1)
        for (int c = 0, nc; c < n; c += nc) {
            const int left = n - c;
            nc = left >= 2 * NCMAX ? NCMAX : left > NCMAX ? (left + 1) / 2 : left;
            switch (nc) {
                case 6:  if (ng == 2) TILE(2, 6); else TILE(1, 6); break;
                case 5:  if (ng == 2) TILE(2, 5); else TILE(1, 5); break;
                case 4:  if (ng == 2) TILE(2, 4); else TILE(1, 4); break;
                default: if (ng == 2) TILE(2, nc); else TILE(1, nc); break;
            }
        }
        #undef TILE
    }
}

#define BAND(NAME, KIND, REPACK, BYTES256) \
    void NAME(void* job, int t0, int t1, int tid) { \
        (void) t0; (void) t1; \
        band((jam_band_job*) job, tid, KIND, REPACK, BYTES256); \
    }
BAND(jam_q4k_band32_avx512,   KIND_K32, repack_q4k,   JAM_Q4K_BYTES)
BAND(jam_q5k_band32_avx512,   KIND_K32, repack_q5k,   JAM_Q5K_BYTES)
BAND(jam_q6k_band32_avx512,   KIND_K16, repack_q6k,   JAM_Q6K_BYTES)
BAND(jam_q8_0_band32_avx512,  KIND_F32, repack_q8_0,  8 * JAM_Q8_0_BYTES)
BAND(jam_q4_0_band32_avx512,  KIND_F32, repack_q4_0,  8 * JAM_Q4_0_BYTES)
BAND(jam_q5_0_band32_avx512,  KIND_F32, repack_q5_0,  8 * JAM_Q5_0_BYTES)
BAND(jam_mxfp4_band32_avx512, KIND_F32, repack_mxfp4, 8 * JAM_MXFP4_BYTES)
BAND(jam_q1_0_band32_avx512,  KIND_F32, repack_q1_0,  2 * JAM_Q1_0_BYTES)
BAND(jam_q2_0_band32_avx512,  KIND_F32, repack_q2_0,  4 * JAM_Q2_0_BYTES)
BAND(jam_pq2_0_band32_avx512, KIND_F32, repack_pq2_0, 2 * JAM_PQ2_0_BYTES)
BAND(jam_ptq1_0_band32_avx512, KIND_F32, repack_ptq1_0, 2 * JAM_PTQ1_0_BYTES)
