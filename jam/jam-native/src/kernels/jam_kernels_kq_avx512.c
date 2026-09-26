/* Quantized-weight @ F32 -> F32 prefill bands, AVX-512-VNNI: one tile kernel for the K-quants (Q4_K / Q5_K /
 * Q6_K: integer sub-block scales) and the 32-block quants (Q8_0 / Q4_0 / Q5_0 / MXFP4: one float scale
 * per 32-block, applied with an FMA instead of vpmulld).
 *
 * Why this shape (measured on Zen 5, single core, L1-resident operands, fraction of the vpdpbusd peak):
 *   16 rows x 4 cols, one activation broadcast per vpdpbusd ..................... 0.52
 *   32 rows x 4 cols, each broadcast feeds two 16-row weight vectors ............ 0.90-0.95
 *   per-sub-block scale in float (cvt + mul + fmadd + fnmadd) ................... 0.62-0.70
 *   per-sub-block scale in int (vpmulld + vpaddd) ............................... 0.80-0.87
 * The broadcast loads are the cost, not the weight loads, so a band of 32 rows shares them; and the
 * sub-block scales stay integer, so the epilogue is two integer ops per accumulator.
 *
 * Math. Activations are quantized per 256-block (the Q8_K scheme, llama.cpp's precision for K-quants):
 * x ~ xd * q, q in s8, stored BIASED as u8 = q + 128 so it can be the unsigned vpdpbusd operand; the
 * weight bytes are the signed operand (Q4_K/Q5_K: the code 0..31; Q6_K: code - 32). For one weight row r
 * and one 256-block:
 *   dot = d * sum_sub sc[sub] * sum_e w[e] q[e]  -  dmin * sum_sub mn[sub] * sum_e q[e]      (Q6_K: no min)
 * The int dot sees (q + 128) instead of q, which adds 128 * sum_sub sc[sub] * sum_e w[e]: a per-row
 * constant, precomputed at repack (K16). The min term is the s16 dot of the 8 mins against the 8
 * per-32 activation sums (4 vpdpwssd), then everything scales by xd in float once per 256-block.
 *
 * Repack (per 16-row group, per 256-block, one 5568-byte blob):
 *   W    [64][16][4] s8   4-k groups: vpdpbusd lanes = rows, 4 bytes = 4 consecutive k
 *   SC   [nsub][16] i32   per-sub-block scale per row (nsub = 8 for Q4_K/Q5_K, 16 for Q6_K)
 *   MN   [4][16][2] i16   min pairs (2p, 2p+1) per row - Q4_K/Q5_K
 *   K16  [16] i32         128 * sum_sub sc * sum_e w   (the u8 bias correction)
 *   D    [16] f32, DMIN [16] f32
 * jam_q4k_job carries the phase-1 output: xq = u8 [seq][k], dx = f32 [seq][k/256] (per-256 scale),
 * xsum = i16 [seq][k/32] (per-32 sums of the s8 codes). The buffers are the same as the 32-block bands',
 * just read with this layout. */
#include "jam_internal.h"
#include "jam_kquant.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdatomic.h>
#include "kernels/jam_mxfp4.h"

#define KQ_W    0
#define KQ_SC   4096
#define KQ_MN   5120
#define KQ_K    5376
#define KQ_D    5440
#define KQ_DMIN 5504
/* JAM_KQ_BLOB (jam_internal.h) = 5568 = the layout above, 64-byte aligned.
 * The 32-block quants use the same blob per 8 blocks (a ragged last one when k % 256 != 0): W holds the
 * s8 codes, SC the 8 per-block float scales dw [8][16] followed by the bias corrections cw = dw * 128 *
 * sum_e w [8][16]. Their activations keep a scale PER 32-BLOCK (jam_kq_quant32: llama.cpp's Q8_0
 * activation precision for these formats), so the epilogue is per block: f += cvt(acc) * dw * xd - cw * xd. */

static inline float kq_h2f(uint16_t h) { return _cvtsh_ss(h); }

/* ---- phase 1: quantize one activation row: per-256 scale, u8 = s8 + 128, per-32 s8 sums ---- */
void jam_kq_quant(void* arg, int s0, int s1, int tid) {
    (void) tid;
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int k = J->dim1, nb = k / JAM_QK, sblocks = (nb + 7) / 8;   /* ragged last super-block */
    const __m512i bias = _mm512_set1_epi8((char) 0x80);
    for (int s = s0; s < s1; s++) {
        const float* x = J->rhs + (size_t) s * J->rhs_stride;
        uint8_t* xq = (uint8_t*) J->xq + (size_t) s * k;
        float* xd = J->dx + (size_t) s * sblocks;
        int16_t* xs = (int16_t*) J->xsum + (size_t) s * sblocks * 8;
        for (int sb = 0; sb < sblocks; sb++, x += JAM_QKK, xq += JAM_QKK, xs += 8) {
            const int nbs = nb - sb * 8 < 8 ? nb - sb * 8 : 8;
            __m512 mx = _mm512_setzero_ps();
            for (int i = 0; i < 2 * nbs; i++) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(x + i * 16)));
            float amax = _mm512_reduce_max_ps(mx);
            float d = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            xd[sb] = d;
            __m512 v = _mm512_set1_ps(inv);
            for (int sub = 0; sub < nbs; sub++) {
                __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + sub * 32), v));
                __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + sub * 32 + 16), v));
                xs[sub] = (int16_t) _mm512_reduce_add_epi32(_mm512_add_epi32(q0, q1));
                __m256i b = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm512_cvtsepi32_epi8(q0)),
                                                    _mm512_cvtsepi32_epi8(q1), 1);
                _mm256_storeu_si256((__m256i*) (xq + sub * 32), _mm256_xor_si256(b, _mm512_castsi512_si256(bias)));
            }
        }
    }
}

/* ---- phase 1 for the 32-block quants: per-32 scale, u8 = s8 + 128 (no sums needed) ---- */
void jam_kq_quant32(void* arg, int s0, int s1, int tid) {
    (void) tid;
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int k = J->dim1, nb = k / JAM_QK;
    const __m256i bias = _mm256_set1_epi8((char) 0x80);
    for (int s = s0; s < s1; s++) {
        const float* x = J->rhs + (size_t) s * J->rhs_stride;
        uint8_t* xq = (uint8_t*) J->xq + (size_t) s * k;
        float* xd = J->dx + (size_t) s * nb;
        for (int b = 0; b < nb; b++, x += JAM_QK, xq += JAM_QK) {
            __m512 a0 = _mm512_loadu_ps(x), a1 = _mm512_loadu_ps(x + 16);
            float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a0), _mm512_abs_ps(a1)));
            float d = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            xd[b] = d;
            __m512 v = _mm512_set1_ps(inv);
            __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(a0, v)), q1 = _mm512_cvtps_epi32(_mm512_mul_ps(a1, v));
            __m256i q = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm512_cvtsepi32_epi8(q0)),
                                                _mm512_cvtsepi32_epi8(q1), 1);
            _mm256_storeu_si256((__m256i*) xq, _mm256_xor_si256(q, bias));
        }
    }
}

/* ---- per-format decode of one row's 256-block: codes (signed), scales, mins, d, dmin ---- */
typedef void (*kq_decode_fn)(const uint8_t* w, int8_t* v, int* sc, int* mn, float* d, float* dmin);

static void dec_q4k(const uint8_t* w, int8_t* v, int* sc, int* mn, float* d, float* dmin) {
    *d = kq_h2f(*(const uint16_t*) w); *dmin = kq_h2f(*(const uint16_t*) (w + 2));
    uint8_t s[8], m[8]; jam_q4k_scales_mins(w + 4, s, m);
    for (int j = 0; j < 8; j++) { sc[j] = s[j]; mn[j] = m[j]; }
    const uint8_t* q = w + 16;
    for (int g = 0; g < 4; g++)
        for (int i = 0; i < 32; i++) {
            v[g * 64 + i]      = (int8_t) (q[g * 32 + i] & 0xF);
            v[g * 64 + 32 + i] = (int8_t) (q[g * 32 + i] >> 4);
        }
}

static void dec_q5k(const uint8_t* w, int8_t* v, int* sc, int* mn, float* d, float* dmin) {
    *d = kq_h2f(*(const uint16_t*) w); *dmin = kq_h2f(*(const uint16_t*) (w + 2));
    uint8_t s[8], m[8]; jam_q4k_scales_mins(w + 4, s, m);
    for (int j = 0; j < 8; j++) { sc[j] = s[j]; mn[j] = m[j]; }
    const uint8_t* qh = w + 16; const uint8_t* q = w + 48;
    for (int g = 0; g < 4; g++)
        for (int i = 0; i < 32; i++) {
            v[g * 64 + i]      = (int8_t) ((q[g * 32 + i] & 0xF) | (((qh[i] >> (2 * g))     & 1) << 4));
            v[g * 64 + 32 + i] = (int8_t) ((q[g * 32 + i] >> 4)  | (((qh[i] >> (2 * g + 1)) & 1) << 4));
        }
}

static void dec_q6k(const uint8_t* w, int8_t* v, int* sc, int* mn, float* d, float* dmin) {
    const uint8_t* ql = w; const uint8_t* qh = w + 128; const int8_t* s = (const int8_t*) (w + 192);
    *d = kq_h2f(*(const uint16_t*) (w + 208)); *dmin = 0.0f;
    for (int j = 0; j < 16; j++) sc[j] = s[j];
    (void) mn;
    for (int h = 0; h < 2; h++) {
        const uint8_t* qlb = ql + h * 64; const uint8_t* qhb = qh + h * 32;
        for (int l = 0; l < 32; l++) {
            int hi = qhb[l];
            v[h * 128 + l]      = (int8_t) (((qlb[l]      & 0xF) | ((hi >> 0 & 3) << 4)) - 32);
            v[h * 128 + 32 + l] = (int8_t) (((qlb[32 + l] & 0xF) | ((hi >> 2 & 3) << 4)) - 32);
            v[h * 128 + 64 + l] = (int8_t) (((qlb[l]      >> 4)  | ((hi >> 4 & 3) << 4)) - 32);
            v[h * 128 + 96 + l] = (int8_t) (((qlb[32 + l] >> 4)  | ((hi >> 6 & 3) << 4)) - 32);
        }
    }
}

/* ---- repack one 16-row group into its blobs (one per 256-block) ----
 * Vectorized: each row's 256 codes are decoded into a 16 x 256 B stage with AVX-512 byte ops, then four
 * 16x16 dword transposes (4-stage vpermt2d butterflies) turn [row][k-group] into the kernel's
 * [k-group][row] layout. The scalar decoders above are the readable reference and serve the tails. */
static inline int kq_sum_u8(__m256i v) {          /* sum of 32 unsigned bytes */
    __m256i s = _mm256_sad_epu8(v, _mm256_setzero_si256());
    __m128i t = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    return (int) (_mm_cvtsi128_si64(t) + _mm_extract_epi64(t, 1));
}
static inline void kq_sum_u8_halves(__m256i v, int* a, int* b) {   /* sums of bytes 0..15 and 16..31 */
    __m256i s = _mm256_sad_epu8(v, _mm256_setzero_si256());
    __m128i lo = _mm256_castsi256_si128(s), hi = _mm256_extracti128_si256(s, 1);
    *a = (int) (_mm_cvtsi128_si64(lo) + _mm_extract_epi64(lo, 1));
    *b = (int) (_mm_cvtsi128_si64(hi) + _mm_extract_epi64(hi, 1));
}

/* decode one row's 256-block into stage[256] (unsigned codes, Q6_K biased +32) + per-sub-block sums */
static inline void vdec_q4k(const uint8_t* w, uint8_t* st, int* qsum, int* sc, int* mn, float* d, float* dmin) {
    *d = kq_h2f(*(const uint16_t*) w); *dmin = kq_h2f(*(const uint16_t*) (w + 2));
    uint8_t s[8], m[8]; jam_q4k_scales_mins(w + 4, s, m);
    for (int j = 0; j < 8; j++) { sc[j] = s[j]; mn[j] = m[j]; }
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (int g = 0; g < 4; g++) {
        __m256i v = _mm256_loadu_si256((const __m256i*) (w + 16 + g * 32));
        __m256i lo = _mm256_and_si256(v, m4), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
        _mm256_storeu_si256((__m256i*) (st + g * 64), lo);
        _mm256_storeu_si256((__m256i*) (st + g * 64 + 32), hi);
        qsum[2 * g] = kq_sum_u8(lo); qsum[2 * g + 1] = kq_sum_u8(hi);
    }
}
static inline void vdec_q5k(const uint8_t* w, uint8_t* st, int* qsum, int* sc, int* mn, float* d, float* dmin) {
    *d = kq_h2f(*(const uint16_t*) w); *dmin = kq_h2f(*(const uint16_t*) (w + 2));
    uint8_t s[8], m[8]; jam_q4k_scales_mins(w + 4, s, m);
    for (int j = 0; j < 8; j++) { sc[j] = s[j]; mn[j] = m[j]; }
    const __m256i m4 = _mm256_set1_epi8(0x0F), m1 = _mm256_set1_epi8(0x01);
    const __m256i qh = _mm256_loadu_si256((const __m256i*) (w + 16));
    for (int g = 0; g < 4; g++) {
        __m256i v = _mm256_loadu_si256((const __m256i*) (w + 48 + g * 32));
        __m256i lo = _mm256_and_si256(v, m4), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
        lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g), m1), 4));
        hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g + 1), m1), 4));
        _mm256_storeu_si256((__m256i*) (st + g * 64), lo);
        _mm256_storeu_si256((__m256i*) (st + g * 64 + 32), hi);
        qsum[2 * g] = kq_sum_u8(lo); qsum[2 * g + 1] = kq_sum_u8(hi);
    }
}
static inline void vdec_q6k(const uint8_t* w, uint8_t* st, int* qsum, int* sc, int* mn, float* d, float* dmin) {
    *d = kq_h2f(*(const uint16_t*) (w + 208)); *dmin = 0.0f; (void) mn;
    const int8_t* s = (const int8_t*) (w + 192);
    for (int j = 0; j < 16; j++) sc[j] = s[j];
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(0x03), b32 = _mm256_set1_epi8(32);
    for (int h = 0; h < 2; h++) {
        __m256i l0 = _mm256_loadu_si256((const __m256i*) (w + h * 64));
        __m256i l1 = _mm256_loadu_si256((const __m256i*) (w + h * 64 + 32));
        __m256i qh = _mm256_loadu_si256((const __m256i*) (w + 128 + h * 32));
        __m256i q[4];
        q[0] = _mm256_or_si256(_mm256_and_si256(l0, m4),                       _mm256_slli_epi16(_mm256_and_si256(qh, m3), 4));
        q[1] = _mm256_or_si256(_mm256_and_si256(l1, m4),                       _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m3), 4));
        q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), m3), 4));
        q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), m3), 4));
        for (int j = 0; j < 4; j++) {
            int a, b; kq_sum_u8_halves(q[j], &a, &b);
            qsum[h * 8 + j * 2] = a - 32 * 16; qsum[h * 8 + j * 2 + 1] = b - 32 * 16;
            _mm256_storeu_si256((__m256i*) (st + h * 128 + j * 32), _mm256_sub_epi8(q[j], b32));
        }
    }
}
typedef void (*kq_vdecode_fn)(const uint8_t* w, uint8_t* st, int* qsum, int* sc, int* mn, float* d, float* dmin);

/* 16x16 dword transpose of v[0..15] (four vpermt2d butterfly stages); on exit v[g] holds column
 * kq_tr_col[g] of the input, i.e. dword kq_tr_col[g] of every row. */
static const int kq_tr_col[16] = { 0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15 };
static inline void kq_tr16(__m512i* v) {
    const __m512i a1 = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    const __m512i b1 = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
    for (int i = 0; i < 16; i += 2) {
        __m512i a = v[i], b = v[i + 1];
        v[i] = _mm512_permutex2var_epi32(a, a1, b); v[i + 1] = _mm512_permutex2var_epi32(a, b1, b);
    }
    const __m512i a2 = _mm512_setr_epi32(0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29);
    const __m512i b2 = _mm512_setr_epi32(2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31);
    for (int i = 0; i < 16; i += 4)
        for (int j = 0; j < 2; j++) {
            __m512i a = v[i + j], b = v[i + 2 + j];
            v[i + j] = _mm512_permutex2var_epi32(a, a2, b); v[i + 2 + j] = _mm512_permutex2var_epi32(a, b2, b);
        }
    const __m512i a3 = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
    const __m512i b3 = _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
    for (int i = 0; i < 16; i += 8)
        for (int j = 0; j < 4; j++) {
            __m512i a = v[i + j], b = v[i + 4 + j];
            v[i + j] = _mm512_permutex2var_epi32(a, a3, b); v[i + 4 + j] = _mm512_permutex2var_epi32(a, b3, b);
        }
    const __m512i a4 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
    const __m512i b4 = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
    for (int i = 0; i < 8; i++) {
        __m512i a = v[i], b = v[i + 8];
        v[i] = _mm512_permutex2var_epi32(a, a4, b); v[i + 8] = _mm512_permutex2var_epi32(a, b4, b);
    }
}

static void kq_repack_group(const uint8_t* wbase, int64_t w_stride, int sblocks, uint8_t* blob,
                            int nsub, int has_min, size_t kbytes, kq_vdecode_fn dec) {
    __attribute__((aligned(64))) uint8_t stage[16 * JAM_QKK];
    for (int sb = 0; sb < sblocks; sb++) {
        uint8_t* b = blob + (size_t) sb * JAM_KQ_BLOB;
        for (int r = 0; r < 16; r++) {
            int qsum[16], sc[16], mn[8]; float d, dmin;
            dec(wbase + r * w_stride + sb * kbytes, stage + r * JAM_QKK, qsum, sc, mn, &d, &dmin);
            int32_t kk = 0;
            for (int sub = 0; sub < nsub; sub++) {
                ((int32_t*) (b + KQ_SC))[sub * 16 + r] = sc[sub];
                kk += sc[sub] * qsum[sub];
            }
            ((int32_t*) (b + KQ_K))[r] = 128 * kk;
            ((float*) (b + KQ_D))[r] = d;
            ((float*) (b + KQ_DMIN))[r] = dmin;
            if (has_min)
                for (int p = 0; p < 4; p++) {
                    ((int16_t*) (b + KQ_MN))[p * 32 + r * 2]     = (int16_t) mn[2 * p];
                    ((int16_t*) (b + KQ_MN))[p * 32 + r * 2 + 1] = (int16_t) mn[2 * p + 1];
                }
        }
        for (int c = 0; c < 4; c++) {                      /* 64 dwords per row = 4 transposes */
            __m512i v[16];
            for (int r = 0; r < 16; r++) v[r] = _mm512_load_si512((const void*) (stage + r * JAM_QKK + c * 64));
            kq_tr16(v);
            for (int g = 0; g < 16; g++)
                _mm512_store_si512((void*) (b + KQ_W + (c * 16 + kq_tr_col[g]) * 64), v[g]);
        }
    }
}

/* exact float dot for the < 16-row tail (same decode, so the tail agrees with the band bit-for-bit
 * on the weight side) */
static float kq_dot_scalar(const uint8_t* w, const float* x, int sblocks, int nsub, int has_min,
                           size_t kbytes, kq_decode_fn dec) {
    const int subk = JAM_QKK / nsub;
    float acc = 0.0f;
    for (int sb = 0; sb < sblocks; sb++, w += kbytes, x += JAM_QKK) {
        int8_t v[JAM_QKK]; int sc[16], mn[8]; float d, dmin;
        dec(w, v, sc, mn, &d, &dmin);
        for (int sub = 0; sub < nsub; sub++) {
            float ds = d * sc[sub], m = has_min ? dmin * mn[sub] : 0.0f;
            for (int e = 0; e < subk; e++) acc += (ds * v[sub * subk + e] - m) * x[sub * subk + e];
        }
    }
    return acc;
}

/* ---- 32-block quants: decode one block into 32 s8 codes + its float scale ---- */
typedef void (*kq_dec32_fn)(const uint8_t* w, __m256i* codes, float* d);
static inline int kq_sum_s8(__m256i v) {          /* sum of 32 signed bytes */
    return kq_sum_u8(_mm256_xor_si256(v, _mm256_set1_epi8((char) 0x80))) - 128 * 32;
}
static inline __m256i kq_nib32(const uint8_t* q) {  /* 16 packed nibbles -> [low nibbles | high nibbles] */
    __m128i v = _mm_loadu_si128((const __m128i*) q);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_and_si128(v, m4)),
                                   _mm_and_si128(_mm_srli_epi16(v, 4), m4), 1);
}
static void dec32_q8_0(const uint8_t* w, __m256i* codes, float* d) {
    *d = kq_h2f(*(const uint16_t*) w);
    *codes = _mm256_loadu_si256((const __m256i*) (w + 2));
}
static void dec32_q4_0(const uint8_t* w, __m256i* codes, float* d) {
    *d = kq_h2f(*(const uint16_t*) w);
    *codes = _mm256_sub_epi8(kq_nib32(w + 2), _mm256_set1_epi8(8));
}
static void dec32_q5_0(const uint8_t* w, __m256i* codes, float* d) {
    *d = kq_h2f(*(const uint16_t*) w);
    uint32_t qh; memcpy(&qh, w + 2, 4);                       /* bit j = 5th bit of element j */
    __m256i hi = _mm256_and_si256(_mm256_movm_epi8((__mmask32) qh), _mm256_set1_epi8(0x10));
    *codes = _mm256_sub_epi8(_mm256_or_si256(kq_nib32(w + 6), hi), _mm256_set1_epi8(16));
}
static void dec32_mxfp4(const uint8_t* w, __m256i* codes, float* d) {
    static const int8_t lut[16] = { JAM_MXFP4_CODES };
    *d = jam_mxfp4_dhalf(w[0]);
    __m256i t = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) lut));
    *codes = _mm256_shuffle_epi8(t, kq_nib32(w + 1));
}

/* repack one 16-row group of a 32-block quant: blobs of 8 blocks (the last may be ragged) */
static void kq_repack_group32(const uint8_t* wbase, int64_t w_stride, int nb, uint8_t* blob,
                              size_t kbytes, kq_dec32_fn dec) {
    __attribute__((aligned(64))) uint8_t stage[16 * JAM_QKK];
    const int sblocks = (nb + 7) / 8;
    for (int sb = 0; sb < sblocks; sb++) {
        uint8_t* b = blob + (size_t) sb * JAM_KQ_BLOB;
        const int nbs = nb - sb * 8 < 8 ? nb - sb * 8 : 8;
        for (int r = 0; r < 16; r++) {
            const uint8_t* w = wbase + r * w_stride + (size_t) sb * 8 * kbytes;
            for (int bl = 0; bl < nbs; bl++, w += kbytes) {
                __m256i codes; float d;
                dec(w, &codes, &d);
                _mm256_store_si256((__m256i*) (stage + r * JAM_QKK + bl * 32), codes);
                ((float*) (b + KQ_SC))[bl * 16 + r] = d;                                          /* dw */
                ((float*) (b + KQ_SC + 512))[bl * 16 + r] = d * 128.0f * (float) kq_sum_s8(codes);  /* cw */
            }
        }
        for (int c = 0; c < (nbs + 1) / 2; c++) {          /* 2 blocks = 16 dwords per row per transpose */
            __m512i v[16];
            for (int r = 0; r < 16; r++) v[r] = _mm512_load_si512((const void*) (stage + r * JAM_QKK + c * 64));
            kq_tr16(v);
            for (int g = 0; g < 16; g++)
                _mm512_store_si512((void*) (b + KQ_W + (c * 16 + kq_tr_col[g]) * 64), v[g]);
        }
    }
}

/* exact float dots for the < 16-row tails of the 32-block quants (nblk = 32-blocks) */
static float tail32_q8_0(const uint8_t* w, const float* x, int nblk) { return jam_q8_0_dot_f32(w, nblk, x); }
static float tail32_q4_0(const uint8_t* w, const float* x, int nblk) { return jam_q4_0_dot_f32(w, nblk, x); }
static float tail32_q5_0(const uint8_t* w, const float* x, int nblk) { return jam_q5_0_dot_f32(w, nblk, x); }
static float tail32_mxfp4(const uint8_t* w, const float* x, int nblk) {
    static const int8_t lut[16] = { JAM_MXFP4_CODES };
    float acc = 0.0f;
    for (int bl = 0; bl < nblk; bl++, w += 17, x += JAM_QK) {
        float dh = jam_mxfp4_dhalf(w[0]);
        float sum = 0.0f;
        for (int j = 0; j < 16; j++) sum += lut[w[1 + j] & 0xF] * x[j] + lut[w[1 + j] >> 4] * x[j + 16];
        acc += dh * sum;
    }
    return acc;
}

/* ---- the tile: ng x 16 rows (blobs b0, b1) by nc columns, all 256-blocks ----
 * fmode selects the epilogue: 0 = integer sub-block scales (K-quants), 1 = float per-32-block scales.
 * All of ng/nc/nsub/has_min are compile-time at every call site (always_inline + literal args). */
static inline __attribute__((always_inline))
void kq_tile(const int ng, const int nc, const int nsub, const int has_min, const int fmode,
             const uint8_t* b0, const uint8_t* b1, int sblocks, int nsub_last,
             const uint8_t* x0, int64_t xstride, const float* xd0, int64_t xdstride, const int16_t* xs0,
             float* out0, int64_t ldc) {
    const int subk = JAM_QKK / nsub, ni = subk / 4;
    __m512 f[2][4];
    for (int g = 0; g < 2; g++) for (int c = 0; c < 4; c++) f[g][c] = _mm512_setzero_ps();
    for (int sb = 0; sb < sblocks; sb++) {
        const uint8_t* w0 = b0 + (size_t) sb * JAM_KQ_BLOB;
        const uint8_t* w1 = b1 + (size_t) sb * JAM_KQ_BLOB;
        const uint8_t* xb = x0 + sb * JAM_QKK;
        const int nsub_sb = sb == sblocks - 1 ? nsub_last : nsub;
        __m512i s1[2][4];
        for (int g = 0; g < 2; g++) for (int c = 0; c < 4; c++) s1[g][c] = _mm512_setzero_si512();
        for (int sub = 0; sub < nsub_sb; sub++) {
            __m512i acc[2][4];
            for (int g = 0; g < 2; g++) for (int c = 0; c < 4; c++) acc[g][c] = _mm512_setzero_si512();
            const uint8_t* wp0 = w0 + KQ_W + sub * subk * 16;
            const uint8_t* wp1 = w1 + KQ_W + sub * subk * 16;
            const uint8_t* xp = xb + sub * subk;
            for (int i = 0; i < ni; i++) {
                __m512i v0 = _mm512_load_si512((const void*) (wp0 + i * 64));
                __m512i v1 = ng > 1 ? _mm512_load_si512((const void*) (wp1 + i * 64)) : v0;
                #pragma GCC unroll 4
                for (int c = 0; c < nc; c++) {
                    int32_t a4; memcpy(&a4, xp + c * xstride + i * 4, 4);
                    __m512i a = _mm512_set1_epi32(a4);
                    acc[0][c] = _mm512_dpbusd_epi32(acc[0][c], a, v0);
                    if (ng > 1) acc[1][c] = _mm512_dpbusd_epi32(acc[1][c], a, v1);
                }
            }
            if (fmode) {                                   /* per-block: f += cvt(acc)*dw*xd - cw*xd */
                __m512 dw0 = _mm512_load_ps(w0 + KQ_SC + sub * 64), cw0 = _mm512_load_ps(w0 + KQ_SC + 512 + sub * 64);
                __m512 dw1 = ng > 1 ? _mm512_load_ps(w1 + KQ_SC + sub * 64) : dw0;
                __m512 cw1 = ng > 1 ? _mm512_load_ps(w1 + KQ_SC + 512 + sub * 64) : cw0;
                #pragma GCC unroll 4
                for (int c = 0; c < nc; c++) {
                    __m512 xdv = _mm512_set1_ps(xd0[c * xdstride + sb * 8 + sub]);
                    f[0][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[0][c]), _mm512_mul_ps(dw0, xdv), f[0][c]);
                    f[0][c] = _mm512_fnmadd_ps(cw0, xdv, f[0][c]);
                    if (ng > 1) {
                        f[1][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[1][c]), _mm512_mul_ps(dw1, xdv), f[1][c]);
                        f[1][c] = _mm512_fnmadd_ps(cw1, xdv, f[1][c]);
                    }
                }
            } else {                                       /* integer sub-block scale: vpmulld + vpaddd */
                __m512i sc0 = _mm512_load_si512((const void*) (w0 + KQ_SC + sub * 64));
                __m512i sc1 = ng > 1 ? _mm512_load_si512((const void*) (w1 + KQ_SC + sub * 64)) : sc0;
                #pragma GCC unroll 4
                for (int c = 0; c < nc; c++) {
                    s1[0][c] = _mm512_add_epi32(s1[0][c], _mm512_mullo_epi32(acc[0][c], sc0));
                    if (ng > 1) s1[1][c] = _mm512_add_epi32(s1[1][c], _mm512_mullo_epi32(acc[1][c], sc1));
                }
            }
        }
        if (fmode) continue;
        for (int g = 0; g < ng; g++) {
            const uint8_t* w = g ? w1 : w0;
            __m512i k16 = _mm512_load_si512((const void*) (w + KQ_K));
            __m512 d16 = _mm512_load_ps(w + KQ_D), dm16 = _mm512_load_ps(w + KQ_DMIN);
            __m512i mn0 = _mm512_setzero_si512(), mn1 = mn0, mn2 = mn0, mn3 = mn0;
            if (has_min) {
                mn0 = _mm512_load_si512((const void*) (w + KQ_MN));
                mn1 = _mm512_load_si512((const void*) (w + KQ_MN + 64));
                mn2 = _mm512_load_si512((const void*) (w + KQ_MN + 128));
                mn3 = _mm512_load_si512((const void*) (w + KQ_MN + 192));
            }
            #pragma GCC unroll 4
            for (int c = 0; c < nc; c++) {
                __m512 v = _mm512_mul_ps(d16, _mm512_cvtepi32_ps(_mm512_sub_epi32(s1[g][c], k16)));
                if (has_min) {
                    const int16_t* xs = xs0 + c * sblocks * 8 + sb * 8;
                    int32_t p0, p1, p2, p3;
                    memcpy(&p0, xs, 4); memcpy(&p1, xs + 2, 4); memcpy(&p2, xs + 4, 4); memcpy(&p3, xs + 6, 4);
                    __m512i s2 = _mm512_dpwssd_epi32(_mm512_setzero_si512(), mn0, _mm512_set1_epi32(p0));
                    s2 = _mm512_dpwssd_epi32(s2, mn1, _mm512_set1_epi32(p1));
                    s2 = _mm512_dpwssd_epi32(s2, mn2, _mm512_set1_epi32(p2));
                    s2 = _mm512_dpwssd_epi32(s2, mn3, _mm512_set1_epi32(p3));
                    v = _mm512_fnmadd_ps(dm16, _mm512_cvtepi32_ps(s2), v);
                }
                f[g][c] = _mm512_fmadd_ps(v, _mm512_set1_ps(xd0[c * xdstride + sb]), f[g][c]);
            }
        }
    }
    for (int g = 0; g < ng; g++)
        for (int c = 0; c < nc; c++) _mm512_storeu_ps(out0 + c * ldc + g * 16, f[g][c]);
}

/* ---- band driver: one 32-row tile range; per tile: repack the 1-2 groups, sweep the columns ---- */
typedef void (*kq_repack_fn)(const uint8_t* wbase, int64_t w_stride, int nb, uint8_t* blob);
typedef float (*kq_tail_fn)(const uint8_t* w, const float* x, int nb);   /* nb = 32-blocks */

static inline __attribute__((always_inline))
void kq_band(const jam_q4k_job* J, int t0, int t1, int tid, const int nsub, const int has_min,
             const int fmode, kq_repack_fn repack, kq_tail_fn tail) {
    const int k = J->dim1, nb = k / JAM_QK, sblocks = (nb + 7) / 8, seq = J->seq;
    const int nsub_last = fmode ? nb - 8 * (sblocks - 1) : nsub;
    const int64_t ldc = J->out_stride;
    const uint8_t* xq = (const uint8_t*) J->xq;
    const int16_t* xs = (const int16_t*) J->xsum;
    jam_repack* rp = &J->repack[tid];
    uint8_t* blob0 = rp->qs;
    uint8_t* blob1 = rp->qs + (size_t) sblocks * JAM_KQ_BLOB;
    /* Tiles are claimed dynamically, not by the static [t0, t1) slice: a worker on an SMT sibling or
     * the slower CCD then just takes fewer tiles instead of stalling the whole call. */
    (void) t0; (void) t1;
    const int ntiles = (J->dim0 + JAM_VNNI_BAND - 1) / JAM_VNNI_BAND;
    for (int tile; (tile = atomic_fetch_add_explicit(&((jam_q4k_job*) J)->next_tile, 1, memory_order_relaxed)) < ntiles; ) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int ng = (row_end - row) / 16;
        for (int g = 0; g < ng; g++)
            repack(J->w + (int64_t) (row + g * 16) * J->w_stride, J->w_stride, nb, g ? blob1 : blob0);
        /* Column sweep over the whole k: the band's blobs stream from L2 (k-chunking them into L1 was
         * measured flat - the kernel is FP-pipe bound, not load bound). */
        for (int c0 = 0; c0 < seq && ng; c0 += 4) {
            int nc = seq - c0 < 4 ? seq - c0 : 4;
            const uint8_t* x0 = xq + (size_t) c0 * k;
            const int64_t xdstride = fmode ? nb : sblocks;        /* per-32 vs per-256 activation scales */
            const float* xd0 = J->dx + (size_t) c0 * xdstride;
            const int16_t* xs0 = xs + (size_t) c0 * sblocks * 8;
            float* out0 = J->out + (int64_t) c0 * ldc + row;
#define KQ_CALL(NG, NC) kq_tile(NG, NC, nsub, has_min, fmode, blob0, blob1, sblocks, nsub_last, x0, k, xd0, xdstride, xs0, out0, ldc)
            if (ng == 2) switch (nc) { case 4: KQ_CALL(2, 4); break; case 3: KQ_CALL(2, 3); break;
                                       case 2: KQ_CALL(2, 2); break; default: KQ_CALL(2, 1); break; }
            else         switch (nc) { case 4: KQ_CALL(1, 4); break; case 3: KQ_CALL(1, 3); break;
                                       case 2: KQ_CALL(1, 2); break; default: KQ_CALL(1, 1); break; }
#undef KQ_CALL
        }
        for (int r = row + ng * 16; r < row_end; r++)               /* < 16-row tail: exact float dot */
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] = tail(J->w + (int64_t) r * J->w_stride,
                                                     J->rhs + (size_t) s * J->rhs_stride, nb);
    }
}

/* per-format repack + tail wrappers (constants folded) */
static void rp_q4k(const uint8_t* w, int64_t ws, int nb, uint8_t* b) { kq_repack_group(w, ws, nb / 8, b, 8, 1, JAM_Q4K_BYTES, vdec_q4k); }
static void rp_q5k(const uint8_t* w, int64_t ws, int nb, uint8_t* b) { kq_repack_group(w, ws, nb / 8, b, 8, 1, JAM_Q5K_BYTES, vdec_q5k); }
static void rp_q6k(const uint8_t* w, int64_t ws, int nb, uint8_t* b) { kq_repack_group(w, ws, nb / 8, b, 16, 0, JAM_Q6K_BYTES, vdec_q6k); }
static float tail_q4k(const uint8_t* w, const float* x, int nb) { return kq_dot_scalar(w, x, nb / 8, 8, 1, JAM_Q4K_BYTES, dec_q4k); }
static float tail_q5k(const uint8_t* w, const float* x, int nb) { return kq_dot_scalar(w, x, nb / 8, 8, 1, JAM_Q5K_BYTES, dec_q5k); }
static float tail_q6k(const uint8_t* w, const float* x, int nb) { return kq_dot_scalar(w, x, nb / 8, 16, 0, JAM_Q6K_BYTES, dec_q6k); }
static void rp_q8_0(const uint8_t* w, int64_t ws, int nb, uint8_t* b) { kq_repack_group32(w, ws, nb, b, JAM_Q8_0_BYTES, dec32_q8_0); }
static void rp_q4_0(const uint8_t* w, int64_t ws, int nb, uint8_t* b) { kq_repack_group32(w, ws, nb, b, JAM_Q4_0_BYTES, dec32_q4_0); }
static void rp_q5_0(const uint8_t* w, int64_t ws, int nb, uint8_t* b) { kq_repack_group32(w, ws, nb, b, JAM_Q5_0_BYTES, dec32_q5_0); }
static void rp_mxfp4(const uint8_t* w, int64_t ws, int nb, uint8_t* b) { kq_repack_group32(w, ws, nb, b, 17, dec32_mxfp4); }

void jam_kq_band_q4k(void* arg, int t0, int t1, int tid) { kq_band((const jam_q4k_job*) arg, t0, t1, tid, 8, 1, 0, rp_q4k, tail_q4k); }
void jam_kq_band_q5k(void* arg, int t0, int t1, int tid) { kq_band((const jam_q4k_job*) arg, t0, t1, tid, 8, 1, 0, rp_q5k, tail_q5k); }
void jam_kq_band_q6k(void* arg, int t0, int t1, int tid) { kq_band((const jam_q4k_job*) arg, t0, t1, tid, 16, 0, 0, rp_q6k, tail_q6k); }
void jam_kq_band_q8_0(void* arg, int t0, int t1, int tid) { kq_band((const jam_q4k_job*) arg, t0, t1, tid, 8, 0, 1, rp_q8_0, tail32_q8_0); }
void jam_kq_band_q4_0(void* arg, int t0, int t1, int tid) { kq_band((const jam_q4k_job*) arg, t0, t1, tid, 8, 0, 1, rp_q4_0, tail32_q4_0); }
void jam_kq_band_q5_0(void* arg, int t0, int t1, int tid) { kq_band((const jam_q4k_job*) arg, t0, t1, tid, 8, 0, 1, rp_q5_0, tail32_q5_0); }
void jam_kq_band_mxfp4(void* arg, int t0, int t1, int tid) { kq_band((const jam_q4k_job*) arg, t0, t1, tid, 8, 0, 1, rp_mxfp4, tail32_mxfp4); }
