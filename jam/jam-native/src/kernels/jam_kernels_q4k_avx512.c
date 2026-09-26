/* AVX-512-VNNI leftovers of the first 16-row repack bands: the shared s8 phase-1 (per-32 scale + per-16
 * exact sums) that the Q1_0 band and the AVX-VNNI/AVX2 8-row bands still consume, the token-major 16-row
 * store, and the Q1_0 packed-sign-bit band. Every other quant's AVX-512 prefill band is the 32x4 tile in
 * jam_kernels_kq_avx512.c. Output token-major: C[col*ldc + row] (feature contig). */
#include "jam_internal.h"
#include "jam_kquant.h"
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

static inline float q4k_h2f(uint16_t h) { return _cvtsh_ss(h); }

/* Column register-tile width: the _nr band kernels process JAM_VNNI_NR activation columns per weight-block
 * load, so the weight decode + per-block float scale amortize across them. NR=4 fits the 32 zmm; NR=8 spills.
 * Shared by the q4_0 / q6k / q5k tiled bands. */
#ifndef JAM_VNNI_NR
#define JAM_VNNI_NR 4   /* tunable via -DJAM_VNNI_NR */
#endif

/* ---- phase 1: quantize one activation row to s8 + per-32 scale + per-16 sums ---- */
static void quantize_row_q8s(const float* x, int kblocks, int8_t* xq, float* dx, float* xs) {
    for (int b = 0; b < kblocks; b++, x += JAM_QK, xq += JAM_QK, xs += 2) {
        __m512 a0 = _mm512_loadu_ps(x), a1 = _mm512_loadu_ps(x + 16);
        float max = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a0), _mm512_abs_ps(a1)));
        float d = max / 127.0f, inv = max > 0.0f ? 127.0f / max : 0.0f;
        dx[b] = d;
        xs[0] = _mm512_reduce_add_ps(a0);
        xs[1] = _mm512_reduce_add_ps(a1);
        __m512 v = _mm512_set1_ps(inv);
        _mm_storeu_si128((__m128i*) xq,        _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(a0, v))));
        _mm_storeu_si128((__m128i*) (xq + 16), _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(a1, v))));
    }
}

void jam_q4k_quant(void* arg, int s0, int s1, int tid) {
    (void) tid;
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    for (int s = s0; s < s1; s++)
        quantize_row_q8s(J->rhs + (size_t) s * J->rhs_stride, J->kblocks,
                         J->xq + (size_t) s * J->kblocks * JAM_QK,
                         J->dx + (size_t) s * J->kblocks,
                         J->xsum + (size_t) s * J->kblocks * 2);
}

/* store a 16-feature result for token `col` into token-major C[col*ldc + row..row+15] - contiguous. */
static inline void q4k_store16(__m512 f, float* out, int64_t ldc, int row, int col) {
    _mm512_storeu_ps(out + (int64_t) col * ldc + row, f);
}

/* ================= Q1_0 16-row VNNI repack - packed SIGN-BIT band (1 bit/weight in scratch) =========
 * block_q1_0 = { fp16 d; uint8_t qs[16] } = 18B / 128 elems; value = d·(2b-1), b ∈ {0,1} (LSB-first).
 * The bit is the UNSIGNED vpdpbusd operand - expanded to 0/1 bytes IN-REGISTER via one kmov +
 * vmovdqu8{k}{z} of a hoisted all-ones vector per 16-row×4-elem group - and the s8 activation
 * broadcast is the signed one:
 *     d·Σ(2b-1)·x  =  d·( 2·dx·Σ b·x_q  −  Σx )
 * where −Σx uses the exact per-16 float activation sums: the same deferred-offset scheme as Q4_0's
 * −8 (offset 1 ⇒ a smaller correction). The repack stores 8 permuted 64-bit masks per 32-elem block
 * (bit r*4+e = row r, elem g*4+e): 64 B/block/16rows - 8× smaller than the Q8_0 band, L1-resident  -
 * plus one d per row/block (mw is not needed: the offset scale IS d). */
#include "jam_q1_0.h"   /* jam_blk_q1_0 layout + the shared exact scalar dot */

static void repack_q1_0_group16(const uint8_t* wbase, int64_t w_stride, int nb,
                                uint8_t* qs, float* dw) {
    uint64_t* m = (uint64_t*) qs;                 /* [nb][8] masks (the scratch is 64-aligned) */
    for (int B = 0; B < nb; B++, m += 8, dw += 16) {
        uint64_t mg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int r = 0; r < 16; r++) {
            const uint8_t* blk = wbase + r * w_stride + (B >> 2) * JAM_Q1_0_BYTES;
            uint32_t bits = *(const uint32_t*) (blk + 2 + (B & 3) * 4);   /* 32 sign bits, LSB-first */
            for (int g = 0; g < 8; g++)
                mg[g] |= (uint64_t)((bits >> (g * 4)) & 0xF) << (r * 4);
            dw[r] = q4k_h2f(*(const uint16_t*) blk);
        }
        for (int g = 0; g < 8; g++) m[g] = mg[g];
    }
}

/* 16-row dot of the packed sign bits against one activation column -> 16 f32 partials. */
static inline __m512 q1_0_block16(const uint8_t* qs, const float* dw,
                                  const int8_t* x, const float* dx, const float* xs, int nb) {
    const __m512i one = _mm512_set1_epi8(1);
    const uint64_t* m = (const uint64_t*) qs;
    __m512 f = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m512i acc = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++)
            acc = _mm512_dpbusd_epi32(acc, _mm512_maskz_mov_epi8((__mmask64) m[g], one),
                                      _mm512_set1_epi32(((const int*) x)[g]));
        __m512 dwv = _mm512_load_ps(dw);
        f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(dwv, _mm512_set1_ps(2.0f * dx[b])), f);
        f = _mm512_fnmadd_ps(dwv, _mm512_set1_ps(xs[2 * b] + xs[2 * b + 1]), f);
        m += 8; dw += 16; x += JAM_QK;
    }
    return f;
}

/* Column register-tiled: each mask expanded ONCE (kmov + masked move), vpdpbusd'd against JAM_VNNI_NR
 * activation columns - the weight side of this band is so small that the expands are the only weight
 * cost, amortized across the NR columns like the other bands' decodes. */
static inline void q1_0_block16_nr(const uint8_t* qs, const float* dw, const int8_t* xq,
                                   const float* dx, const float* xs, int s0, int nb,
                                   int64_t ldc, float* out, int r) {
    const __m512i one = _mm512_set1_epi8(1);
    const uint64_t* m = (const uint64_t*) qs;
    __m512 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        f[c] = _mm512_setzero_ps();
        x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK;
    }
    for (int b = 0; b < nb; b++) {
        __m512i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i w = _mm512_maskz_mov_epi8((__mmask64) m[g], one);   /* expand once, reuse NR cols */
            for (int c = 0; c < JAM_VNNI_NR; c++)
                acc[c] = _mm512_dpbusd_epi32(acc[c], w, _mm512_set1_epi32(((const int*) x[c])[g]));
        }
        __m512 dwv = _mm512_load_ps(dw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            int64_t idx = (int64_t)(s0 + c) * nb + b;
            f[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[c]),
                                   _mm512_mul_ps(dwv, _mm512_set1_ps(2.0f * dx[idx])), f[c]);
            f[c] = _mm512_fnmadd_ps(dwv, _mm512_set1_ps(xs[idx * 2] + xs[idx * 2 + 1]), f[c]);
            x[c] += JAM_QK;
        }
        m += 8; dw += 16;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) q4k_store16(f[c], out, ldc, r, s0 + c);
}

void jam_q1_0_repack_band(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int nb = J->kblocks, seq = J->seq;           /* nb = k/32 activation blocks; k % 128 == 0 */
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 15 < row_end; r += 16, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * nb * 64;   /* packed bits: 64 B/block */
            float* dw = rp->dw + (int64_t) group * nb * 16;
            repack_q1_0_group16(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw);
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                q1_0_block16_nr(qs, dw, J->xq, J->dx, J->xsum, s, nb, ldc, J->out, r);
            for (; s < seq; s++)               /* column tail (< NR) */
                q4k_store16(q1_0_block16(qs, dw, J->xq + (int64_t) s * nb * JAM_QK,
                                         J->dx + (int64_t) s * nb, J->xsum + (int64_t) s * nb * 2, nb),
                            J->out, ldc, r, s);
        }
        for (int r = row + group * 16; r < row_end; r++)        /* <16-row tail: scalar exact dot */
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    jam_q1_0_dot_f32(J->w + (int64_t) r * J->w_stride, J->dim1 / 128,
                                     J->rhs + (int64_t) s * J->rhs_stride);
    }
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

/* ---- K-quant int8 FLOOR at full width: Q4_K @ s8 activations, 512-bit VNNI ----
 * The seq < JAM_VNNI_MIN_SEQ path (DECODE: no repack amortization) used to fall to the SSE3
 * floor - quarter-width SIMD on the machines that matter. Same job contract and math as
 * jam_mm_q4k_sse3, restructured for gemv reality:
 *   - 512-bit pair dots: a sub-block PAIR's activations are one contiguous 64B load and one
 *     vpdpbusd yields both sub-block dots (lanes 0-7 = lo nibbles' sub-block, 8-15 = hi's);
 *   - the per-pair scale vector [fLo x8 | fHi x8] is ONE permutexvar over a per-super-block
 *     f16 register (ad8 * d * sc8, decoded with SIMD, not byte-by-byte);
 *   - the dmin*min term is one 256-bit fnmadd per super-block (mn8 * ad8*as8 * dmin);
 *   - 4-row streaming per column: the four activation vectors and ad8/as8 stay in registers
 *     across the row group, so per extra row only the weight stream and dots are paid. */

/* Q4_K packed-scale decode, branch- and loop-free (the llama.cpp 3-mask word trick): 12 scale
 * bytes -> sc[8] in out[0..1], mn[8] in out[2..3], one byte each. ~10 int ops vs the 4-pass
 * byte loop - the header decode was ~40% of the gemv floor's uops. */
static inline void q4k_scales_mins_words(const uint8_t* b, uint32_t out[4]) {
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t u0, u1, u2;
    __builtin_memcpy(&u0, b, 4); __builtin_memcpy(&u1, b + 4, 4); __builtin_memcpy(&u2, b + 8, 4);
    out[3] = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
    out[2] = u1 & kmask1;
    out[1] = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
    out[0] = u0 & kmask1;
}

static const int32_t q4k_pair_idx[4][16] = {
    {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1}, {2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3},
    {4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5}, {6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7}};

void jam_mm_q4k_avx512vnni(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad; const float* AS = J->asum;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t)(J->lda / JAM_QKK) * JAM_Q4K_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (int j = 0; j < n; ++j) {
        const int8_t* aq = AQ + (size_t) j * k;
        const float* ad = AD + (size_t) j * nb;
        const float* as = AS + (size_t) j * nb;
        int i = rb;
        for (; i + 3 < re; i += 4) {
            const uint8_t* w0 = (const uint8_t*) (W + (size_t) i * w_stride);
            const uint8_t* w1 = w0 + w_stride, *w2 = w1 + w_stride, *w3 = w2 + w_stride;
            __m512 f0 = _mm512_setzero_ps(), f1 = _mm512_setzero_ps();
            __m512 f2 = _mm512_setzero_ps(), f3 = _mm512_setzero_ps();
            float  m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;
            for (int B = 0; B < sblocks; ++B) {
                _mm_prefetch((const char*) (w0 + 4 * JAM_Q4K_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*) (w1 + 4 * JAM_Q4K_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*) (w2 + 4 * JAM_Q4K_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*) (w3 + 4 * JAM_Q4K_BYTES), _MM_HINT_T0);
                __m512i acts[4];
                for (int g = 0; g < 4; ++g)
                    acts[g] = _mm512_loadu_si512((const void*) (aq + (size_t) B * JAM_QKK + g * 64));
                __m256 ad8 = _mm256_loadu_ps(ad + B * 8);
                __m256 asd8 = _mm256_mul_ps(_mm256_loadu_ps(as + B * 8), ad8); /* ad*as per sub-block */
                /* per row: decode once, 4 pair dots, 1 vectorized scale apply, 1 vectorized min */
                #define Q4K_ROW(WR, FR, MR) do { \
                    float d = jam_half2float(*(const uint16_t*) (WR)); \
                    float dmin = jam_half2float(*(const uint16_t*) ((WR) + 2)); \
                    uint32_t sm[4]; q4k_scales_mins_words((WR) + 4, sm); \
                    __m256 sc8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) sm))); \
                    __m256 mn8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (sm + 2)))); \
                    __m256 f8 = _mm256_mul_ps(_mm256_mul_ps(ad8, sc8), _mm256_set1_ps(d)); \
                    __m512 f16 = _mm512_insertf32x8(_mm512_castps256_ps512(f8), f8, 1); \
                    const uint8_t* q = (WR) + 16; \
                    for (int g = 0; g < 4; ++g) { \
                        __m256i qb = _mm256_loadu_si256((const __m256i*) (q + g * 32)); \
                        __m256i lo = _mm256_and_si256(qb, m4); \
                        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4); \
                        __m512i wp = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1); \
                        __m512i idot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wp, acts[g]); \
                        __m512 scale = _mm512_permutexvar_ps( \
                                _mm512_load_si512((const void*) q4k_pair_idx[g]), f16); \
                        (FR) = _mm512_fmadd_ps(_mm512_cvtepi32_ps(idot), scale, (FR)); \
                    } \
                    __m256 mm = _mm256_mul_ps(mn8, asd8); \
                    (MR) += dmin * hsum256(mm); \
                } while (0)
                Q4K_ROW(w0, f0, m0);
                Q4K_ROW(w1, f1, m1);
                Q4K_ROW(w2, f2, m2);
                Q4K_ROW(w3, f3, m3);
                #undef Q4K_ROW
                w0 += JAM_Q4K_BYTES; w1 += JAM_Q4K_BYTES; w2 += JAM_Q4K_BYTES; w3 += JAM_Q4K_BYTES;
            }
            C[(size_t) j * ldc + i]     = _mm512_reduce_add_ps(f0) - m0;
            C[(size_t) j * ldc + i + 1] = _mm512_reduce_add_ps(f1) - m1;
            C[(size_t) j * ldc + i + 2] = _mm512_reduce_add_ps(f2) - m2;
            C[(size_t) j * ldc + i + 3] = _mm512_reduce_add_ps(f3) - m3;
        }
        for (; i < re; ++i) {   /* row tail */
            const uint8_t* w = (const uint8_t*) (W + (size_t) i * w_stride);
            __m512 f0 = _mm512_setzero_ps(); float mAcc = 0.f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q4K_BYTES) {
                __m512i acts[4];
                for (int g = 0; g < 4; ++g)
                    acts[g] = _mm512_loadu_si512((const void*) (aq + (size_t) B * JAM_QKK + g * 64));
                __m256 ad8 = _mm256_loadu_ps(ad + B * 8);
                __m256 asd8 = _mm256_mul_ps(_mm256_loadu_ps(as + B * 8), ad8);
                float d = jam_half2float(*(const uint16_t*) w);
                float dmin = jam_half2float(*(const uint16_t*) (w + 2));
                uint32_t sm[4]; q4k_scales_mins_words(w + 4, sm);
                __m256 sc8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) sm)));
                __m256 mn8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (sm + 2))));
                __m256 f8 = _mm256_mul_ps(_mm256_mul_ps(ad8, sc8), _mm256_set1_ps(d));
                __m512 f16 = _mm512_insertf32x8(_mm512_castps256_ps512(f8), f8, 1);
                const uint8_t* q = w + 16;
                for (int g = 0; g < 4; ++g) {
                    __m256i qb = _mm256_loadu_si256((const __m256i*) (q + g * 32));
                    __m256i lo = _mm256_and_si256(qb, m4);
                    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4);
                    __m512i wp = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                    __m512i idot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wp, acts[g]);
                    __m512 scale = _mm512_permutexvar_ps(
                            _mm512_load_si512((const void*) q4k_pair_idx[g]), f16);
                    f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(idot), scale, f0);
                }
                mAcc += dmin * hsum256(_mm256_mul_ps(mn8, asd8));
            }
            C[(size_t) j * ldc + i] = _mm512_reduce_add_ps(f0) - mAcc;
        }
    }
}

/* ---- Q5_K / Q6_K decode floors at full width: the jam_mm_q4k_avx512vnni scheme (pair dots over
 * 64 contiguous activations, per-pair scale permute, 4-row streaming per column), one per quant.
 * KQ_VNNI_ROWS is the shared row skeleton: PRO() runs once per super-block per column and defines
 * what ROW reads (acts[4] is already loaded); ROW(w, f, m) folds one super-block of one row into
 * the f32 lanes f (and the min term m). */
#define KQ_VNNI_ROWS(BYTES, PRO, ROW) \
    for (int j = 0; j < n; ++j) { \
        const int8_t* aq = AQ + (size_t) j * k; \
        const float* ad = AD + (size_t) j * nb; \
        const float* as = AS ? AS + (size_t) j * nb : ad; (void) as; \
        int i = rb; \
        for (; i + 3 < re; i += 4) { \
            const uint8_t* w0 = (const uint8_t*) (W + (size_t) i * w_stride); \
            const uint8_t* w1 = w0 + w_stride, *w2 = w1 + w_stride, *w3 = w2 + w_stride; \
            __m512 f0 = _mm512_setzero_ps(), f1 = f0, f2 = f0, f3 = f0; \
            float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f; \
            for (int B = 0; B < sblocks; ++B) { \
                _mm_prefetch((const char*) (w0 + 4 * (BYTES)), _MM_HINT_T0); \
                _mm_prefetch((const char*) (w1 + 4 * (BYTES)), _MM_HINT_T0); \
                _mm_prefetch((const char*) (w2 + 4 * (BYTES)), _MM_HINT_T0); \
                _mm_prefetch((const char*) (w3 + 4 * (BYTES)), _MM_HINT_T0); \
                __m512i acts[4]; \
                for (int g = 0; g < 4; ++g) \
                    acts[g] = _mm512_loadu_si512((const void*) (aq + (size_t) B * JAM_QKK + g * 64)); \
                PRO(); \
                ROW(w0, f0, m0); ROW(w1, f1, m1); ROW(w2, f2, m2); ROW(w3, f3, m3); \
                w0 += (BYTES); w1 += (BYTES); w2 += (BYTES); w3 += (BYTES); \
            } \
            C[(size_t) j * ldc + i]     = _mm512_reduce_add_ps(f0) - m0; \
            C[(size_t) j * ldc + i + 1] = _mm512_reduce_add_ps(f1) - m1; \
            C[(size_t) j * ldc + i + 2] = _mm512_reduce_add_ps(f2) - m2; \
            C[(size_t) j * ldc + i + 3] = _mm512_reduce_add_ps(f3) - m3; \
        } \
        for (; i < re; ++i) {   /* row tail */ \
            const uint8_t* w0 = (const uint8_t*) (W + (size_t) i * w_stride); \
            __m512 f0 = _mm512_setzero_ps(); float m0 = 0.f; \
            for (int B = 0; B < sblocks; ++B, w0 += (BYTES)) { \
                __m512i acts[4]; \
                for (int g = 0; g < 4; ++g) \
                    acts[g] = _mm512_loadu_si512((const void*) (aq + (size_t) B * JAM_QKK + g * 64)); \
                PRO(); \
                ROW(w0, f0, m0); \
            } \
            C[(size_t) j * ldc + i] = _mm512_reduce_add_ps(f0) - m0; \
        } \
    }

/* Q5_K: Q4_K plus the 5th bit plane. Layout d(f16) dmin(f16) scales[12] qh[32] qs[128]; sub-block
 * 2g takes bit 2g of qh[e] into nibble bit 4, sub-block 2g+1 takes bit 2g+1. Weights stay 0..31
 * (unsigned dpbusd operand); the dmin*min term is the same float correction as Q4_K. */
void jam_mm_q5k_avx512vnni(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad; const float* AS = J->asum;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t)(J->lda / JAM_QKK) * JAM_Q5K_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F), bit1 = _mm256_set1_epi8(1);
    #define Q5K_PRO() \
        __m256 ad8 = _mm256_loadu_ps(ad + B * 8); \
        __m256 asd8 = _mm256_mul_ps(_mm256_loadu_ps(as + B * 8), ad8)
    #define Q5K_ROW(WR, FR, MR) do { \
        float d = jam_half2float(*(const uint16_t*) (WR)); \
        float dmin = jam_half2float(*(const uint16_t*) ((WR) + 2)); \
        uint32_t sm[4]; q4k_scales_mins_words((WR) + 4, sm); \
        __m256 sc8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) sm))); \
        __m256 mn8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (sm + 2)))); \
        __m256 f8 = _mm256_mul_ps(_mm256_mul_ps(ad8, sc8), _mm256_set1_ps(d)); \
        __m512 f16 = _mm512_insertf32x8(_mm512_castps256_ps512(f8), f8, 1); \
        __m256i qh = _mm256_loadu_si256((const __m256i*) ((WR) + 16)); \
        const uint8_t* q = (WR) + 48; \
        for (int g = 0; g < 4; ++g) { \
            __m256i qb = _mm256_loadu_si256((const __m256i*) (q + g * 32)); \
            __m256i hl = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g), bit1), 4); \
            __m256i hh = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g + 1), bit1), 4); \
            __m256i lo = _mm256_or_si256(_mm256_and_si256(qb, m4), hl); \
            __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb, 4), m4), hh); \
            __m512i wp = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1); \
            __m512i idot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wp, acts[g]); \
            __m512 scale = _mm512_permutexvar_ps(_mm512_loadu_si512((const void*) q4k_pair_idx[g]), f16); \
            (FR) = _mm512_fmadd_ps(_mm512_cvtepi32_ps(idot), scale, (FR)); \
        } \
        (MR) += dmin * hsum256(_mm256_mul_ps(mn8, asd8)); \
    } while (0)
    KQ_VNNI_ROWS(JAM_Q5K_BYTES, Q5K_PRO, Q5K_ROW)
    #undef Q5K_PRO
    #undef Q5K_ROW
}

/* Q6_K: value = d*sc16*(q-32), q 6-bit = ql nibble | qh 2 bits << 4, one int8 scale per 16 elements,
 * no min term. Layout ql[128] qh[64] scales[16] d(f16). Per half h of the super-block, the 64 ql
 * bytes' low nibbles are sub-blocks (0,1) and the high nibbles (2,3) - each pair is one 64-byte
 * dpbusd against 64 contiguous activations. The weight stays unsigned 0..63 for dpbusd; the -32 is
 * folded into the accumulator start: bias[g] = -32 * (per-4-lane activation sums), computed once
 * per super-block per column and shared by all rows. The 16 per-16 scales map 1:1 onto the 16 f32
 * lanes of the two pair dots of a half (lane l of pair g reads scale 4g + l/4). */
static const int32_t q6k_lane_idx[4][16] = {
    {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3},     {4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7},
    {8,8,8,8, 9,9,9,9, 10,10,10,10, 11,11,11,11}, {12,12,12,12, 13,13,13,13, 14,14,14,14, 15,15,15,15}};
static const int32_t q6k_ad_idx[16] = {0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7};

void jam_mm_q6k_avx512vnni(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad; const float* AS = NULL;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t)(J->lda / JAM_QKK) * JAM_Q6K_BYTES;
    const __m512i m4z = _mm512_set1_epi8(0x0F), c32 = _mm512_set1_epi8(32), zero = _mm512_setzero_si512();
    const __m512i m30 = _mm512_set1_epi8(0x30);
    /* per-16-bit-lane shift counts: low 256 bits one count, high 256 bits the other */
    const __m512i shl42 = _mm512_inserti64x4(_mm512_set1_epi16(4), _mm256_set1_epi16(2), 1);
    const __m512i shr02 = _mm512_inserti64x4(_mm512_set1_epi16(0), _mm256_set1_epi16(2), 1);
    const __m512i adidx = _mm512_loadu_si512((const void*) q6k_ad_idx);
    #define Q6K_PRO() \
        __m512i bias[4]; \
        for (int g = 0; g < 4; ++g) bias[g] = _mm512_sub_epi32(zero, _mm512_dpbusd_epi32(zero, c32, acts[g])); \
        __m512 adx = _mm512_permutexvar_ps(adidx, _mm512_castps256_ps512(_mm256_loadu_ps(ad + B * 8)))
    #define Q6K_ROW(WR, FR, MR) do { \
        float d = jam_half2float(*(const uint16_t*) ((WR) + 208)); \
        __m512 f16 = _mm512_mul_ps(_mm512_mul_ps(adx, _mm512_set1_ps(d)), _mm512_cvtepi32_ps( \
                _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) ((WR) + 192))))); \
        for (int h = 0; h < 2; ++h) { \
            __m512i L = _mm512_loadu_si512((const void*) ((WR) + h * 64)); \
            /* the 2-bit planes: HH = [qh | qh]; sub-blocks (0,1) want bits 0-1 / 2-3 of each */ \
            /* byte at bits 4-5, (2,3) want bits 4-5 / 6-7 there - one variable shift per pair */ \
            __m512i HH = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) ((WR) + 128 + h * 32))); \
            __m512i p01 = _mm512_and_si512(_mm512_sllv_epi16(HH, shl42), m30); \
            __m512i p23 = _mm512_and_si512(_mm512_srlv_epi16(HH, shr02), m30); \
            __m512i w01 = _mm512_or_si512(_mm512_and_si512(L, m4z), p01); \
            __m512i w23 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(L, 4), m4z), p23); \
            __m512i d01 = _mm512_dpbusd_epi32(bias[2 * h], w01, acts[2 * h]); \
            __m512i d23 = _mm512_dpbusd_epi32(bias[2 * h + 1], w23, acts[2 * h + 1]); \
            (FR) = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d01), \
                    _mm512_permutexvar_ps(_mm512_loadu_si512((const void*) q6k_lane_idx[2 * h]), f16), (FR)); \
            (FR) = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d23), \
                    _mm512_permutexvar_ps(_mm512_loadu_si512((const void*) q6k_lane_idx[2 * h + 1]), f16), (FR)); \
        } \
        (void) (MR); \
    } while (0)
    KQ_VNNI_ROWS(JAM_Q6K_BYTES, Q6K_PRO, Q6K_ROW)
    #undef Q6K_PRO
    #undef Q6K_ROW
}
#undef KQ_VNNI_ROWS
