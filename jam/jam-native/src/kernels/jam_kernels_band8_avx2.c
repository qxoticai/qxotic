/* AVX2 8-row repack bands (ymm + maddubs): the quantized prefill below AVX-512-VNNI. This TU is
 * compiled with -mavx2 -mfma -mf16c only (part of the jam_avx2 target). The K-quants and MXFP4 run
 * here on every such CPU without VNNI (Haswell..Rocket Lake, Zen 2/3); Q8_0, Q5_0 and Q4_0 too.
 *
 * The file builds a second time, with JAM_BAND8_VNNI, as the AVX-VNNI clients' K-quant and MXFP4
 * bands (jam_kernels_band8_avxvnni.c): the same repack, tiles and float epilogues, where a dot is one
 * vpdpbusd into int32 lanes instead of the maddubs ladder below, so the int16 chains and their bounds
 * do not apply. Those CPUs have their own Q8_0, Q5_0 and Q4_0 bands (jam_kernels_avxvnni.c).
 *
 * The same per-worker scratch as the VNNI bands (jam_repack from ensure_kquant, whose 32-row blob
 * also holds a band's four 8-row groups: at most 256 B of codes per group per 32-block), the same
 * two-phase launch, scales and corrections in float. Only the int8 dot differs: no vpdpbusd, so the
 * base ladder is maddubs(w_u8, x_s8) -> madd(1) -> add int32 - and wherever the per-maddubs int16
 * bound allows, products ACCUMULATE IN INT16 (wrap/saturation provably unreachable) with ONE
 * deferred madd, cutting the ladder to ~2 ops/group:
 *   Q4_K, Q4_0 : peak 2*15*127 = 3810/maddubs -> all 8 groups chain (8*3810 = 30480 < 32767)
 *   MXFP4      : peak 2*255*12 = 6120        -> chains of 4 (24480)
 *   Q5_K       : peak 2*31*127 = 7874        -> chains of 4 (31496)
 *   Q6_K       : peak 2*63*127 = 16002       -> NO chaining (2 already wrap); per-group madd
 *   Q8_0, Q5_0 : sign trick peak 2*128*127 = 32512 (w = -128) -> NO chaining; per-group madd
 * Weight operands are UNSIGNED except for Q8_0 and Q5_0 (sign trick: u = |w|, s = sign(x,w)) and
 * MXFP4 (u = x + 128, s = w). */
#include "jam_internal.h"
#include "jam_kquant.h"
#include "jam_mxfp4.h"
#include "jam_decode_x86_256.h"   /* decoders, jam_put32_rows8, jam_quant32_256, JAM_BAND8_DRIVER */
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include <string.h>

/* Activation columns per register tile, per quant - measured optima (m=4096 n=512 k=2048,
 * capped at avx2): Q4_K peaks at 4 (dual lo/hi accumulators fill the file), the single-accumulator
 * chained kernels at 6, and Q8_0's sign-trick pipeline at 8 (the extra independent columns hide
 * the abs/sign latency). */
#define B8_NR_Q4K 4
#define B8_NR_KQ  6
#define B8_NR_Q8  8

#ifdef JAM_BAND8_VNNI
#define B8_NAME(quant) jam_##quant##_band8_avxvnni
/* one 32-byte weight group (8 rows x 4 elems) against a 4-elem activation broadcast: one vpdpbusd.
 * The chains of the maddubs build are plain int32 accumulators here, and folding one is an add. */
static inline __m256i b8_dot(__m256i acc, __m256i wu, __m256i xb) { return _mm256_dpbusd_avx_epi32(acc, wu, xb); }
static inline __m256i b8_chain(__m256i i32, __m256i wu, __m256i xb) { return _mm256_dpbusd_avx_epi32(i32, wu, xb); }
static inline __m256i b8_chain4(__m256i i32, __m256i wu, __m256i xb) { return _mm256_dpbusd_avx_epi32(i32, wu, xb); }
static inline __m256i b8_fold(__m256i acc, __m256i i32) { return _mm256_add_epi32(acc, i32); }
#define B8_UNROLL2
#else
#define B8_NAME(quant) jam_##quant##_band8_avx2
/* one 32-byte weight group (8 rows x 4 elems) dotted against a 4-elem activation broadcast */
static inline __m256i b8_dot(__m256i acc, __m256i wu, __m256i xb) {
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wu, xb), _mm256_set1_epi16(1)));
}

/* int16-chained variants: accumulate maddubs products in int16 - callers guarantee the chain length
 * keeps |sum| under 32767 (see the header table) - then fold the chain to int32 with one madd. Integer
 * adds are associative, so gcc rebalances a fully unrolled chain into a sum tree that computes every
 * maddubs of the tile first and spills them (q4k_block8_nr: 167 stack accesses, ~30% behind clang). Each
 * chain shape is written so neither compiler can: the 8-chains unroll by 2 only (a 2-term tree per step),
 * the 4-chains (b8_chain4) use a SATURATING add, which is not associative and equals the wrapping add
 * because the bound is never reached. Not saturating everywhere: on the 8-chains it costs clang ~6%. */
static inline __m256i b8_chain(__m256i i16, __m256i wu, __m256i xb) {
    return _mm256_add_epi16(i16, _mm256_maddubs_epi16(wu, xb));
}
static inline __m256i b8_chain4(__m256i i16, __m256i wu, __m256i xb) {
    return _mm256_adds_epi16(i16, _mm256_maddubs_epi16(wu, xb));
}
#define B8_UNROLL2 _Pragma("GCC unroll 2")   /* honoured by gcc and clang alike */

static inline __m256i b8_fold(__m256i acc, __m256i i16) {
    return _mm256_add_epi32(acc, _mm256_madd_epi16(i16, _mm256_set1_epi16(1)));
}

#endif

/* bit `shift` (mask 1) or bits shift..shift+1 (mask 3) of every byte of v, moved to bit 4 */
static inline __m256i b8_plane(__m256i v, int shift, int mask) {
    return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(v, shift), _mm256_set1_epi8((char) mask)), 4);
}

#ifndef JAM_BAND8_VNNI   /* phase 1 is the AVX2 build's, for both */
/* ---- phase 1: F32 activations -> int8 (jam_quant32_256) + per-32 scale + per-16 RAW float sums,
 * which feed the exact dmin*min / -32-offset corrections ---- */
static void quantize_row_q8s_avx2(const float* x, int kblocks, int8_t* xq, float* dx, float* xs) {
    for (int b = 0; b < kblocks; b++, x += JAM_QK, xq += JAM_QK, xs += 2) {
        __m256 a[4]; __m256i q[4];
        dx[b] = jam_quant32_256(x, xq, a, q);
        xs[0] = jam_hsum8_256(_mm256_add_ps(a[0], a[1]));   /* raw per-16 sums (exact min term) */
        xs[1] = jam_hsum8_256(_mm256_add_ps(a[2], a[3]));
    }
}

void jam_band8_quant_avx2(void* arg, int s0, int s1, int tid) {
    (void) tid;
    const jam_band_job* J = (const jam_band_job*) arg;
    for (int s = s0; s < s1; s++)
        quantize_row_q8s_avx2(J->rhs + (size_t) s * J->rhs_stride, J->kblocks,
                              J->xq + (size_t) s * J->kblocks * JAM_QK,
                              J->dx + (size_t) s * J->kblocks,
                              J->xsum + (size_t) s * J->kblocks * 2);
}
#endif

/* ================= Q4_K ============================================================== */

/* repack 8 Q4_K rows: packed dual-nibble bytes (lo = sbLo, hi = sbHi), 256 B/sub-block-pair
 * per group of 8; per-row sub-block scales dw / mins mw at [sb*8 + r]. */
static void repack_q4k_group8(const uint8_t* wbase, int64_t w_stride, int sblocks,
                              uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += JAM_Q4K_BYTES) {
            float d = jam_half_at(w), dmin = jam_half_at(w + 2);
            uint8_t sc[8], mn[8];
            jam_q4k_scales_mins(w + 4, sc, mn);
            const uint8_t* q = w + 16;
            for (int g = 0; g < 4; g++) {
                int sbLo = B * 8 + g * 2, sbHi = sbLo + 1, pairIdx = B * 4 + g;
                dw[sbLo * 8 + r] = d * sc[g * 2];     mw[sbLo * 8 + r] = dmin * mn[g * 2];
                dw[sbHi * 8 + r] = d * sc[g * 2 + 1]; mw[sbHi * 8 + r] = dmin * mn[g * 2 + 1];
                const uint8_t* qg = q + g * 32;
                uint8_t* dst = qs + (int64_t) pairIdx * 256 + r * 4;
                for (int k = 0; k < 8; k++)
                    memcpy(dst + k * 32, qg + k * 4, 4);
            }
        }
    }
}

/* single activation column: 8-row partials for the seq tail */
static inline __m256 q4k_block8(const uint8_t* qs, const float* dw, const float* mw,
                                const int8_t* x, const float* d, const float* s, int pairs) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 f = _mm256_setzero_ps();
    for (int p = 0; p < pairs; p++) {
        __m256i iLo = _mm256_setzero_si256(), iHi = _mm256_setzero_si256();
        B8_UNROLL2 for (int g = 0; g < 8; g++) {   /* 8 chained maddubs stay under int16 (8*3810) */
            __m256i pk = _mm256_load_si256((const void*) (qs + g * 32));
            iLo = b8_chain(iLo, _mm256_and_si256(pk, m4), _mm256_set1_epi32(jam_load32(x + 4 * g)));
            iHi = b8_chain(iHi, _mm256_and_si256(_mm256_srli_epi16(pk, 4), m4),
                           _mm256_set1_epi32(jam_load32(x + JAM_QK + 4 * g)));
        }
        __m256i aLo = b8_fold(_mm256_setzero_si256(), iLo);
        __m256i aHi = b8_fold(_mm256_setzero_si256(), iHi);
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(aLo), _mm256_mul_ps(_mm256_load_ps(dw),     _mm256_set1_ps(d[2*p])),   f);
        f = _mm256_fnmadd_ps(_mm256_load_ps(mw),     _mm256_set1_ps(s[4*p] + s[4*p+1]), f);
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(aHi), _mm256_mul_ps(_mm256_load_ps(dw + 8), _mm256_set1_ps(d[2*p+1])), f);
        f = _mm256_fnmadd_ps(_mm256_load_ps(mw + 8), _mm256_set1_ps(s[4*p+2] + s[4*p+3]), f);
        qs += 256; dw += 16; mw += 16; x += 2 * JAM_QK;
    }
    return f;
}

/* register-tiled: decode each packed block once, dot against the per-quant column tile */
static inline void q4k_block8_nr(const uint8_t* qs, const float* dw, const float* mw, const int8_t* xq,
                                 const float* dx, const float* xs, int s0, int pairs, int kblocks,
                                 int64_t ldc, float* out, int r) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 f[B8_NR_Q4K];
    const int8_t* x[B8_NR_Q4K]; const float* d[B8_NR_Q4K]; const float* sc[B8_NR_Q4K];
    for (int c = 0; c < B8_NR_Q4K; c++) {
        f[c] = _mm256_setzero_ps();
        x[c]  = xq + (int64_t)(s0 + c) * kblocks * JAM_QK;
        d[c]  = dx + (int64_t)(s0 + c) * kblocks;
        sc[c] = xs + (int64_t)(s0 + c) * kblocks * 2;
    }
    for (int p = 0; p < pairs; p++) {
        __m256i iLo[B8_NR_Q4K], iHi[B8_NR_Q4K];
        for (int c = 0; c < B8_NR_Q4K; c++) { iLo[c] = _mm256_setzero_si256(); iHi[c] = _mm256_setzero_si256(); }
        B8_UNROLL2 for (int g = 0; g < 8; g++) {   /* 8 chained maddubs stay under int16 (8*3810) */
            __m256i pk = _mm256_load_si256((const void*) (qs + g * 32));
            __m256i lo = _mm256_and_si256(pk, m4);                       /* decode once, reuse NR cols */
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(pk, 4), m4);
            for (int c = 0; c < B8_NR_Q4K; c++) {
                iLo[c] = b8_chain(iLo[c], lo, _mm256_set1_epi32(jam_load32(x[c] + 4 * g)));
                iHi[c] = b8_chain(iHi[c], hi, _mm256_set1_epi32(jam_load32(x[c] + JAM_QK + 4 * g)));
            }
        }
        __m256 dwLo = _mm256_load_ps(dw), mwLo = _mm256_load_ps(mw);
        __m256 dwHi = _mm256_load_ps(dw + 8), mwHi = _mm256_load_ps(mw + 8);
        for (int c = 0; c < B8_NR_Q4K; c++) {
            __m256i aLo = b8_fold(_mm256_setzero_si256(), iLo[c]);
            __m256i aHi = b8_fold(_mm256_setzero_si256(), iHi[c]);
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(aLo), _mm256_mul_ps(dwLo, _mm256_set1_ps(d[c][2*p])),   f[c]);
            f[c] = _mm256_fnmadd_ps(mwLo, _mm256_set1_ps(sc[c][4*p] + sc[c][4*p+1]), f[c]);
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(aHi), _mm256_mul_ps(dwHi, _mm256_set1_ps(d[c][2*p+1])), f[c]);
            f[c] = _mm256_fnmadd_ps(mwHi, _mm256_set1_ps(sc[c][4*p+2] + sc[c][4*p+3]), f[c]);
            x[c] += 2 * JAM_QK;
        }
        qs += 256; dw += 16; mw += 16;
    }
    for (int c = 0; c < B8_NR_Q4K; c++)
        _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

void B8_NAME(q4k)(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int kblocks = J->kblocks, sblocks = J->dim1 / JAM_QKK, seq = J->seq;
    JAM_BAND8_DRIVER(kblocks * 128, kblocks * 8, B8_NR_Q4K,   /* packed: 128 B/32-block/group8 */
        repack_q4k_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, sblocks, qs, dw, mw),
        q4k_block8_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, kblocks / 2, kblocks, ldc, J->out, r),
        q4k_block8(qs, dw, mw, J->xq + (int64_t) s * kblocks * JAM_QK, J->dx + (int64_t) s * kblocks,
                   J->xsum + (int64_t) s * kblocks * 2, kblocks / 2),
        jam_q4k_dot_f32(J->w + (int64_t) r * J->w_stride, sblocks, J->rhs + (int64_t) s * J->rhs_stride));
}

/* ================= Q5_K - Q4_K scheme, 5-bit byte-expanded (256 B/sub-block/group8) ==== */

static void repack_q5k_group8(const uint8_t* wbase, int64_t w_stride, int sblocks,
                              uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += JAM_Q5K_BYTES) {
            float d = jam_half_at(w), dmin = jam_half_at(w + 2);
            uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
            const uint8_t* qh = w + 16; const uint8_t* q5 = w + 48;
            for (int g = 0; g < 4; g++) {
                int sbLo = B * 8 + g * 2, sbHi = sbLo + 1;
                dw[sbLo * 8 + r] = d * sc[g * 2];     mw[sbLo * 8 + r] = dmin * mn[g * 2];
                dw[sbHi * 8 + r] = d * sc[g * 2 + 1]; mw[sbHi * 8 + r] = dmin * mn[g * 2 + 1];
                const __m256i q = _mm256_loadu_si256((const __m256i*) (q5 + g * 32)), m4 = _mm256_set1_epi8(0x0F);
                const __m256i h = _mm256_loadu_si256((const __m256i*) qh);
                jam_put32_rows8(qs + (int64_t) sbLo * 256 + r * 4,                          /* 0..31 */
                                _mm256_or_si256(_mm256_and_si256(q, m4), b8_plane(h, 2 * g, 1)));
                jam_put32_rows8(qs + (int64_t) sbHi * 256 + r * 4,
                                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4), b8_plane(h, 2 * g + 1, 1)));
            }
        }
    }
}

static inline __m256 q5k_block8(const uint8_t* qs, const float* dw, const float* mw,
                                const int8_t* x, const float* d, const float* s, int subs) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int h = 0; h < 2; h++) {   /* chains of 4 maddubs stay under int16 (4*7874) */
            __m256i i16 = _mm256_setzero_si256();
            for (int g = h * 4; g < h * 4 + 4; g++)
                i16 = b8_chain4(i16, _mm256_load_si256((const void*) (qs + g * 32)),
                               _mm256_set1_epi32(jam_load32(x + 4 * g)));
            acc = b8_fold(acc, i16);
        }
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(_mm256_load_ps(dw), _mm256_set1_ps(d[b])), f);
        f = _mm256_fnmadd_ps(_mm256_load_ps(mw), _mm256_set1_ps(s[2 * b] + s[2 * b + 1]), f);
        qs += 256; dw += 8; mw += 8; x += JAM_QK;
    }
    return f;
}

static inline void q5k_block8_nr(const uint8_t* qs, const float* dw, const float* mw, const int8_t* xq,
                                 const float* dx, const float* xs, int s0, int kblocks, int64_t ldc,
                                 float* out, int r) {
    __m256 f[B8_NR_KQ];
    const int8_t* x[B8_NR_KQ]; const float* d[B8_NR_KQ]; const float* sc[B8_NR_KQ];
    for (int c = 0; c < B8_NR_KQ; c++) {
        f[c] = _mm256_setzero_ps();
        x[c]  = xq + (int64_t)(s0 + c) * kblocks * JAM_QK;
        d[c]  = dx + (int64_t)(s0 + c) * kblocks;
        sc[c] = xs + (int64_t)(s0 + c) * kblocks * 2;
    }
    for (int b = 0; b < kblocks; b++) {
        __m256i acc[B8_NR_KQ];
        for (int c = 0; c < B8_NR_KQ; c++) acc[c] = _mm256_setzero_si256();
        for (int h = 0; h < 2; h++) {   /* chains of 4 maddubs stay under int16 (4*7874) */
            __m256i i16[B8_NR_KQ];
            for (int c = 0; c < B8_NR_KQ; c++) i16[c] = _mm256_setzero_si256();
            for (int g = h * 4; g < h * 4 + 4; g++) {
                __m256i w = _mm256_load_si256((const void*) (qs + g * 32));   /* shared across NR cols */
                for (int c = 0; c < B8_NR_KQ; c++)
                    i16[c] = b8_chain4(i16[c], w, _mm256_set1_epi32(jam_load32(x[c] + 4 * g)));
            }
            for (int c = 0; c < B8_NR_KQ; c++) acc[c] = b8_fold(acc[c], i16[c]);
        }
        __m256 dwv = _mm256_load_ps(dw), mwv = _mm256_load_ps(mw);
        for (int c = 0; c < B8_NR_KQ; c++) {
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[c]), _mm256_mul_ps(dwv, _mm256_set1_ps(d[c][b])), f[c]);
            f[c] = _mm256_fnmadd_ps(mwv, _mm256_set1_ps(sc[c][2 * b] + sc[c][2 * b + 1]), f[c]);
            x[c] += JAM_QK;
        }
        qs += 256; dw += 8; mw += 8;
    }
    for (int c = 0; c < B8_NR_KQ; c++)
        _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

void B8_NAME(q5k)(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int kblocks = J->kblocks, sblocks = J->dim1 / JAM_QKK, seq = J->seq;
    JAM_BAND8_DRIVER(kblocks * 256, kblocks * 8, B8_NR_KQ,    /* byte-expanded: 256 B/sub-block/group8 */
        repack_q5k_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, sblocks, qs, dw, mw),
        q5k_block8_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, kblocks, ldc, J->out, r),
        q5k_block8(qs, dw, mw, J->xq + (int64_t) s * kblocks * JAM_QK, J->dx + (int64_t) s * kblocks,
                   J->xsum + (int64_t) s * kblocks * 2, kblocks),
        jam_q5k_dot_f32(J->w + (int64_t) r * J->w_stride, sblocks, J->rhs + (int64_t) s * J->rhs_stride));
}

/* ================= Q6_K - 6-bit biased +32 (0..63 unsigned), per-16 scales, no min ===== */

static void repack_q6k_group8(const uint8_t* wbase, int64_t w_stride, int sblocks, uint8_t* qs, float* dw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += JAM_Q6K_BYTES) {
            const uint8_t* ql = w; const uint8_t* qh = w + 128;
            const int8_t* sc = (const int8_t*) (w + 192);
            float d = jam_half_at(w + 208);
            for (int h = 0; h < 2; h++) {
                const uint8_t* qlb = ql + h * 64; const uint8_t* qhb = qh + h * 32;
                for (int j = 0; j < 4; j++) {
                    int t0 = B * 16 + h * 8 + j * 2;
                    dw[t0 * 8 + r]       = d * sc[h * 8 + j * 2];
                    dw[(t0 + 1) * 8 + r] = d * sc[h * 8 + j * 2 + 1];
                    /* codes l = 0..31 of sub-blocks t0, t0+1 (16 each): dword l/4 lands at t0*128 + (l/4)*32 */
                    const __m256i l4 = _mm256_loadu_si256((const __m256i*) (qlb + (j & 1) * 32));
                    const __m256i nib = _mm256_and_si256(j < 2 ? l4 : _mm256_srli_epi16(l4, 4), _mm256_set1_epi8(0x0F));
                    const __m256i h = _mm256_loadu_si256((const __m256i*) qhb);
                    jam_put32_rows8(qs + (int64_t) t0 * 128 + r * 4, _mm256_or_si256(nib, b8_plane(h, 2 * j, 3)));
                }
            }
        }
    }
}

static inline __m256 q6k_block8(const uint8_t* qs, const float* dw,
                                const int8_t* x, const float* d, const float* s, int subs) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int g = 0; g < 4; g++)
            acc = b8_dot(acc, _mm256_load_si256((const void*) (qs + g * 32)),
                         _mm256_set1_epi32(jam_load32(x + 4 * g)));
        __m256 dwv = _mm256_load_ps(dw);
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(dwv, _mm256_set1_ps(d[b >> 1])), f);
        f = _mm256_fnmadd_ps(dwv, _mm256_set1_ps(32.0f * s[b]), f);          /* -32 offset, exact sum */
        qs += 128; dw += 8; x += 16;
    }
    return f;
}

static inline void q6k_block8_nr(const uint8_t* qs, const float* dw, const int8_t* xq,
                                 const float* dx, const float* xs, int s0, int subs,
                                 int kblocks, int64_t ldc, float* out, int r) {
    __m256 f[B8_NR_KQ];
    const int8_t* x[B8_NR_KQ]; const float* d[B8_NR_KQ]; const float* sc[B8_NR_KQ];
    for (int c = 0; c < B8_NR_KQ; c++) {
        f[c] = _mm256_setzero_ps();
        x[c]  = xq + (int64_t)(s0 + c) * kblocks * JAM_QK;
        d[c]  = dx + (int64_t)(s0 + c) * kblocks;
        sc[c] = xs + (int64_t)(s0 + c) * kblocks * 2;
    }
    for (int b = 0; b < subs; b++) {
        __m256i acc[B8_NR_KQ];
        for (int c = 0; c < B8_NR_KQ; c++) acc[c] = _mm256_setzero_si256();
        for (int g = 0; g < 4; g++) {
            __m256i w = _mm256_load_si256((const void*) (qs + g * 32));   /* shared across NR cols */
            for (int c = 0; c < B8_NR_KQ; c++)
                acc[c] = b8_dot(acc[c], w, _mm256_set1_epi32(jam_load32(x[c] + 4 * g)));
        }
        __m256 dwv = _mm256_load_ps(dw);
        for (int c = 0; c < B8_NR_KQ; c++) {
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[c]), _mm256_mul_ps(dwv, _mm256_set1_ps(d[c][b >> 1])), f[c]);
            f[c] = _mm256_fnmadd_ps(dwv, _mm256_set1_ps(32.0f * sc[c][b]), f[c]);
            x[c] += 16;
        }
        qs += 128; dw += 8;
    }
    for (int c = 0; c < B8_NR_KQ; c++)
        _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

void B8_NAME(q6k)(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int kblocks = J->kblocks, sblocks = J->dim1 / JAM_QKK, subs16 = J->dim1 / 16, seq = J->seq;
    JAM_BAND8_DRIVER(kblocks * 256, subs16 * 8, B8_NR_KQ,     /* 2 sub16 x 128 B per 32-block */
        repack_q6k_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, sblocks, qs, dw),
        q6k_block8_nr(qs, dw, J->xq, J->dx, J->xsum, s, subs16, kblocks, ldc, J->out, r),
        q6k_block8(qs, dw, J->xq + (int64_t) s * kblocks * JAM_QK, J->dx + (int64_t) s * kblocks,
                   J->xsum + (int64_t) s * kblocks * 2, subs16),
        jam_q6k_dot_f32(J->w + (int64_t) r * J->w_stride, sblocks, J->rhs + (int64_t) s * J->rhs_stride));
}

/* ================= 32-block quants (Q8_0 / Q4_0 / MXFP4) - 8-row bands ==================
 * Same band machinery as the K-quants above; per-quant dot schemes chosen by saturation math:
 *   Q8_0 : signed weights, so maddubs needs the SIGN TRICK - u = |w|, s = sign(x, w): |w| is the u8
 *          operand because a weight may be -128 (|-128| reads as u8 128, exact) while sign() cannot
 *          negate -128; the activation is requantized to +-127, so sign(x, w) is always exact
 *          (int16 peak 2*128*127 = 32512 < 32767) -> exact w*x, no bias correction.
 *   Q4_0 : unsigned nibbles 0..15 as the u operand, x signed as s (peak 15*127*2) - the -8
 *          offset folds into mw = 8*d against the RAW per-32 activation sums (exact).
 *   MXFP4: signed codes |w| <= 12, so the a+128 scheme is safe (peak 255*12*2 = 6120):
 *          u = x^0x80 broadcast, s = w; corrected per row/block via cw = d*128*sum(w). */

#ifndef JAM_BAND8_VNNI   /* AVX-VNNI has its own Q8_0, Q5_0 and Q4_0 bands */
/* ---- Q8_0: repack 8 rows (raw signed bytes, 256 B/block/group8) + per-row scale ---- */
static void repack_q8s_group8(const uint8_t* wbase, int64_t w_stride, int nb, uint8_t* qs, float* dw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q8_0_BYTES) {
            dw[(int64_t) B * 8 + r] = jam_half_at(w);
            jam_put32_rows8(qs + (int64_t) B * 256 + r * 4, _mm256_loadu_si256((const __m256i*) (w + 2)));
        }
    }
}

static inline __m256 q8s_block8(const uint8_t* qs, const float* dw, const int8_t* x, const float* d, int nb) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {
            __m256i xb = _mm256_set1_epi32(jam_load32(x + 4 * g));
            __m256i w = _mm256_load_si256((const void*) (qs + g * 32));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(
                      _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(xb, w)),
                      _mm256_set1_epi16(1)));
        }
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(_mm256_load_ps(dw), _mm256_set1_ps(d[b])), f);
        qs += 256; dw += 8; x += JAM_QK;
    }
    return f;
}

static inline void q8s_block8_nr(const uint8_t* qs, const float* dw, const int8_t* xq,
                                 const float* dx, int s0, int nb, int64_t ldc, float* out, int r) {
    __m256 f[B8_NR_Q8];
    const int8_t* x[B8_NR_Q8]; const float* d[B8_NR_Q8];
    for (int c = 0; c < B8_NR_Q8; c++) {
        f[c] = _mm256_setzero_ps();
        x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK;
        d[c] = dx + (int64_t)(s0 + c) * nb;
    }
    for (int b = 0; b < nb; b++) {
        __m256i acc[B8_NR_Q8];
        for (int c = 0; c < B8_NR_Q8; c++) acc[c] = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {
            __m256i w = _mm256_load_si256((const void*) (qs + g * 32));   /* shared across NR cols */
            for (int c = 0; c < B8_NR_Q8; c++) {
                __m256i xb = _mm256_set1_epi32(jam_load32(x[c] + 4 * g));
                acc[c] = _mm256_add_epi32(acc[c], _mm256_madd_epi16(
                             _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(xb, w)),
                             _mm256_set1_epi16(1)));
            }
        }
        __m256 dwv = _mm256_load_ps(dw);
        for (int c = 0; c < B8_NR_Q8; c++) {
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[c]), _mm256_mul_ps(dwv, _mm256_set1_ps(d[c][b])), f[c]);
            x[c] += JAM_QK;
        }
        qs += 256; dw += 8;
    }
    for (int c = 0; c < B8_NR_Q8; c++)
        _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

void jam_q8_0_band8_avx2(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    JAM_BAND8_DRIVER(nb * 256, nb * 8, B8_NR_Q8,
        repack_q8s_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw),
        q8s_block8_nr(qs, dw, J->xq, J->dx, s, nb, ldc, J->out, r),
        q8s_block8(qs, dw, J->xq + (int64_t) s * nb * JAM_QK, J->dx + (int64_t) s * nb, nb),
        jam_q8_0_dot_f32(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride));
}

/* Q5_0: the Q8_0 sign-trick band over Q5_0 blocks (q-16 is an int8; only the block unpack differs). */
static void repack_q5_0s_group8(const uint8_t* wbase, int64_t w_stride, int nb, uint8_t* qs, float* dw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q5_0_BYTES) {
            dw[(int64_t) B * 8 + r] = jam_half_at(w);
            jam_put32_rows8(qs + (int64_t) B * 256 + r * 4, jam_q5_0_codes_256(w));
        }
    }
}

void jam_q5_0_band8_avx2(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    JAM_BAND8_DRIVER(nb * 256, nb * 8, B8_NR_Q8,
        repack_q5_0s_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw),
        q8s_block8_nr(qs, dw, J->xq, J->dx, s, nb, ldc, J->out, r),
        q8s_block8(qs, dw, J->xq + (int64_t) s * nb * JAM_QK, J->dx + (int64_t) s * nb, nb),
        jam_q5_0_dot_f32(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride));
}

/* ---- Q4_0: unsigned nibbles (0..15), -8 offset via mw = 8*d against raw per-32 x sums ---- */
static void repack_q4_0_group8(const uint8_t* wbase, int64_t w_stride, int nb,
                               uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q4_0_BYTES) {
            float d = jam_half_at(w);
            dw[(int64_t) B * 8 + r] = d;
            mw[(int64_t) B * 8 + r] = 8.0f * d;
            jam_put32_rows8(qs + (int64_t) B * 256 + r * 4, jam_nibbles_256(w + 2));   /* 16 packed bytes = 32 nibbles */
        }
    }
}

static inline __m256 q4_0_block8(const uint8_t* qs, const float* dw, const float* mw,
                                 const int8_t* x, const float* d, const float* s, int nb) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i i16 = _mm256_setzero_si256();
        B8_UNROLL2 for (int g = 0; g < 8; g++)   /* 8 chained maddubs stay under int16 (8*3810) */
            i16 = b8_chain(i16, _mm256_load_si256((const void*) (qs + g * 32)),
                           _mm256_set1_epi32(jam_load32(x + 4 * g)));
        __m256i acc = b8_fold(_mm256_setzero_si256(), i16);
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(_mm256_load_ps(dw), _mm256_set1_ps(d[b])), f);
        f = _mm256_fnmadd_ps(_mm256_load_ps(mw), _mm256_set1_ps(s[2 * b] + s[2 * b + 1]), f);
        qs += 256; dw += 8; mw += 8; x += JAM_QK;
    }
    return f;
}

static inline void q4_0_block8_nr(const uint8_t* qs, const float* dw, const float* mw, const int8_t* xq,
                                  const float* dx, const float* xs, int s0, int nb, int64_t ldc,
                                  float* out, int r) {
    __m256 f[B8_NR_KQ];
    const int8_t* x[B8_NR_KQ]; const float* d[B8_NR_KQ]; const float* sc[B8_NR_KQ];
    for (int c = 0; c < B8_NR_KQ; c++) {
        f[c] = _mm256_setzero_ps();
        x[c]  = xq + (int64_t)(s0 + c) * nb * JAM_QK;
        d[c]  = dx + (int64_t)(s0 + c) * nb;
        sc[c] = xs + (int64_t)(s0 + c) * nb * 2;
    }
    for (int b = 0; b < nb; b++) {
        __m256i i16[B8_NR_KQ];
        for (int c = 0; c < B8_NR_KQ; c++) i16[c] = _mm256_setzero_si256();
        B8_UNROLL2 for (int g = 0; g < 8; g++) {   /* 8 chained maddubs stay under int16 (8*3810) */
            __m256i w = _mm256_load_si256((const void*) (qs + g * 32));
            for (int c = 0; c < B8_NR_KQ; c++)
                i16[c] = b8_chain(i16[c], w, _mm256_set1_epi32(jam_load32(x[c] + 4 * g)));
        }
        __m256 dwv = _mm256_load_ps(dw), mwv = _mm256_load_ps(mw);
        for (int c = 0; c < B8_NR_KQ; c++) {
            __m256i acc = b8_fold(_mm256_setzero_si256(), i16[c]);
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(dwv, _mm256_set1_ps(d[c][b])), f[c]);
            f[c] = _mm256_fnmadd_ps(mwv, _mm256_set1_ps(sc[c][2 * b] + sc[c][2 * b + 1]), f[c]);
            x[c] += JAM_QK;
        }
        qs += 256; dw += 8; mw += 8;
    }
    for (int c = 0; c < B8_NR_KQ; c++)
        _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

void jam_q4_0_band8_avx2(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    JAM_BAND8_DRIVER(nb * 256, nb * 8, B8_NR_KQ,
        repack_q4_0_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, mw),
        q4_0_block8_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, nb, ldc, J->out, r),
        q4_0_block8(qs, dw, mw, J->xq + (int64_t) s * nb * JAM_QK, J->dx + (int64_t) s * nb,
                    J->xsum + (int64_t) s * nb * 2, nb),
        jam_q4_0_dot_f32(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride));
}

#endif

/* ---- MXFP4: LUT-decoded signed codes |w|<=12; a+128 scheme, cw = d*128*sum(w) ---- */
static void repack_mxfp4_group8(const uint8_t* wbase, int64_t w_stride, int nb,
                                uint8_t* qs, float* dw, float* cw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_MXFP4_BYTES) {
            float d = jam_mxfp4_dhalf(w[0]);
            const __m256i lut = _mm256_setr_epi8(JAM_MXFP4_CODES, JAM_MXFP4_CODES);
            const __m256i v = _mm256_shuffle_epi8(lut, jam_nibbles_256(w + 1));   /* 16 packed bytes = 32 codes */
            jam_put32_rows8(qs + (int64_t) B * 256 + r * 4, v);
            const int sumw = jam_sum_i8_256(v);
            dw[(int64_t) B * 8 + r] = d;
            cw[(int64_t) B * 8 + r] = d * 128.0f * (float) sumw;
        }
    }
}

static inline __m256 mxfp4_block8(const uint8_t* qs, const float* dw, const float* cw,
                                  const int8_t* x, const float* d, int nb) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int h = 0; h < 2; h++) {   /* chains of 4 maddubs stay under int16 (4*6120) */
            __m256i i16 = _mm256_setzero_si256();
            for (int g = h * 4; g < h * 4 + 4; g++)
                i16 = b8_chain4(i16, jam_bcast4_biased(x + 4 * g),
                               _mm256_load_si256((const void*) (qs + g * 32)));
            acc = b8_fold(acc, i16);
        }
        __m256 da = _mm256_set1_ps(d[b]);
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(_mm256_load_ps(dw), da), f);
        f = _mm256_fnmadd_ps(_mm256_load_ps(cw), da, f);
        qs += 256; dw += 8; cw += 8; x += JAM_QK;
    }
    return f;
}

static inline void mxfp4_block8_nr(const uint8_t* qs, const float* dw, const float* cw, const int8_t* xq,
                                   const float* dx, int s0, int nb, int64_t ldc, float* out, int r) {
    __m256 f[B8_NR_KQ];
    const int8_t* x[B8_NR_KQ]; const float* d[B8_NR_KQ];
    for (int c = 0; c < B8_NR_KQ; c++) {
        f[c] = _mm256_setzero_ps();
        x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK;
        d[c] = dx + (int64_t)(s0 + c) * nb;
    }
    for (int b = 0; b < nb; b++) {
        __m256i acc[B8_NR_KQ];
        for (int c = 0; c < B8_NR_KQ; c++) acc[c] = _mm256_setzero_si256();
        for (int h = 0; h < 2; h++) {   /* chains of 4 maddubs stay under int16 (4*6120) */
            __m256i i16[B8_NR_KQ];
            for (int c = 0; c < B8_NR_KQ; c++) i16[c] = _mm256_setzero_si256();
            for (int g = h * 4; g < h * 4 + 4; g++) {
                __m256i w = _mm256_load_si256((const void*) (qs + g * 32));
                for (int c = 0; c < B8_NR_KQ; c++)
                    i16[c] = b8_chain4(i16[c], jam_bcast4_biased(x[c] + 4 * g), w);
            }
            for (int c = 0; c < B8_NR_KQ; c++) acc[c] = b8_fold(acc[c], i16[c]);
        }
        __m256 dwv = _mm256_load_ps(dw), cwv = _mm256_load_ps(cw);
        for (int c = 0; c < B8_NR_KQ; c++) {
            __m256 da = _mm256_set1_ps(d[c][b]);
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[c]), _mm256_mul_ps(dwv, da), f[c]);
            f[c] = _mm256_fnmadd_ps(cwv, da, f[c]);
            x[c] += JAM_QK;
        }
        qs += 256; dw += 8; cw += 8;
    }
    for (int c = 0; c < B8_NR_KQ; c++)
        _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

void B8_NAME(mxfp4)(void* arg, int t0, int t1, int tid) {
    const jam_band_job* J = (const jam_band_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    JAM_BAND8_DRIVER(nb * 256, nb * 8, B8_NR_KQ,              /* mw holds the corrections cw */
        repack_mxfp4_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, mw),
        mxfp4_block8_nr(qs, dw, mw, J->xq, J->dx, s, nb, ldc, J->out, r),
        mxfp4_block8(qs, dw, mw, J->xq + (int64_t) s * nb * JAM_QK, J->dx + (int64_t) s * nb, nb),
        jam_mxfp4_dot_f32(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride));
}
