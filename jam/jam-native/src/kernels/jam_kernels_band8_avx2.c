/* AVX2 8-row K-quant repack bands — the jam_q4k/q5k/q6k avx512 bands ported to ymm + maddubs.
 * This TU is compiled with -mavx2 -mfma -mf16c only (part of the jam_avx2 target): it is the
 * K-quant prefill fast path for CPUs BELOW avx512-vnni (Haswell..Rocket Lake, Zen 2/3, and
 * Alder-Lake-class clients for whom no K-quant VNNI band exists).
 *
 * Same bounded per-worker scratch as the VNNI bands (jam_repack, sized by ensure_kquant: the
 * 4-groups-of-8 layouts are byte-for-byte the 2-groups-of-16 ones), same two-phase launch, same
 * deferred-float scale/min math. Only the int8 dot differs: no vpdpbusd, so the base ladder is
 * maddubs(w_u8, x_s8) -> madd(1) -> add int32 — and wherever the per-maddubs int16 bound allows,
 * products ACCUMULATE IN INT16 (plain add_epi16, wrap provably unreachable) with ONE deferred
 * madd, cutting the ladder to ~2 ops/group:
 *   Q4_K, Q4_0 : peak 2*15*127 = 3810/maddubs -> all 8 groups chain (8*3810 = 30480 < 32767)
 *   MXFP4      : peak 2*255*12 = 6120        -> chains of 4 (24480)
 *   Q5_K       : peak 2*31*127 = 7874        -> chains of 4 (31496)
 *   Q6_K       : peak 2*63*127 = 16002       -> NO chaining (2 already wrap); per-group madd
 *   Q8_0       : sign trick peaks at 32258   -> NO chaining; per-group madd
 * Weight operands are UNSIGNED for every quant except Q8_0 (sign trick: u = |x|, s = sign(w,x)). */
#include "jam_internal.h"
#include "jam_kquant.h"
#include "jam_mxfp4.h"
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

/* Activation columns per register tile, per quant - measured optima (m=4096 n=512 k=2048,
 * JAM_ISA=avx2): Q4_K peaks at 4 (dual lo/hi accumulators fill the file), the single-accumulator
 * chained kernels at 6, and Q8_0's sign-trick pipeline at 8 (the extra independent columns hide
 * the abs/sign latency). */
#define B8_NR_Q4K 4
#define B8_NR_KQ  6
#define B8_NR_Q8  8

static inline float b8_h2f(uint16_t h) { return _cvtsh_ss(h); }

/* one 32-byte weight group (8 rows x 4 elems) dotted against a 4-elem activation broadcast */
static inline __m256i b8_dot(__m256i acc, __m256i wu, __m256i xb) {
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wu, xb), _mm256_set1_epi16(1)));
}

/* int16-chained variants: accumulate maddubs products with a plain (wrapping) add_epi16 - callers
 * guarantee the chain length keeps |sum| under 32767 (see the header table) - then fold the chain
 * to int32 with one madd. */
static inline __m256i b8_chain(__m256i i16, __m256i wu, __m256i xb) {
    return _mm256_add_epi16(i16, _mm256_maddubs_epi16(wu, xb));
}

static inline __m256i b8_fold(__m256i acc, __m256i i16) {
    return _mm256_add_epi32(acc, _mm256_madd_epi16(i16, _mm256_set1_epi16(1)));
}

/* ---- phase 1: F32 activations -> int8 + per-32 scale + per-16 RAW float sums ----
 * Semantics identical to the avx512 quantize_row_q8s (raw sums feed the exact dmin*min /
 * -32-offset corrections); 256-bit only. */
static inline float b8_hmax8(__m256 v) {
    __m128 h = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_max_ps(h, _mm_movehl_ps(h, h));
    h = _mm_max_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}

static inline float b8_hsum8(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}

static void quantize_row_q8s_avx2(const float* x, int kblocks, int8_t* xq, float* dx, float* xs) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int b = 0; b < kblocks; b++, x += JAM_QK, xq += JAM_QK, xs += 2) {
        __m256 a0 = _mm256_loadu_ps(x),      a1 = _mm256_loadu_ps(x + 8);
        __m256 a2 = _mm256_loadu_ps(x + 16), a3 = _mm256_loadu_ps(x + 24);
        __m256 mx = _mm256_max_ps(_mm256_max_ps(_mm256_and_ps(a0, absmask), _mm256_and_ps(a1, absmask)),
                                  _mm256_max_ps(_mm256_and_ps(a2, absmask), _mm256_and_ps(a3, absmask)));
        float max = b8_hmax8(mx);
        float d = max / 127.0f, inv = max > 0.0f ? 127.0f / max : 0.0f;
        dx[b] = d;
        xs[0] = b8_hsum8(_mm256_add_ps(a0, a1));   /* raw per-16 sums (exact min term) */
        xs[1] = b8_hsum8(_mm256_add_ps(a2, a3));
        __m256 v = _mm256_set1_ps(inv);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(a0, v)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(a1, v));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(a2, v)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(a3, v));
        /* packs are 128-lane-local; the {0,4,1,5,2,6,3,7} permute restores element order */
        __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        _mm256_storeu_si256((__m256i*) xq, _mm256_permutevar8x32_epi32(p, perm));
    }
}

void jam_q4k_quant_avx2(void* arg, int s0, int s1, int tid) {
    (void) tid;
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    for (int s = s0; s < s1; s++)
        quantize_row_q8s_avx2(J->rhs + (size_t) s * J->rhs_stride, J->kblocks,
                              J->xq + (size_t) s * J->kblocks * JAM_QK,
                              J->dx + (size_t) s * J->kblocks,
                              J->xsum + (size_t) s * J->kblocks * 2);
}

/* ================= Q4_K ============================================================== */

/* repack 8 Q4_K rows: packed dual-nibble bytes (lo = sbLo, hi = sbHi), 256 B/sub-block-pair
 * per group of 8; per-row sub-block scales dw / mins mw at [sb*8 + r]. */
static void repack_q4k_group8(const uint8_t* wbase, int64_t w_stride, int sblocks,
                              uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += JAM_Q4K_BYTES) {
            float d = b8_h2f(*(const uint16_t*) w), dmin = b8_h2f(*(const uint16_t*) (w + 2));
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
                    *(uint32_t*) (dst + k * 32) = *(const uint32_t*) (qg + k * 4);
            }
        }
    }
}

static float q4k_dot_scalar(const uint8_t* w, const float* x, int sblocks) {   /* <8-row tail: exact */
    float acc = 0.0f;
    for (int B = 0; B < sblocks; B++, w += JAM_Q4K_BYTES, x += JAM_QKK) {
        float d = b8_h2f(*(const uint16_t*) w), dmin = b8_h2f(*(const uint16_t*) (w + 2));
        uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
        const uint8_t* q = w + 16;
        for (int g = 0; g < 4; g++) {
            float dl = d*sc[g*2], ml = dmin*mn[g*2], dh = d*sc[g*2+1], mh = dmin*mn[g*2+1];
            for (int i = 0; i < 32; i++) {
                acc += (dl * (q[g*32+i] & 0xF) - ml) * x[g*64+i];
                acc += (dh * (q[g*32+i] >> 4) - mh) * x[g*64+32+i];
            }
        }
    }
    return acc;
}

/* single activation column: 8-row partials for the seq tail */
static inline __m256 q4k_block8(const uint8_t* qs, const float* dw, const float* mw,
                                const int8_t* x, const float* d, const float* s, int pairs) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 f = _mm256_setzero_ps();
    for (int p = 0; p < pairs; p++) {
        __m256i iLo = _mm256_setzero_si256(), iHi = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {   /* 8 chained maddubs stay under int16 (8*3810) */
            __m256i pk = _mm256_load_si256((const void*) (qs + g * 32));
            iLo = b8_chain(iLo, _mm256_and_si256(pk, m4), _mm256_set1_epi32(((const int*) x)[g]));
            iHi = b8_chain(iHi, _mm256_and_si256(_mm256_srli_epi16(pk, 4), m4),
                           _mm256_set1_epi32(((const int*) (x + JAM_QK))[g]));
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
        for (int g = 0; g < 8; g++) {   /* 8 chained maddubs stay under int16 (8*3810) */
            __m256i pk = _mm256_load_si256((const void*) (qs + g * 32));
            __m256i lo = _mm256_and_si256(pk, m4);                       /* decode once, reuse NR cols */
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(pk, 4), m4);
            for (int c = 0; c < B8_NR_Q4K; c++) {
                iLo[c] = b8_chain(iLo[c], lo, _mm256_set1_epi32(((const int*) x[c])[g]));
                iHi[c] = b8_chain(iHi[c], hi, _mm256_set1_epi32(((const int*) (x[c] + JAM_QK))[g]));
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

void jam_q4k_band8_avx2(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int kblocks = J->kblocks, sblocks = J->dim1 / JAM_QKK, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 7 < row_end; r += 8, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * kblocks * 128;   /* packed: 128 B/32-block/group8 */
            float* dw = rp->dw + (int64_t) group * kblocks * 8;
            float* mw = rp->mw + (int64_t) group * kblocks * 8;
            repack_q4k_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, sblocks, qs, dw, mw);
            int s = 0;
            for (; s + B8_NR_Q4K <= seq; s += B8_NR_Q4K)
                q4k_block8_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, kblocks / 2, kblocks, ldc, J->out, r);
            for (; s < seq; s++) {
                const int8_t* x = J->xq + (int64_t) s * kblocks * JAM_QK;
                const float* d = J->dx + (int64_t) s * kblocks;
                const float* sv = J->xsum + (int64_t) s * kblocks * 2;
                _mm256_storeu_ps(J->out + (int64_t) s * ldc + r, q4k_block8(qs, dw, mw, x, d, sv, kblocks / 2));
            }
        }
        for (int r = row + group * 8; r < row_end; r++)          /* <8-row tail: scalar */
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q4k_dot_scalar(J->w + (int64_t) r * J->w_stride, J->rhs + (int64_t) s * J->rhs_stride, sblocks);
    }
}

/* ================= Q5_K — Q4_K scheme, 5-bit byte-expanded (256 B/sub-block/group8) ==== */

static void repack_q5k_group8(const uint8_t* wbase, int64_t w_stride, int sblocks,
                              uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += JAM_Q5K_BYTES) {
            float d = b8_h2f(*(const uint16_t*) w), dmin = b8_h2f(*(const uint16_t*) (w + 2));
            uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
            const uint8_t* qh = w + 16; const uint8_t* q5 = w + 48;
            for (int g = 0; g < 4; g++) {
                int sbLo = B * 8 + g * 2, sbHi = sbLo + 1;
                dw[sbLo * 8 + r] = d * sc[g * 2];     mw[sbLo * 8 + r] = dmin * mn[g * 2];
                dw[sbHi * 8 + r] = d * sc[g * 2 + 1]; mw[sbHi * 8 + r] = dmin * mn[g * 2 + 1];
                const uint8_t* q = q5 + g * 32;
                for (int i = 0; i < 32; i++) {
                    uint8_t lo = (q[i] & 0xF) | (((qh[i] >> (2 * g))     & 1) << 4);   /* 0..31 */
                    uint8_t hi = (q[i] >> 4)  | (((qh[i] >> (2 * g + 1)) & 1) << 4);
                    qs[(int64_t) sbLo * 256 + (i >> 2) * 32 + r * 4 + (i & 3)] = lo;
                    qs[(int64_t) sbHi * 256 + (i >> 2) * 32 + r * 4 + (i & 3)] = hi;
                }
            }
        }
    }
}

static float q5k_dot_scalar(const uint8_t* w, const float* x, int sblocks) {
    float acc = 0.0f;
    for (int B = 0; B < sblocks; B++, w += JAM_Q5K_BYTES, x += JAM_QKK) {
        float d = b8_h2f(*(const uint16_t*) w), dmin = b8_h2f(*(const uint16_t*) (w + 2));
        uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
        const uint8_t* qh = w + 16; const uint8_t* q5 = w + 48;
        for (int g = 0; g < 4; g++) {
            float dl=d*sc[g*2], ml=dmin*mn[g*2], dh=d*sc[g*2+1], mh=dmin*mn[g*2+1];
            const uint8_t* q = q5 + g*32; const float* xlo = x + g*64; const float* xhi = xlo + 32;
            for (int i = 0; i < 32; i++) {
                int qlo = (q[i] & 0xF) | (((qh[i] >> (2*g))   & 1) << 4);
                int qhi = (q[i] >> 4)  | (((qh[i] >> (2*g+1)) & 1) << 4);
                acc += (dl * qlo - ml) * xlo[i];
                acc += (dh * qhi - mh) * xhi[i];
            }
        }
    }
    return acc;
}

static inline __m256 q5k_block8(const uint8_t* qs, const float* dw, const float* mw,
                                const int8_t* x, const float* d, const float* s, int subs) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int h = 0; h < 2; h++) {   /* chains of 4 maddubs stay under int16 (4*7874) */
            __m256i i16 = _mm256_setzero_si256();
            for (int g = h * 4; g < h * 4 + 4; g++)
                i16 = b8_chain(i16, _mm256_load_si256((const void*) (qs + g * 32)),
                               _mm256_set1_epi32(((const int*) x)[g]));
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
                    i16[c] = b8_chain(i16[c], w, _mm256_set1_epi32(((const int*) x[c])[g]));
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

void jam_q5k_band8_avx2(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int kblocks = J->kblocks, sblocks = J->dim1 / JAM_QKK, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 7 < row_end; r += 8, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * kblocks * 256;   /* byte-expanded: 256 B/sub-block/group8 */
            float* dw = rp->dw + (int64_t) group * kblocks * 8;
            float* mw = rp->mw + (int64_t) group * kblocks * 8;
            repack_q5k_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, sblocks, qs, dw, mw);
            int s = 0;
            for (; s + B8_NR_KQ <= seq; s += B8_NR_KQ)
                q5k_block8_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, kblocks, ldc, J->out, r);
            for (; s < seq; s++)
                _mm256_storeu_ps(J->out + (int64_t) s * ldc + r,
                                 q5k_block8(qs, dw, mw, J->xq + (int64_t) s * kblocks * JAM_QK,
                                            J->dx + (int64_t) s * kblocks,
                                            J->xsum + (int64_t) s * kblocks * 2, kblocks));
        }
        for (int r = row + group * 8; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q5k_dot_scalar(J->w + (int64_t) r * J->w_stride, J->rhs + (int64_t) s * J->rhs_stride, sblocks);
    }
}

/* ================= Q6_K — 6-bit biased +32 (0..63 unsigned), per-16 scales, no min ===== */

static void repack_q6k_group8(const uint8_t* wbase, int64_t w_stride, int sblocks, uint8_t* qs, float* dw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += JAM_Q6K_BYTES) {
            const uint8_t* ql = w; const uint8_t* qh = w + 128;
            const int8_t* sc = (const int8_t*) (w + 192);
            float d = b8_h2f(*(const uint16_t*) (w + 208));
            for (int h = 0; h < 2; h++) {
                const uint8_t* qlb = ql + h * 64; const uint8_t* qhb = qh + h * 32;
                for (int j = 0; j < 4; j++) {
                    int t0 = B * 16 + h * 8 + j * 2;
                    dw[t0 * 8 + r]       = d * sc[h * 8 + j * 2];
                    dw[(t0 + 1) * 8 + r] = d * sc[h * 8 + j * 2 + 1];
                    for (int l = 0; l < 32; l++) {
                        int qv;
                        switch (j) { case 0: qv = qlb[l] & 0xF; break; case 1: qv = qlb[32+l] & 0xF; break;
                                     case 2: qv = qlb[l] >> 4; break; default: qv = qlb[32+l] >> 4; break; }
                        qv |= ((qhb[l] >> (2 * j)) & 3) << 4;
                        int t = t0 + l / 16, e = l % 16;
                        qs[(int64_t) t * 128 + (e / 4) * 32 + r * 4 + (e % 4)] = (uint8_t) qv;
                    }
                }
            }
        }
    }
}

static float q6k_dot_scalar(const uint8_t* w, const float* x, int sblocks) {
    float acc = 0.0f;
    for (int B = 0; B < sblocks; B++, w += JAM_Q6K_BYTES, x += JAM_QKK) {
        const uint8_t* ql = w; const uint8_t* qh = w + 128;
        const int8_t* sc = (const int8_t*) (w + 192);
        float d = b8_h2f(*(const uint16_t*) (w + 208));
        for (int h = 0; h < 2; h++) {
            const uint8_t* qlb = ql + h * 64; const uint8_t* qhb = qh + h * 32;
            for (int j = 0; j < 4; j++)
                for (int l = 0; l < 32; l++) {
                    int qv;
                    switch (j) { case 0: qv = qlb[l] & 0xF; break; case 1: qv = qlb[32+l] & 0xF; break;
                                 case 2: qv = qlb[l] >> 4; break; default: qv = qlb[32+l] >> 4; break; }
                    qv |= ((qhb[l] >> (2 * j)) & 3) << 4;
                    acc += d * sc[h*8 + j*2 + l/16] * (qv - 32) * x[h*128 + j*32 + l];
                }
        }
    }
    return acc;
}

static inline __m256 q6k_block8(const uint8_t* qs, const float* dw,
                                const int8_t* x, const float* d, const float* s, int subs) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int g = 0; g < 4; g++)
            acc = b8_dot(acc, _mm256_load_si256((const void*) (qs + g * 32)),
                         _mm256_set1_epi32(((const int*) x)[g]));
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
                acc[c] = b8_dot(acc[c], w, _mm256_set1_epi32(((const int*) x[c])[g]));
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

void jam_q6k_band8_avx2(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int kblocks = J->kblocks, sblocks = J->dim1 / JAM_QKK, subs16 = J->dim1 / 16, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 7 < row_end; r += 8, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * kblocks * 256;   /* 2 sub16 x 128 B per 32-block */
            float* dw = rp->dw + (int64_t) group * subs16 * 8;
            repack_q6k_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, sblocks, qs, dw);
            int s = 0;
            for (; s + B8_NR_KQ <= seq; s += B8_NR_KQ)
                q6k_block8_nr(qs, dw, J->xq, J->dx, J->xsum, s, subs16, kblocks, ldc, J->out, r);
            for (; s < seq; s++) {
                const int8_t* x = J->xq + (int64_t) s * kblocks * JAM_QK;
                const float* d = J->dx + (int64_t) s * kblocks;
                const float* sv = J->xsum + (int64_t) s * kblocks * 2;
                _mm256_storeu_ps(J->out + (int64_t) s * ldc + r, q6k_block8(qs, dw, x, d, sv, subs16));
            }
        }
        for (int r = row + group * 8; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q6k_dot_scalar(J->w + (int64_t) r * J->w_stride, J->rhs + (int64_t) s * J->rhs_stride, sblocks);
    }
}

/* ================= 32-block quants (Q8_0 / Q4_0 / MXFP4) — 8-row bands ==================
 * Same band machinery as the K-quants above; per-quant dot schemes chosen by saturation math:
 *   Q8_0 : signed weights, so maddubs needs the SIGN TRICK - u = |x| broadcast, s = sign(w, x)
 *          (int16 peak 127*127*2 = 32258, the exact margin) -> exact w*x, no bias correction.
 *   Q4_0 : unsigned nibbles 0..15 as the u operand, x signed as s (peak 15*127*2) - the -8
 *          offset folds into mw = 8*d against the RAW per-32 activation sums (exact).
 *   MXFP4: signed codes |w| <= 12, so the a+128 scheme is safe (peak 255*12*2 = 6120):
 *          u = x^0x80 broadcast, s = w; corrected per row/block via cw = d*128*sum(w). */

#define JAM_MXFP4_BYTES 17   /* 1 shared E8M0 exponent + 16 packed FP4 bytes */

/* ---- Q8_0: repack 8 rows (raw signed bytes, 256 B/block/group8) + per-row scale ---- */
static void repack_q8s_group8(const uint8_t* wbase, int64_t w_stride, int nb, uint8_t* qs, float* dw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q8_0_BYTES) {
            dw[(int64_t) B * 8 + r] = b8_h2f(*(const uint16_t*) w);
            const int8_t* q = (const int8_t*) (w + 2);
            for (int g = 0; g < 8; g++)
                *(uint32_t*) (qs + (int64_t) B * 256 + g * 32 + r * 4) = *(const uint32_t*) (q + g * 4);
        }
    }
}

static inline __m256 q8s_block8(const uint8_t* qs, const float* dw, const int8_t* x, const float* d, int nb) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {
            __m256i xb = _mm256_set1_epi32(((const int*) x)[g]);
            __m256i w = _mm256_load_si256((const void*) (qs + g * 32));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(
                      _mm256_maddubs_epi16(_mm256_abs_epi8(xb), _mm256_sign_epi8(w, xb)),
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
                __m256i xb = _mm256_set1_epi32(((const int*) x[c])[g]);
                acc[c] = _mm256_add_epi32(acc[c], _mm256_madd_epi16(
                             _mm256_maddubs_epi16(_mm256_abs_epi8(xb), _mm256_sign_epi8(w, xb)),
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

static float q8s_dot_scalar(const uint8_t* w, int nb, const float* x) {   /* <8-row tail: exact */
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += JAM_Q8_0_BYTES, x += JAM_QK) {
        float d = b8_h2f(*(const uint16_t*) w);
        const int8_t* q = (const int8_t*) (w + 2);
        float s = 0.0f;
        for (int e = 0; e < 32; e++) s += (float) q[e] * x[e];
        acc += d * s;
    }
    return acc;
}

void jam_q8_0_band8_avx2(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 7 < row_end; r += 8, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * nb * 256;
            float* dw = rp->dw + (int64_t) group * nb * 8;
            repack_q8s_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw);
            int s = 0;
            for (; s + B8_NR_Q8 <= seq; s += B8_NR_Q8)
                q8s_block8_nr(qs, dw, J->xq, J->dx, s, nb, ldc, J->out, r);
            for (; s < seq; s++)
                _mm256_storeu_ps(J->out + (int64_t) s * ldc + r,
                                 q8s_block8(qs, dw, J->xq + (int64_t) s * nb * JAM_QK,
                                            J->dx + (int64_t) s * nb, nb));
        }
        for (int r = row + group * 8; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q8s_dot_scalar(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride);
    }
}

/* ---- Q4_0: unsigned nibbles (0..15), -8 offset via mw = 8*d against raw per-32 x sums ---- */
static void repack_q4_0_group8(const uint8_t* wbase, int64_t w_stride, int nb,
                               uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q4_0_BYTES) {
            float d = b8_h2f(*(const uint16_t*) w);
            dw[(int64_t) B * 8 + r] = d;
            mw[(int64_t) B * 8 + r] = 8.0f * d;
            const uint8_t* q = w + 2;                       /* 16 packed bytes = 32 nibbles */
            for (int e = 0; e < 32; e++) {
                uint8_t v = e < 16 ? (q[e] & 0xF) : (q[e - 16] >> 4);
                qs[(int64_t) B * 256 + (e / 4) * 32 + r * 4 + (e % 4)] = v;
            }
        }
    }
}

static inline __m256 q4_0_block8(const uint8_t* qs, const float* dw, const float* mw,
                                 const int8_t* x, const float* d, const float* s, int nb) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i i16 = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++)   /* 8 chained maddubs stay under int16 (8*3810) */
            i16 = b8_chain(i16, _mm256_load_si256((const void*) (qs + g * 32)),
                           _mm256_set1_epi32(((const int*) x)[g]));
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
        for (int g = 0; g < 8; g++) {   /* 8 chained maddubs stay under int16 (8*3810) */
            __m256i w = _mm256_load_si256((const void*) (qs + g * 32));
            for (int c = 0; c < B8_NR_KQ; c++)
                i16[c] = b8_chain(i16[c], w, _mm256_set1_epi32(((const int*) x[c])[g]));
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

static float q4_0_dot_scalar(const uint8_t* w, int nb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += JAM_Q4_0_BYTES, x += JAM_QK) {
        float d = b8_h2f(*(const uint16_t*) w);
        const uint8_t* q = w + 2;
        float s = 0.0f;
        for (int e = 0; e < 16; e++) {
            s += (float) ((q[e] & 0xF) - 8) * x[e];
            s += (float) ((q[e] >> 4) - 8) * x[16 + e];
        }
        acc += d * s;
    }
    return acc;
}

void jam_q4_0_band8_avx2(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 7 < row_end; r += 8, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * nb * 256;
            float* dw = rp->dw + (int64_t) group * nb * 8;
            float* mw = rp->mw + (int64_t) group * nb * 8;
            repack_q4_0_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, mw);
            int s = 0;
            for (; s + B8_NR_KQ <= seq; s += B8_NR_KQ)
                q4_0_block8_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, nb, ldc, J->out, r);
            for (; s < seq; s++)
                _mm256_storeu_ps(J->out + (int64_t) s * ldc + r,
                                 q4_0_block8(qs, dw, mw, J->xq + (int64_t) s * nb * JAM_QK,
                                             J->dx + (int64_t) s * nb,
                                             J->xsum + (int64_t) s * nb * 2, nb));
        }
        for (int r = row + group * 8; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q4_0_dot_scalar(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride);
    }
}

/* ---- MXFP4: LUT-decoded signed codes |w|<=12; a+128 scheme, cw = d*128*sum(w) ---- */
static const int8_t b8_mxfp4_lut[16] = { JAM_MXFP4_CODES };

static void repack_mxfp4_group8(const uint8_t* wbase, int64_t w_stride, int nb,
                                uint8_t* qs, float* dw, float* cw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_MXFP4_BYTES) {
            float d = jam_mxfp4_dhalf(w[0]);
            const uint8_t* q = w + 1;                       /* 16 packed bytes = 32 codes */
            int sumw = 0;
            for (int e = 0; e < 32; e++) {
                int8_t v = b8_mxfp4_lut[e < 16 ? (q[e] & 0xF) : (q[e - 16] >> 4)];
                qs[(int64_t) B * 256 + (e / 4) * 32 + r * 4 + (e % 4)] = (uint8_t) v;
                sumw += v;
            }
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
                i16 = b8_chain(i16, _mm256_set1_epi32((int) (((const uint32_t*) x)[g] ^ 0x80808080u)),
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
                    i16[c] = b8_chain(i16[c], _mm256_set1_epi32((int) (((const uint32_t*) x[c])[g] ^ 0x80808080u)), w);
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

static float mxfp4_dot_scalar(const uint8_t* w, int nb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += JAM_MXFP4_BYTES, x += JAM_QK) {
        float d = jam_mxfp4_dhalf(w[0]);
        const uint8_t* q = w + 1;
        float s = 0.0f;
        for (int e = 0; e < 16; e++) {
            s += (float) b8_mxfp4_lut[q[e] & 0xF] * x[e];
            s += (float) b8_mxfp4_lut[q[e] >> 4] * x[16 + e];
        }
        acc += d * s;
    }
    return acc;
}

void jam_mxfp4_band8_avx2(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 7 < row_end; r += 8, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * nb * 256;
            float* dw = rp->dw + (int64_t) group * nb * 8;
            float* cw = rp->mw + (int64_t) group * nb * 8;
            repack_mxfp4_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, cw);
            int s = 0;
            for (; s + B8_NR_KQ <= seq; s += B8_NR_KQ)
                mxfp4_block8_nr(qs, dw, cw, J->xq, J->dx, s, nb, ldc, J->out, r);
            for (; s < seq; s++)
                _mm256_storeu_ps(J->out + (int64_t) s * ldc + r,
                                 mxfp4_block8(qs, dw, cw, J->xq + (int64_t) s * nb * JAM_QK,
                                              J->dx + (int64_t) s * nb, nb));
        }
        for (int r = row + group * 8; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    mxfp4_dot_scalar(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride);
    }
}
