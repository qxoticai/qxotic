/* Q4_K @ F32 -> F32, AVX-512-VNNI. Ported from jinferjni.c (the proven fast path). The scheme: repack
 * 16 weight rows into a VNNI layout so one vpdpbusd accumulates 16 rows across the 16 i32 lanes; the
 * activation is quantized to s8 (the unsigned operand is the 4-bit weight nibble 0..15) plus per-16 f32
 * sums so the Q4_K `dmin·min` term stays out of the int dot and is corrected in float (fnmadd).
 *
 * Two jam row-range workers: jam_q4k_quant (over `seq` activation rows) then jam_q4k_band (over row
 * tiles; `tid` indexes per-worker repack scratch). Output token-major: C[col*ldc + row] (feature contig). */
#include "jam_internal.h"
#include "jam_kquant.h"
#include "jam_mxfp4.h"
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

static inline float q4k_h2f(uint16_t h) { return _cvtsh_ss(h); }

/* Column register-tile width: the _nr band kernels process JAM_VNNI_NR activation columns per weight-block
 * load, so the weight decode + per-block float scale amortize across them. NR=4 fits the 32 zmm; NR=8 spills.
 * Shared by the q4_0 / q6k / q5k tiled bands. */
#define JAM_VNNI_NR 4

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

/* ---- repack 16 Q4_K rows into the VNNI layout (qs) + per-row sub-block scales dw / mins mw ---- */
static void repack_q4k_group16(const uint8_t* wbase, int64_t w_stride, int sblocks,
                               uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 16; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += JAM_Q4K_BYTES) {
            float d = q4k_h2f(*(const uint16_t*) w), dmin = q4k_h2f(*(const uint16_t*) (w + 2));
            uint8_t sc[8], mn[8];
            jam_q4k_scales_mins(w + 4, sc, mn);
            const uint8_t* q = w + 16;
            for (int g = 0; g < 4; g++) {                        /* g = sub-block pair within the super-block */
                int sbLo = B * 8 + g * 2, sbHi = sbLo + 1, pairIdx = B * 4 + g;
                dw[sbLo * 16 + r] = d * sc[g * 2];     mw[sbLo * 16 + r] = dmin * mn[g * 2];
                dw[sbHi * 16 + r] = d * sc[g * 2 + 1]; mw[sbHi * 16 + r] = dmin * mn[g * 2 + 1];
                const uint8_t* qg = q + g * 32;                  /* raw byte = lo(sbLo) | hi(sbHi)<<4, store as-is */
                uint8_t* dst = qs + (int64_t) pairIdx * 512 + r * 4;
                for (int k = 0; k < 8; k++)
                    *(uint32_t*) (dst + k * 64) = *(const uint32_t*) (qg + k * 4);
            }
        }
    }
}

/* int dot of the repacked 16-row block-pair against one activation column -> 16 f32 partials. */
/* PACKED: each byte holds two sub-blocks' SAME element position (low nibble = sbLo, high = sbHi), so one
 * load + and/srli feeds two accumulators. `pairs` = sub-block pairs (kblocks/2). x advances 64/pair
 * (sbLo block then sbHi block); dw/mw advance 32 (per-sub-block scales); d/s indexed per sub-block. */
static inline __m512 q4k_block16(const uint8_t* qs, const float* dw, const float* mw,
                                 const int8_t* x, const float* d, const float* s, int pairs) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512 f = _mm512_setzero_ps();
    for (int p = 0; p < pairs; p++) {
        __m512i aLo = _mm512_setzero_si512(), aHi = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i pk = _mm512_load_si512((const void*) (qs + g * 64));
            aLo = _mm512_dpbusd_epi32(aLo, _mm512_and_si512(pk, m4), _mm512_set1_epi32(((const int*) x)[g]));
            aHi = _mm512_dpbusd_epi32(aHi, _mm512_and_si512(_mm512_srli_epi16(pk, 4), m4),
                                      _mm512_set1_epi32(((const int*) (x + JAM_QK))[g]));
        }
        f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(aLo), _mm512_mul_ps(_mm512_load_ps(dw),      _mm512_set1_ps(d[2*p])),   f);
        f = _mm512_fnmadd_ps(_mm512_load_ps(mw),      _mm512_set1_ps(s[4*p] + s[4*p+1]), f);
        f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(aHi), _mm512_mul_ps(_mm512_load_ps(dw + 16), _mm512_set1_ps(d[2*p+1])), f);
        f = _mm512_fnmadd_ps(_mm512_load_ps(mw + 16), _mm512_set1_ps(s[4*p+2] + s[4*p+3]), f);
        qs += 512; dw += 32; mw += 32; x += 2 * JAM_QK;
    }
    return f;
}

/* store a 16-feature result for token `col` into token-major C[col*ldc + row..row+15] — contiguous. */
static inline void q4k_store16(__m512 f, float* out, int64_t ldc, int row, int col) {
    _mm512_storeu_ps(out + (int64_t) col * ldc + row, f);
}

static float q4k_dot_scalar(const uint8_t* w, const float* x, int sblocks) {
    float acc = 0.0f;
    for (int B = 0; B < sblocks; B++, w += JAM_Q4K_BYTES, x += JAM_QKK) {
        float d = q4k_h2f(*(const uint16_t*) w), dmin = q4k_h2f(*(const uint16_t*) (w + 2));
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

/* ---- phase 2: one weight-row tile (band of JAM_VNNI_BAND rows) ---- */
/* Q4_K column register-tiled: each packed weight block loaded + nibble-decoded (and/srli) ONCE, then both
 * sub-blocks of the pair vpdpbusd'd against JAM_VNNI_NR activation columns — amortizes load AND decode. The
 * dmin·min term stays per-column float fnmadd vs the per-16 x-sums. Two int accumulators (lo/hi) per col. */
static inline void q4k_block16_nr(const uint8_t* qs, const float* dw, const float* mw, const int8_t* xq,
                                  const float* dx, const float* xs, int s0, int pairs, int kblocks,
                                  int64_t ldc, float* out, int r) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR]; const float* d[JAM_VNNI_NR]; const float* sc[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        f[c] = _mm512_setzero_ps();
        x[c]  = xq + (int64_t)(s0 + c) * kblocks * JAM_QK;
        d[c]  = dx + (int64_t)(s0 + c) * kblocks;
        sc[c] = xs + (int64_t)(s0 + c) * kblocks * 2;
    }
    for (int p = 0; p < pairs; p++) {
        __m512i aLo[JAM_VNNI_NR], aHi[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) { aLo[c] = _mm512_setzero_si512(); aHi[c] = _mm512_setzero_si512(); }
        for (int g = 0; g < 8; g++) {
            __m512i pk = _mm512_load_si512((const void*) (qs + g * 64));
            __m512i lo = _mm512_and_si512(pk, m4);                              /* decode once, reuse NR cols */
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(pk, 4), m4);
            for (int c = 0; c < JAM_VNNI_NR; c++) {
                aLo[c] = _mm512_dpbusd_epi32(aLo[c], lo, _mm512_set1_epi32(((const int*) x[c])[g]));
                aHi[c] = _mm512_dpbusd_epi32(aHi[c], hi, _mm512_set1_epi32(((const int*) (x[c] + JAM_QK))[g]));
            }
        }
        __m512 dwLo = _mm512_load_ps(dw), mwLo = _mm512_load_ps(mw);
        __m512 dwHi = _mm512_load_ps(dw + 16), mwHi = _mm512_load_ps(mw + 16);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            f[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(aLo[c]), _mm512_mul_ps(dwLo, _mm512_set1_ps(d[c][2*p])),   f[c]);
            f[c] = _mm512_fnmadd_ps(mwLo, _mm512_set1_ps(sc[c][4*p] + sc[c][4*p+1]), f[c]);
            f[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(aHi[c]), _mm512_mul_ps(dwHi, _mm512_set1_ps(d[c][2*p+1])), f[c]);
            f[c] = _mm512_fnmadd_ps(mwHi, _mm512_set1_ps(sc[c][4*p+2] + sc[c][4*p+3]), f[c]);
            x[c] += 2 * JAM_QK;
        }
        qs += 512; dw += 32; mw += 32;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) q4k_store16(f[c], out, ldc, r, s0 + c);
}

void jam_q4k_band(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int kblocks = J->kblocks, sblocks = J->dim1 / JAM_QKK, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 15 < row_end; r += 16, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * kblocks * 256;   /* packed: 256 B/sub-block-pair */
            float* dw = rp->dw + (int64_t) group * kblocks * 16;
            float* mw = rp->mw + (int64_t) group * kblocks * 16;
            repack_q4k_group16(J->w + (int64_t) r * J->w_stride, J->w_stride, sblocks, qs, dw, mw);
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                q4k_block16_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, kblocks / 2, kblocks, ldc, J->out, r);
            for (; s < seq; s++) {                  /* column tail (< NR) */
                const int8_t* x = J->xq + (int64_t) s * kblocks * JAM_QK;
                const float* d = J->dx + (int64_t) s * kblocks;
                const float* sv = J->xsum + (int64_t) s * kblocks * 2;
                q4k_store16(q4k_block16(qs, dw, mw, x, d, sv, kblocks / 2), J->out, ldc, r, s);
            }
        }
        for (int r = row + group * 16; r < row_end; r++)        /* <16-row tail: scalar */
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q4k_dot_scalar(J->w + (int64_t) r * J->w_stride, J->rhs + (int64_t) s * J->rhs_stride, sblocks);
    }
}

/* ================= Q6_K (ql[128] qh[64] scales[16] d(f16) = 210B) — shares the s8 activation quant ===
 * 6-bit weights (ql nibble + qh 2 bits, biased +32 -> 0..63 unsigned operand); scales per-16 elements;
 * the -32 offset folds into d·scale·32·xsum16 (exact activation), so no per-row min array. */
static void repack_q6k_group16(const uint8_t* wbase, int64_t w_stride, int sblocks, uint8_t* qs, float* dw) {
    for (int r = 0; r < 16; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += JAM_Q6K_BYTES) {
            const uint8_t* ql = w; const uint8_t* qh = w + 128;
            const int8_t* sc = (const int8_t*) (w + 192);
            float d = q4k_h2f(*(const uint16_t*) (w + 208));
            for (int h = 0; h < 2; h++) {
                const uint8_t* qlb = ql + h * 64; const uint8_t* qhb = qh + h * 32;
                for (int j = 0; j < 4; j++) {
                    int t0 = B * 16 + h * 8 + j * 2;
                    dw[t0 * 16 + r]       = d * sc[h * 8 + j * 2];
                    dw[(t0 + 1) * 16 + r] = d * sc[h * 8 + j * 2 + 1];
                    for (int l = 0; l < 32; l++) {
                        int qv;
                        switch (j) { case 0: qv = qlb[l] & 0xF; break; case 1: qv = qlb[32+l] & 0xF; break;
                                     case 2: qv = qlb[l] >> 4; break; default: qv = qlb[32+l] >> 4; break; }
                        qv |= ((qhb[l] >> (2 * j)) & 3) << 4;
                        int t = t0 + l / 16, e = l % 16;
                        qs[(int64_t) t * 256 + (e / 4) * 64 + r * 4 + (e % 4)] = (uint8_t) qv;
                    }
                }
            }
        }
    }
}

static inline __m512 q6k_block16(const uint8_t* qs, const float* dw,
                                 const int8_t* x, const float* d, const float* s, int subs) {
    __m512 f = _mm512_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m512i acc = _mm512_setzero_si512();
        for (int g = 0; g < 4; g++)
            acc = _mm512_dpbusd_epi32(acc, _mm512_load_si512((const void*) (qs + g * 64)),
                                      _mm512_set1_epi32(((const int*) x)[g]));
        __m512 dwv = _mm512_load_ps(dw);
        f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(dwv, _mm512_set1_ps(d[b >> 1])), f);
        f = _mm512_fnmadd_ps(dwv, _mm512_set1_ps(32.0f * s[b]), f);          /* -32 offset, exact sum */
        qs += 256; dw += 16; x += 16;
    }
    return f;
}

/* Q6_K column register-tiled: each weight block loaded once, vpdpbusd against JAM_VNNI_NR activation
 * columns into NR accumulators (the -32 offset stays the per-column exact-sum fnmadd). Stores NR results. */
static inline void q6k_block16_nr(const uint8_t* qs, const float* dw, const int8_t* xq,
                                  const float* dx, const float* xs, int s0, int subs,
                                  int kblocks, int64_t ldc, float* out, int r) {
    __m512 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR]; const float* d[JAM_VNNI_NR]; const float* sc[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        f[c] = _mm512_setzero_ps();
        x[c]  = xq + (int64_t)(s0 + c) * kblocks * JAM_QK;
        d[c]  = dx + (int64_t)(s0 + c) * kblocks;
        sc[c] = xs + (int64_t)(s0 + c) * kblocks * 2;
    }
    for (int b = 0; b < subs; b++) {
        __m512i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm512_setzero_si512();
        for (int g = 0; g < 4; g++) {
            __m512i w = _mm512_load_si512((const void*) (qs + g * 64));    /* shared across NR cols */
            for (int c = 0; c < JAM_VNNI_NR; c++)
                acc[c] = _mm512_dpbusd_epi32(acc[c], w, _mm512_set1_epi32(((const int*) x[c])[g]));
        }
        __m512 dwv = _mm512_load_ps(dw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            f[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[c]), _mm512_mul_ps(dwv, _mm512_set1_ps(d[c][b >> 1])), f[c]);
            f[c] = _mm512_fnmadd_ps(dwv, _mm512_set1_ps(32.0f * sc[c][b]), f[c]);
            x[c] += 16;
        }
        qs += 256; dw += 16;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) q4k_store16(f[c], out, ldc, r, s0 + c);
}

static float q6k_dot_scalar(const uint8_t* w, const float* x, int sblocks) {
    float acc = 0.0f;
    for (int B = 0; B < sblocks; B++, w += JAM_Q6K_BYTES, x += JAM_QKK) {
        const uint8_t* ql = w; const uint8_t* qh = w + 128;
        const int8_t* sc = (const int8_t*) (w + 192);
        float d = q4k_h2f(*(const uint16_t*) (w + 208));
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

void jam_q6k_band(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int kblocks = J->kblocks, sblocks = J->dim1 / JAM_QKK, subs16 = J->dim1 / 16, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 15 < row_end; r += 16, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * kblocks * 512;
            float* dw = rp->dw + (int64_t) group * subs16 * 16;
            repack_q6k_group16(J->w + (int64_t) r * J->w_stride, J->w_stride, sblocks, qs, dw);
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                q6k_block16_nr(qs, dw, J->xq, J->dx, J->xsum, s, subs16, kblocks, ldc, J->out, r);
            for (; s < seq; s++) {                  /* column tail (< NR) */
                const int8_t* x = J->xq + (int64_t) s * kblocks * JAM_QK;
                const float* d = J->dx + (int64_t) s * kblocks;
                const float* sv = J->xsum + (int64_t) s * kblocks * 2;
                q4k_store16(q6k_block16(qs, dw, x, d, sv, subs16), J->out, ldc, r, s);
            }
        }
        for (int r = row + group * 16; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q6k_dot_scalar(J->w + (int64_t) r * J->w_stride, J->rhs + (int64_t) s * J->rhs_stride, sblocks);
    }
}

/* ================= Q8_0 16-row VNNI repack (the kernel that beat llama.cpp; mirrors Q4_K) ===========
 * Q8_0 weights are signed int8, so vpdpbusd's unsigned operand is the ACTIVATION (a+128) and the signed
 * operand is the repacked weight; the +128 bias is corrected in float per row/block via cw = d·128·Σw.
 * One vpdpbusd accumulates 16 ROWS in the 16 i32 lanes -> per-row scales are per-lane, 16 outputs store
 * contiguously (token-major), zero horizontal reductions. block_q8_0 = { fp16 d; int8 qs[32] } = 34B. */
#define JAM_Q8_BYTES 34

static void repack_q8_group16(const uint8_t* wbase, int64_t w_stride, int nb,
                              uint8_t* qs, float* dw, float* cw) {
    for (int r = 0; r < 16; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q8_BYTES) {
            float d = q4k_h2f(*(const uint16_t*) w);
            const int8_t* q = (const int8_t*) (w + 2);
            int sumw = 0;
            for (int g = 0; g < 8; g++)
                for (int e = 0; e < 4; e++) {
                    int8_t v = q[g * 4 + e];
                    qs[(int64_t) B * 512 + g * 64 + r * 4 + e] = (uint8_t) v;   /* s8 weight, lane = row r */
                    sumw += v;
                }
            dw[(int64_t) B * 16 + r] = d;
            cw[(int64_t) B * 16 + r] = d * 128.0f * (float) sumw;               /* +128-bias correction */
        }
    }
}

/* 16-row dot of the repacked weights against one activation column -> 16 f32 partials. */
static inline __m512 q8_block16(const uint8_t* qs, const float* dw, const float* cw,
                                const int8_t* x, const float* dx, int nb) {
    __m512 f = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m512i acc = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++)
            acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(((const int*) x)[g] ^ 0x80808080),  /* a+128 (u8) */
                                      _mm512_load_si512((const void*) (qs + g * 64)));            /* weight (s8) */
        __m512 da = _mm512_set1_ps(dx[b]);
        f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(_mm512_load_ps(dw), da), f);   /* + dot·dw·da */
        f = _mm512_fnmadd_ps(_mm512_load_ps(cw), da, f);                                          /* − cw·da     */
        qs += 512; dw += 16; cw += 16; x += JAM_QK;
    }
    return f;
}

static float q8_scalar_dot(const uint8_t* w, int nb, const float* x) {   /* <16-row tail: exact float dot */
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += JAM_Q8_BYTES, x += 32) {
        float d = q4k_h2f(*(const uint16_t*) w);
        const int8_t* q = (const int8_t*) (w + 2);
        float s = 0.0f;
        for (int e = 0; e < 32; e++) s += (float) q[e] * x[e];
        acc += d * s;
    }
    return acc;
}

/* Q8_0 column register-tiled: the s8 weight block is loaded once and vpdpbusd'd against JAM_VNNI_NR
 * activation columns (each a+128 -> the u8 operand; the +128 bias stays the per-col cw·da fnmadd). Q8_0 is
 * the most byte-heavy band (512 B/block unpacked), so amortizing the weight load across NR cols helps most. */
static inline void q8_block16_nr(const uint8_t* qs, const float* dw, const float* cw, const int8_t* xq,
                                 const float* dx, int s0, int nb, int64_t ldc, float* out, int r) {
    __m512 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR]; const float* d[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        f[c] = _mm512_setzero_ps();
        x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK;
        d[c] = dx + (int64_t)(s0 + c) * nb;
    }
    for (int b = 0; b < nb; b++) {
        __m512i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i w = _mm512_load_si512((const void*) (qs + g * 64));    /* weight s8, shared across NR cols */
            for (int c = 0; c < JAM_VNNI_NR; c++)
                acc[c] = _mm512_dpbusd_epi32(acc[c], _mm512_set1_epi32(((const int*) x[c])[g] ^ 0x80808080), w);
        }
        __m512 dwv = _mm512_load_ps(dw), cwv = _mm512_load_ps(cw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            __m512 da = _mm512_set1_ps(d[c][b]);
            f[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[c]), _mm512_mul_ps(dwv, da), f[c]);
            f[c] = _mm512_fnmadd_ps(cwv, da, f[c]);
            x[c] += JAM_QK;
        }
        qs += 512; dw += 16; cw += 16;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) q4k_store16(f[c], out, ldc, r, s0 + c);
}

void jam_q8_0_repack_band(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 15 < row_end; r += 16, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * nb * 512;
            float* dw = rp->dw + (int64_t) group * nb * 16;
            float* cw = rp->mw + (int64_t) group * nb * 16;     /* mw scratch repurposed for cw */
            repack_q8_group16(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, cw);
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                q8_block16_nr(qs, dw, cw, J->xq, J->dx, s, nb, ldc, J->out, r);
            for (; s < seq; s++)               /* column tail (< NR) */
                q4k_store16(q8_block16(qs, dw, cw, J->xq + (int64_t) s * nb * JAM_QK,
                                       J->dx + (int64_t) s * nb, nb), J->out, ldc, r, s);
        }
        for (int r = row + group * 16; r < row_end; r++)        /* <16-row tail: scalar exact dot */
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q8_scalar_dot(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride);
    }
}

/* ================= Q4_0 16-row VNNI repack — reuses q4k_block16/q4k_store16/jam_q4k_quant ===========
 * Q4_0 = { fp16 d; nibble qs[16] } = 18B, value = d·(nibble-8). The nibble (0..15) is the UNSIGNED
 * vpdpbusd operand (like Q4_K, unlike Q8_0); the -8 offset is the Q4_K "min" with (dw,mw)=(d,8·d),
 * corrected in float via the exact activation sums. Only the nibble unpack is Q4_0-specific.
 *
 * PERF NOTE — this kernel is at the compute ceiling; do NOT expect a "Q4_0-specialized" speedup.
 * Q4_0 and Q8_0 dot the SAME K, so they issue the SAME 8 vpdpbusd/block (m·n·k int8 MACs either way);
 * 4-bit can only ADD the nibble unpack, never remove MACs. We hide the unpack (one and + one srli/and
 * straight into vpdpbusd, -8 folded into the mw bias, no vpshufb LUT), so Q4_0 ≈ Q8_0 at the GEMM level
 * (~2.6 GMAC/s @ m4096·n512·k2048, ~3x llama.cpp's tinyBLAS) — the most you can ask of a compute-bound
 * prefill. At MATCHED threads the full model TIES llama.cpp: Llama-1B Q4_0 pp512 ~2234 t/s vs llama.cpp
 * ~2262 (both 32 threads), and Q8_0 wins outright (~1.4x). jinfer just needs all logical CPUs — both
 * JAM_NUM_THREADS and the jinfer FJP pool. Capping either to 16 starves the GEMM + the Java non-GEMM and
 * is what makes Q4_0 *look* ~10% slow; it is the thread budget, NOT this band. */
#define JAM_Q40_BYTES 18

/* PACKED repack: keep 2 nibbles/byte (256 B/block/16rows, HALF of Q8_0) so the band stays L1-resident
 * and weight reads halve. Each byte holds two element-planes: low = element i*8+e, high = i*8+4+e, so one
 * load + and/srli yields both vpdpbusd operands. */
static void repack_q4_0_group16(const uint8_t* wbase, int64_t w_stride, int nb,
                                uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 16; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q40_BYTES) {
            float d = q4k_h2f(*(const uint16_t*) w);
            const uint8_t* q = w + 2;                              /* qs[16]; byte j: low=elem j, high=elem j+16 */
            #define Q40_NIB(idx) ((idx) < 16 ? (q[idx] & 0xF) : (q[(idx) - 16] >> 4))
            for (int i = 0; i < 4; i++)                           /* 4 planes × 8 elements */
                for (int e = 0; e < 4; e++) {
                    uint8_t lo = Q40_NIB(i * 8 + e), hi = Q40_NIB(i * 8 + 4 + e);
                    qs[(int64_t) B * 256 + i * 64 + r * 4 + e] = (uint8_t)(lo | (hi << 4));
                }
            #undef Q40_NIB
            dw[(int64_t) B * 16 + r] = d;
            mw[(int64_t) B * 16 + r] = 8.0f * d;                  /* the -8 offset term */
        }
    }
}

/* 16-row dot from the PACKED repack: per 8-element plane, one load -> lo(&0xF) + hi(>>4) -> two vpdpbusd. */
static inline __m512 q4_0_block16(const uint8_t* qs, const float* dw, const float* mw,
                                  const int8_t* x, const float* dx, const float* xs, int nb) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512 f = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m512i acc = _mm512_setzero_si512();
        for (int i = 0; i < 4; i++) {
            __m512i pk = _mm512_load_si512((const void*) (qs + i * 64));
            acc = _mm512_dpbusd_epi32(acc, _mm512_and_si512(pk, m4),
                                      _mm512_set1_epi32(((const int*) x)[2 * i]));         /* low nibbles  */
            acc = _mm512_dpbusd_epi32(acc, _mm512_and_si512(_mm512_srli_epi16(pk, 4), m4),
                                      _mm512_set1_epi32(((const int*) x)[2 * i + 1]));     /* high nibbles */
        }
        __m512 da = _mm512_set1_ps(dx[b]);
        f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(_mm512_load_ps(dw), da), f);
        f = _mm512_fnmadd_ps(_mm512_load_ps(mw), _mm512_set1_ps(xs[2 * b] + xs[2 * b + 1]), f);
        qs += 256; dw += 16; mw += 16; x += JAM_QK;
    }
    return f;
}

/* Column register-tiled variant: load+decode each weight block ONCE, vpdpbusd it against JAM_VNNI_NR
 * activation columns into NR accumulators — amortizes the weight load + nibble decode (and/srli) + the
 * float scale across NR columns, so vpdpbusd stops being ~1/3 of the inner loop. Stores all NR results. */
static inline void q4_0_block16_nr(const uint8_t* qs, const float* dw, const float* mw,
                                   const int8_t* xq, const float* dx, const float* xs,
                                   int s0, int nb, int64_t ldc, float* out, int r) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) { f[c] = _mm512_setzero_ps(); x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK; }
    for (int b = 0; b < nb; b++) {
        __m512i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm512_setzero_si512();
        for (int i = 0; i < 4; i++) {
            __m512i pk = _mm512_load_si512((const void*) (qs + i * 64));
            __m512i lo = _mm512_and_si512(pk, m4);                          /* decode ONCE, reuse across NR cols */
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(pk, 4), m4);
            for (int c = 0; c < JAM_VNNI_NR; c++) {
                acc[c] = _mm512_dpbusd_epi32(acc[c], lo, _mm512_set1_epi32(((const int*) x[c])[2 * i]));
                acc[c] = _mm512_dpbusd_epi32(acc[c], hi, _mm512_set1_epi32(((const int*) x[c])[2 * i + 1]));
            }
        }
        __m512 dwv = _mm512_load_ps(dw), mwv = _mm512_load_ps(mw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            __m512 da = _mm512_set1_ps(dx[(int64_t)(s0 + c) * nb + b]);
            float xsum = xs[((int64_t)(s0 + c) * nb + b) * 2] + xs[((int64_t)(s0 + c) * nb + b) * 2 + 1];
            f[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[c]), _mm512_mul_ps(dwv, da), f[c]);
            f[c] = _mm512_fnmadd_ps(mwv, _mm512_set1_ps(xsum), f[c]);
            x[c] += JAM_QK;
        }
        qs += 256; dw += 16; mw += 16;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) q4k_store16(f[c], out, ldc, r, s0 + c);
}

static float q4_0_scalar_dot(const uint8_t* w, int nb, const float* x) {   /* <16-row tail: exact float dot */
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += JAM_Q40_BYTES, x += 32) {
        float d = q4k_h2f(*(const uint16_t*) w);
        const uint8_t* q = w + 2;
        float s = 0.0f;
        for (int e = 0; e < 16; e++) {
            s += (float)((q[e] & 0xF) - 8) * x[e];
            s += (float)((q[e] >> 4)  - 8) * x[e + 16];
        }
        acc += d * s;
    }
    return acc;
}

void jam_q4_0_repack_band(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 15 < row_end; r += 16, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * nb * 256;   /* packed: 256 B/block */
            float* dw = rp->dw + (int64_t) group * nb * 16;
            float* mw = rp->mw + (int64_t) group * nb * 16;
            repack_q4_0_group16(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, mw);
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                q4_0_block16_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, nb, ldc, J->out, r);
            for (; s < seq; s++)               /* column tail (< NR) */
                q4k_store16(q4_0_block16(qs, dw, mw, J->xq + (int64_t) s * nb * JAM_QK,
                                         J->dx + (int64_t) s * nb, J->xsum + (int64_t) s * nb * 2, nb),
                            J->out, ldc, r, s);
        }
        for (int r = row + group * 16; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q4_0_scalar_dot(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride);
    }
}

/* ================= MXFP4 16-row VNNI repack (gpt-oss experts) — Q4_0 packing + Q8_0 +128/cw + vpshufb ===
 * Block = { e8m0; nibble qs[16] } = 17B; nibble -> SIGNED int8 code (LUT). value = code · dhalf(e). The
 * codes are signed so (like Q8_0) the unsigned vpdpbusd operand is the activation (a^0x80=+128) and the
 * signed operand is the decoded code; +128 bias corrected per row via cw = 128·dhalf·Σcode. Nibbles kept
 * PACKED (256 B/block); q4k-style vpshufb decodes both planes per load. */
static const int8_t jam_mxfp4_codes[16] = { JAM_MXFP4_CODES };

static void repack_mxfp4_group16(const uint8_t* wbase, int64_t w_stride, int nb,
                                 uint8_t* qs, float* dw, float* cw) {
    for (int r = 0; r < 16; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += 17) {              /* mxfp4 block = 17 bytes */
            float dh = jam_mxfp4_dhalf(w[0]);
            const uint8_t* q = w + 1;                         /* qs[16]; byte j: low=elem j, high=elem j+16 */
            int sumc = 0;
            #define MXNIB(idx) ((idx) < 16 ? (q[idx] & 0xF) : (q[(idx) - 16] >> 4))
            for (int i = 0; i < 4; i++)
                for (int e = 0; e < 4; e++) {
                    uint8_t lo = MXNIB(i * 8 + e), hi = MXNIB(i * 8 + 4 + e);
                    qs[(int64_t) B * 256 + i * 64 + r * 4 + e] = (uint8_t)(lo | (hi << 4));   /* PACKED nibbles */
                    sumc += jam_mxfp4_codes[lo] + jam_mxfp4_codes[hi];                        /* Σ decoded code */
                }
            #undef MXNIB
            dw[(int64_t) B * 16 + r] = dh;
            cw[(int64_t) B * 16 + r] = 128.0f * dh * (float) sumc;   /* +128 bias correction */
        }
    }
}

static inline __m512 mxfp4_block16(const uint8_t* qs, const float* dw, const float* cw,
                                   const int8_t* x, const float* dx, int nb) {
    const __m512i m4  = _mm512_set1_epi8(0x0F);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(JAM_MXFP4_CODES));   /* nibble -> s8 code, ×4 lanes */
    __m512 f = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m512i acc = _mm512_setzero_si512();
        for (int i = 0; i < 4; i++) {
            __m512i pk  = _mm512_load_si512((const void*) (qs + i * 64));
            __m512i loc = _mm512_shuffle_epi8(lut, _mm512_and_si512(pk, m4));                       /* lo nibbles -> codes */
            __m512i hic = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(pk, 4), m4)); /* hi nibbles -> codes */
            acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(((const int*) x)[2*i]   ^ 0x80808080), loc); /* act u8 · code s8 */
            acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(((const int*) x)[2*i+1] ^ 0x80808080), hic);
        }
        __m512 da = _mm512_set1_ps(dx[b]);
        f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(_mm512_load_ps(dw), da), f);
        f = _mm512_fnmadd_ps(_mm512_load_ps(cw), da, f);
        qs += 256; dw += 16; cw += 16; x += JAM_QK;
    }
    return f;
}

static float mxfp4_scalar_dot(const uint8_t* w, int nb, const float* x) {   /* <16-row tail: exact float dot */
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += 17, x += 32) {
        float dh = jam_mxfp4_dhalf(w[0]);
        const uint8_t* q = w + 1;
        float s = 0.0f;
        for (int e = 0; e < 16; e++) {
            s += (float) jam_mxfp4_codes[q[e] & 0xF] * x[e];
            s += (float) jam_mxfp4_codes[q[e] >> 4]  * x[e + 16];
        }
        acc += dh * s;
    }
    return acc;
}

/* MXFP4 column register-tiled: each packed block loaded + vpshufb-LUT-decoded (lo/hi nibble -> s8 code)
 * ONCE, then vpdpbusd'd against JAM_VNNI_NR activation columns (a+128 u8; +128 bias is the per-col cw·da
 * fnmadd). MXFP4 has the heaviest decode (2 vpshufb + and/srli per nibble-plane), so this amortizes most. */
static inline void mxfp4_block16_nr(const uint8_t* qs, const float* dw, const float* cw, const int8_t* xq,
                                    const float* dx, int s0, int nb, int64_t ldc, float* out, int r) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(JAM_MXFP4_CODES));
    __m512 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR]; const float* d[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        f[c] = _mm512_setzero_ps();
        x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK;
        d[c] = dx + (int64_t)(s0 + c) * nb;
    }
    for (int b = 0; b < nb; b++) {
        __m512i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm512_setzero_si512();
        for (int i = 0; i < 4; i++) {
            __m512i pk  = _mm512_load_si512((const void*) (qs + i * 64));     /* loaded + decoded once */
            __m512i loc = _mm512_shuffle_epi8(lut, _mm512_and_si512(pk, m4));
            __m512i hic = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(pk, 4), m4));
            for (int c = 0; c < JAM_VNNI_NR; c++) {
                acc[c] = _mm512_dpbusd_epi32(acc[c], _mm512_set1_epi32(((const int*) x[c])[2*i]   ^ 0x80808080), loc);
                acc[c] = _mm512_dpbusd_epi32(acc[c], _mm512_set1_epi32(((const int*) x[c])[2*i+1] ^ 0x80808080), hic);
            }
        }
        __m512 dwv = _mm512_load_ps(dw), cwv = _mm512_load_ps(cw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            __m512 da = _mm512_set1_ps(d[c][b]);
            f[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[c]), _mm512_mul_ps(dwv, da), f[c]);
            f[c] = _mm512_fnmadd_ps(cwv, da, f[c]);
            x[c] += JAM_QK;
        }
        qs += 256; dw += 16; cw += 16;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) q4k_store16(f[c], out, ldc, r, s0 + c);
}

void jam_mxfp4_repack_band(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int nb = J->kblocks, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 15 < row_end; r += 16, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * nb * 256;   /* packed: 256 B/block */
            float* dw = rp->dw + (int64_t) group * nb * 16;
            float* cw = rp->mw + (int64_t) group * nb * 16;
            repack_mxfp4_group16(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, cw);
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                mxfp4_block16_nr(qs, dw, cw, J->xq, J->dx, s, nb, ldc, J->out, r);
            for (; s < seq; s++)               /* column tail (< NR) */
                q4k_store16(mxfp4_block16(qs, dw, cw, J->xq + (int64_t) s * nb * JAM_QK,
                                          J->dx + (int64_t) s * nb, nb), J->out, ldc, r, s);
        }
        for (int r = row + group * 16; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    mxfp4_scalar_dot(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride);
    }
}

/* ================= Q5_K 16-row VNNI repack — Q4_K scheme, 5-bit (byte-expanded) =====================
 * Q5_K = { d dmin scales[12] qh[32] qs[128] } = 176B. value = d·sc·q5 - dmin·mn, q5 = qs nibble | (qh
 * bit<<4) ∈ 0..31. 5+5 bits don't pack 2/byte, so the repack stores the 0..31 value as a byte (512 B/
 * sub-block); the dot/scale/min handling is exactly Q4_K (unsigned weight, dmin·min via xsum). */
static inline __m512 q5k_block16(const uint8_t* qs, const float* dw, const float* mw,
                                 const int8_t* x, const float* d, const float* s, int subs) {
    __m512 f = _mm512_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m512i acc = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++)
            acc = _mm512_dpbusd_epi32(acc, _mm512_load_si512((const void*) (qs + g * 64)),
                                      _mm512_set1_epi32(((const int*) x)[g]));
        f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(_mm512_load_ps(dw), _mm512_set1_ps(d[b])), f);
        f = _mm512_fnmadd_ps(_mm512_load_ps(mw), _mm512_set1_ps(s[2 * b] + s[2 * b + 1]), f);
        qs += 512; dw += 16; mw += 16; x += JAM_QK;
    }
    return f;
}

static void repack_q5k_group16(const uint8_t* wbase, int64_t w_stride, int sblocks,
                               uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 16; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += JAM_Q5K_BYTES) {
            float d = q4k_h2f(*(const uint16_t*) w), dmin = q4k_h2f(*(const uint16_t*) (w + 2));
            uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
            const uint8_t* qh = w + 16; const uint8_t* q5 = w + 48;
            for (int g = 0; g < 4; g++) {
                int sbLo = B * 8 + g * 2, sbHi = sbLo + 1;
                dw[sbLo * 16 + r] = d * sc[g * 2];     mw[sbLo * 16 + r] = dmin * mn[g * 2];
                dw[sbHi * 16 + r] = d * sc[g * 2 + 1]; mw[sbHi * 16 + r] = dmin * mn[g * 2 + 1];
                const uint8_t* q = q5 + g * 32;
                for (int i = 0; i < 32; i++) {
                    uint8_t lo = (q[i] & 0xF) | (((qh[i] >> (2 * g))     & 1) << 4);   /* 0..31 */
                    uint8_t hi = (q[i] >> 4)  | (((qh[i] >> (2 * g + 1)) & 1) << 4);
                    qs[(int64_t) sbLo * 512 + (i >> 2) * 64 + r * 4 + (i & 3)] = lo;
                    qs[(int64_t) sbHi * 512 + (i >> 2) * 64 + r * 4 + (i & 3)] = hi;
                }
            }
        }
    }
}

static float q5k_dot_scalar(const uint8_t* w, const float* x, int sblocks) {   /* <16-row tail: exact */
    float acc = 0.0f;
    for (int B = 0; B < sblocks; B++, w += JAM_Q5K_BYTES, x += JAM_QKK) {
        float d = q4k_h2f(*(const uint16_t*) w), dmin = q4k_h2f(*(const uint16_t*) (w + 2));
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

/* Q5_K column register-tiled: each weight block loaded once, vpdpbusd against JAM_VNNI_NR activation
 * columns; the dmin·min term stays the per-column float fnmadd against the per-16 x-sums. Stores NR results. */
static inline void q5k_block16_nr(const uint8_t* qs, const float* dw, const float* mw, const int8_t* xq,
                                  const float* dx, const float* xs, int s0, int kblocks, int64_t ldc,
                                  float* out, int r) {
    __m512 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR]; const float* d[JAM_VNNI_NR]; const float* sc[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        f[c] = _mm512_setzero_ps();
        x[c]  = xq + (int64_t)(s0 + c) * kblocks * JAM_QK;
        d[c]  = dx + (int64_t)(s0 + c) * kblocks;
        sc[c] = xs + (int64_t)(s0 + c) * kblocks * 2;
    }
    for (int b = 0; b < kblocks; b++) {
        __m512i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i w = _mm512_load_si512((const void*) (qs + g * 64));    /* shared across NR cols */
            for (int c = 0; c < JAM_VNNI_NR; c++)
                acc[c] = _mm512_dpbusd_epi32(acc[c], w, _mm512_set1_epi32(((const int*) x[c])[g]));
        }
        __m512 dwv = _mm512_load_ps(dw), mwv = _mm512_load_ps(mw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            f[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[c]), _mm512_mul_ps(dwv, _mm512_set1_ps(d[c][b])), f[c]);
            f[c] = _mm512_fnmadd_ps(mwv, _mm512_set1_ps(sc[c][2 * b] + sc[c][2 * b + 1]), f[c]);
            x[c] += JAM_QK;
        }
        qs += 512; dw += 16; mw += 16;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) q4k_store16(f[c], out, ldc, r, s0 + c);
}

void jam_q5k_repack_band(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int kblocks = J->kblocks, sblocks = J->dim1 / JAM_QKK, seq = J->seq;
    const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 15 < row_end; r += 16, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * kblocks * 512;   /* byte-expanded: 512 B/sub-block */
            float* dw = rp->dw + (int64_t) group * kblocks * 16;
            float* mw = rp->mw + (int64_t) group * kblocks * 16;
            repack_q5k_group16(J->w + (int64_t) r * J->w_stride, J->w_stride, sblocks, qs, dw, mw);
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                q5k_block16_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, kblocks, ldc, J->out, r);
            for (; s < seq; s++)               /* column tail (< NR) */
                q4k_store16(q5k_block16(qs, dw, mw, J->xq + (int64_t) s * kblocks * JAM_QK,
                                        J->dx + (int64_t) s * kblocks, J->xsum + (int64_t) s * kblocks * 2, kblocks),
                            J->out, ldc, r, s);
        }
        for (int r = row + group * 16; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q5k_dot_scalar(J->w + (int64_t) r * J->w_stride, J->rhs + (int64_t) s * J->rhs_stride, sblocks);
    }
}
