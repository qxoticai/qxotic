/* AVX-VNNI kernels — this TU only, built with -mavxvnni -mavx2 -mfma -mf16c. Bound at create when the
 * CPU has AVX-VNNI (256-bit vpdpbusd in the VEX encoding) but NOT AVX-512 — i.e. modern CLIENT chips
 * (Alder/Raptor Lake). The F32 path reuses the AVX2 kernel (same 256-bit float); only Q8_0 changes:
 * the int8 dot is one vpdpbusd instead of AVX2's maddubs+madd (no int16 saturation edge, fewer ops).
 *
 * NOTE: _mm256_dpbusd_avx_epi32 (the _avx_ spelling) is the AVX-VNNI VEX intrinsic — distinct from the
 * AVX-512-VL _mm256_dpbusd_epi32 used by the AVX-512 TU. */
#include "jam_internal.h"
#include "jam_kquant.h"
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#define JAM_DOT(aqa, sqb) \
    _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aqa, sqb))

#define JAM_BLK    jam_q8_blk
#define JAM_DECODE jam_decode_q8_0_256
#define JAM_MM_NAME jam_mm_q8_0_avxvnni
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK    jam_mxfp4_blk
#define JAM_DECODE jam_decode_mxfp4_256
#define JAM_MM_NAME jam_mm_mxfp4_avxvnni
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK    jam_q4_0_blk
#define JAM_DECODE jam_decode_q4_0_256
#define JAM_MM_NAME jam_mm_q4_0_avxvnni
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME
#undef JAM_DOT

/* ---- cached-repack rp kernels, AVX-VNNI: same 8-feature-wide layout + repacks as the avx2 maddubs versions,
 * but the dot is ONE vpdpbusd (u8×s8 -> int32 accumulate) instead of maddubs+madd. vpdpbusd has no scale
 * operand, so the K-quant 6-bit scale is applied with a vpmulld after each sub-block (still 1 dot op vs 2). */
#define VDOT(acc, u, s) _mm256_dpbusd_avx_epi32((acc), (u), (s))

void jam_mm_q4k_rp_avxvnni(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* RP = (const uint8_t*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t grp_bytes = (size_t) sblocks * sizeof(jam_q4k_rpblock);
    const int mrows = J->m;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    for (int grp = rb; grp < re; ++grp) {
        int i0 = grp * 8, nf = mrows - i0 < 8 ? mrows - i0 : 8;
        const uint8_t* gbase = RP + (size_t) grp * grp_bytes;
        for (int j0 = 0; j0 < n; j0 += 4) {
            int nt = n - j0 < 4 ? n - j0 : 4;
            __m256 acc[4]; for (int t = 0; t < 4; ++t) acc[t] = _mm256_setzero_ps();
            for (int B = 0; B < sblocks; ++B) {
                const jam_q4k_rpblock* blk = (const jam_q4k_rpblock*) gbase + B;
                const float* d = blk->d; const float* dmin = blk->dmin;
                const uint8_t* sc = blk->sc; const uint8_t* mn = blk->mn; const uint8_t* qs = blk->qs;
                __m256i sumi[4]; for (int t = 0; t < 4; ++t) sumi[t] = _mm256_setzero_si256();
                for (int s = 0; s < 8; ++s) {
                    __m256i sc_v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(sc + s*8)));
                    __m256i sb[4]; for (int t = 0; t < nt; ++t) sb[t] = _mm256_setzero_si256();
                    for (int gp = 0; gp < 8; ++gp) {
                        __m128i x = _mm_loadu_si128((const __m128i*)(qs + (size_t)(s*8 + gp)*16));
                        __m256i w = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(x, 4), m4), _mm_and_si128(x, m4));
                        for (int t = 0; t < nt; ++t)
                            sb[t] = VDOT(sb[t], w, _mm256_set1_epi32(*(const int*)(AQ + (size_t)(j0+t)*k + (size_t) B*JAM_QKK + s*32 + gp*4)));
                    }
                    for (int t = 0; t < nt; ++t) sumi[t] = _mm256_add_epi32(sumi[t], _mm256_mullo_epi32(sb[t], sc_v));
                }
                __m256 d_v = _mm256_loadu_ps(d), dmin_v = _mm256_loadu_ps(dmin);
                __m256 minsum[4]; for (int t = 0; t < nt; ++t) minsum[t] = _mm256_setzero_ps();
                for (int s = 0; s < 8; ++s) {       /* load each min once, fan out over tokens */
                    __m256 mnf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(mn + s*8))));
                    for (int t = 0; t < nt; ++t) minsum[t] = _mm256_fmadd_ps(mnf, _mm256_set1_ps(J->asum[(size_t)(j0+t)*nb + B*8 + s]), minsum[t]);
                }
                for (int t = 0; t < nt; ++t) {
                    __m256 contrib = _mm256_sub_ps(_mm256_mul_ps(d_v, _mm256_cvtepi32_ps(sumi[t])), _mm256_mul_ps(dmin_v, minsum[t]));
                    acc[t] = _mm256_fmadd_ps(contrib, _mm256_set1_ps(AD[(size_t)(j0+t)*nb + B]), acc[t]);
                }
            }
            for (int t = 0; t < nt; ++t) { float tmp[8]; _mm256_storeu_ps(tmp, acc[t]); for (int f = 0; f < nf; ++f) C[(size_t)(j0+t)*ldc + (i0+f)] = tmp[f]; }
        }
    }
}

void jam_mm_q5k_rp_avxvnni(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* RP = (const uint8_t*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t grp_bytes = (size_t) sblocks * sizeof(jam_q5k_rpblock);
    const int mrows = J->m;
    for (int grp = rb; grp < re; ++grp) {
        int i0 = grp * 8, nf = mrows - i0 < 8 ? mrows - i0 : 8;
        const uint8_t* gbase = RP + (size_t) grp * grp_bytes;
        for (int j0 = 0; j0 < n; j0 += 4) {
            int nt = n - j0 < 4 ? n - j0 : 4;
            __m256 acc[4]; for (int t = 0; t < 4; ++t) acc[t] = _mm256_setzero_ps();
            for (int B = 0; B < sblocks; ++B) {
                const jam_q5k_rpblock* blk = (const jam_q5k_rpblock*) gbase + B;
                const float* d = blk->d; const float* dmin = blk->dmin;
                const uint8_t* sc = blk->sc; const uint8_t* mn = blk->mn; const uint8_t* qs = blk->qs;
                __m256i sumi[4]; for (int t = 0; t < 4; ++t) sumi[t] = _mm256_setzero_si256();
                for (int s = 0; s < 8; ++s) {
                    __m256i sc_v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(sc + s*8)));
                    __m256i sb[4]; for (int t = 0; t < nt; ++t) sb[t] = _mm256_setzero_si256();
                    for (int gp = 0; gp < 8; ++gp) {
                        __m256i w = _mm256_loadu_si256((const __m256i*)(qs + (size_t)(s*8 + gp)*32));
                        for (int t = 0; t < nt; ++t)
                            sb[t] = VDOT(sb[t], w, _mm256_set1_epi32(*(const int*)(AQ + (size_t)(j0+t)*k + (size_t) B*JAM_QKK + s*32 + gp*4)));
                    }
                    for (int t = 0; t < nt; ++t) sumi[t] = _mm256_add_epi32(sumi[t], _mm256_mullo_epi32(sb[t], sc_v));
                }
                __m256 d_v = _mm256_loadu_ps(d), dmin_v = _mm256_loadu_ps(dmin);
                __m256 minsum[4]; for (int t = 0; t < nt; ++t) minsum[t] = _mm256_setzero_ps();
                for (int s = 0; s < 8; ++s) {       /* load each min once, fan out over tokens */
                    __m256 mnf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(mn + s*8))));
                    for (int t = 0; t < nt; ++t) minsum[t] = _mm256_fmadd_ps(mnf, _mm256_set1_ps(J->asum[(size_t)(j0+t)*nb + B*8 + s]), minsum[t]);
                }
                for (int t = 0; t < nt; ++t) {
                    __m256 contrib = _mm256_sub_ps(_mm256_mul_ps(d_v, _mm256_cvtepi32_ps(sumi[t])), _mm256_mul_ps(dmin_v, minsum[t]));
                    acc[t] = _mm256_fmadd_ps(contrib, _mm256_set1_ps(AD[(size_t)(j0+t)*nb + B]), acc[t]);
                }
            }
            for (int t = 0; t < nt; ++t) { float tmp[8]; _mm256_storeu_ps(tmp, acc[t]); for (int f = 0; f < nf; ++f) C[(size_t)(j0+t)*ldc + (i0+f)] = tmp[f]; }
        }
    }
}

void jam_mm_q6k_rp_avxvnni(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* RP = (const uint8_t*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t grp_bytes = (size_t) sblocks * sizeof(jam_q6k_rpblock);
    const int mrows = J->m;
    for (int grp = rb; grp < re; ++grp) {
        int i0 = grp * 8, nf = mrows - i0 < 8 ? mrows - i0 : 8;
        const uint8_t* gbase = RP + (size_t) grp * grp_bytes;
        for (int j0 = 0; j0 < n; j0 += 4) {
            int nt = n - j0 < 4 ? n - j0 : 4;
            __m256 acc[4]; for (int t = 0; t < 4; ++t) acc[t] = _mm256_setzero_ps();
            for (int B = 0; B < sblocks; ++B) {
                const jam_q6k_rpblock* blk = (const jam_q6k_rpblock*) gbase + B;
                const float* d = blk->d; const int8_t* sc = blk->sc; const uint8_t* qs = blk->qs;
                __m256i sumi[4]; for (int t = 0; t < 4; ++t) sumi[t] = _mm256_setzero_si256();
                for (int s16 = 0; s16 < 16; ++s16) {
                    int h = s16 / 8, rem = s16 % 8, g = rem / 2, half = rem & 1;
                    int ebase = (h*4 + g)*32 + half*16;
                    __m256i sc_v = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(sc + s16*8)));   /* signed */
                    /* All 4 weight vectors up front, ONE sb rotating over tokens: live set is
                     * acc[4]+sumi[4]+w[4]+sb+sc_v+broadcast = 15 ymm. The sb[4]-staging shape spilled
                     * sumi to the stack (perf: vpaddd (%rsp) 14.7%) - same disease the avx2 kernel had. */
                    __m256i w0 = _mm256_loadu_si256((const __m256i*)(qs + (size_t)(s16*4    )*32));
                    __m256i w1 = _mm256_loadu_si256((const __m256i*)(qs + (size_t)(s16*4 + 1)*32));
                    __m256i w2 = _mm256_loadu_si256((const __m256i*)(qs + (size_t)(s16*4 + 2)*32));
                    __m256i w3 = _mm256_loadu_si256((const __m256i*)(qs + (size_t)(s16*4 + 3)*32));
                    #define Q6K_TOK(t) do { \
                        const int8_t* ap = AQ + (size_t)(j0+(t))*k + (size_t) B*JAM_QKK + ebase; \
                        __m256i sb = VDOT(_mm256_setzero_si256(), w0, _mm256_set1_epi32(*(const int*)(ap    ))); \
                        sb = VDOT(sb, w1, _mm256_set1_epi32(*(const int*)(ap +  4))); \
                        sb = VDOT(sb, w2, _mm256_set1_epi32(*(const int*)(ap +  8))); \
                        sb = VDOT(sb, w3, _mm256_set1_epi32(*(const int*)(ap + 12))); \
                        sumi[t] = _mm256_add_epi32(sumi[t], _mm256_mullo_epi32(sb, sc_v)); \
                    } while (0)
                    if (nt == 4) { Q6K_TOK(0); Q6K_TOK(1); Q6K_TOK(2); Q6K_TOK(3); }   /* full tile: fixed indices */
                    else for (int t = 0; t < nt; ++t) Q6K_TOK(t);
                    #undef Q6K_TOK
                }
                __m256 d_v = _mm256_loadu_ps(d);
                __m256 bias[4]; for (int t = 0; t < nt; ++t) bias[t] = _mm256_setzero_ps();
                for (int s16 = 0; s16 < 16; ++s16) {
                    __m256 scf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(sc + s16*8))));
                    for (int t = 0; t < nt; ++t) bias[t] = _mm256_fmadd_ps(scf, _mm256_set1_ps(J->asum[(size_t)(j0+t)*(sblocks*16) + B*16 + s16]), bias[t]);
                }
                for (int t = 0; t < nt; ++t) {
                    __m256 inner = _mm256_sub_ps(_mm256_cvtepi32_ps(sumi[t]), _mm256_mul_ps(_mm256_set1_ps(32.0f), bias[t]));
                    acc[t] = _mm256_fmadd_ps(_mm256_mul_ps(d_v, inner), _mm256_set1_ps(AD[(size_t)(j0+t)*nb + B]), acc[t]);
                }
            }
            for (int t = 0; t < nt; ++t) { float tmp[8]; _mm256_storeu_ps(tmp, acc[t]); for (int f = 0; f < nf; ++f) C[(size_t)(j0+t)*ldc + (i0+f)] = tmp[f]; }
        }
    }
}

void jam_mm_q8_0_rp_avxvnni(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* RP = (const uint8_t*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const size_t grp_bytes = (size_t) nb * sizeof(jam_q8_0_rpblock);
    const int mrows = J->m;
    for (int grp = rb; grp < re; ++grp) {
        int i0 = grp * 8, nf = mrows - i0 < 8 ? mrows - i0 : 8;
        const uint8_t* gbase = RP + (size_t) grp * grp_bytes;
        for (int j0 = 0; j0 < n; j0 += 4) {
            int nt = n - j0 < 4 ? n - j0 : 4;
            __m256 acc[4]; for (int t = 0; t < 4; ++t) acc[t] = _mm256_setzero_ps();
            for (int blk = 0; blk < nb; ++blk) {
                const jam_q8_0_rpblock* rpb = (const jam_q8_0_rpblock*) gbase + blk;
                __m256 d8 = _mm256_loadu_ps(rpb->d);
                __m256i sb[4]; for (int t = 0; t < nt; ++t) sb[t] = _mm256_setzero_si256();
                for (int g = 0; g < 8; ++g) {
                    __m256i w = _mm256_loadu_si256((const __m256i*)(rpb->qs + g*32));
                    for (int t = 0; t < nt; ++t) {
                        __m256i a4 = _mm256_set1_epi32(*(const int*)(AQ + (size_t)(j0+t)*k + (size_t) blk*32 + g*4));
                        sb[t] = VDOT(sb[t], _mm256_abs_epi8(a4), _mm256_sign_epi8(w, a4));
                    }
                }
                for (int t = 0; t < nt; ++t)
                    acc[t] = _mm256_fmadd_ps(_mm256_mul_ps(d8, _mm256_set1_ps(AD[(size_t)(j0+t)*nb + blk])), _mm256_cvtepi32_ps(sb[t]), acc[t]);
            }
            for (int t = 0; t < nt; ++t) { float tmp[8]; _mm256_storeu_ps(tmp, acc[t]); for (int f = 0; f < nf; ++f) C[(size_t)(j0+t)*ldc + (i0+f)] = tmp[f]; }
        }
    }
}
#undef VDOT

/* ================= Q8_0 8-row 256-bit AVX-VNNI repack band — the AVX-512 jam_q8_0_repack_band ported to
 * ymm (8 i32 lanes = 8 rows/group, _mm256_dpbusd_avx_epi32). Same scheme: weight s8 is the signed operand,
 * activation a+128 the unsigned one, +128 bias corrected per row via cw = d·128·Σw. The per-worker repack
 * scratch is byte-for-byte the AVX-512 one (4 groups of 8 == 2 groups of 16). ================= */
#define JAM_Q8B_BYTES 34
#ifndef JAM_VNNI_NR
#define JAM_VNNI_NR 4
#endif
static inline float q8b_h2f(uint16_t h) { return _cvtsh_ss(h); }

static void repack_q8_group8(const uint8_t* wbase, int64_t w_stride, int nb,
                             uint8_t* qs, float* dw, float* cw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += JAM_Q8B_BYTES) {
            float d = q8b_h2f(*(const uint16_t*) w);
            const int8_t* q = (const int8_t*) (w + 2);
            int sumw = 0;
            for (int g = 0; g < 8; g++)
                for (int e = 0; e < 4; e++) {
                    int8_t v = q[g * 4 + e];
                    qs[(int64_t) B * 256 + g * 32 + r * 4 + e] = (uint8_t) v;   /* 8 rows × 4 = 32 B/group */
                    sumw += v;
                }
            dw[(int64_t) B * 8 + r] = d;
            cw[(int64_t) B * 8 + r] = d * 128.0f * (float) sumw;
        }
    }
}

static inline __m256 q8_block8(const uint8_t* qs, const float* dw, const float* cw,
                               const int8_t* x, const float* dx, int nb) {
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++)
            acc = _mm256_dpbusd_avx_epi32(acc, _mm256_set1_epi32(((const int*) x)[g] ^ 0x80808080),
                                          _mm256_load_si256((const void*) (qs + g * 32)));
        __m256 da = _mm256_set1_ps(dx[b]);
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(_mm256_load_ps(dw), da), f);
        f = _mm256_fnmadd_ps(_mm256_load_ps(cw), da, f);
        qs += 256; dw += 8; cw += 8; x += JAM_QK;
    }
    return f;
}

static inline void q8_block8_nr(const uint8_t* qs, const float* dw, const float* cw, const int8_t* xq,
                                const float* dx, int s0, int nb, int64_t ldc, float* out, int r) {
    __m256 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR]; const float* d[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) {
        f[c] = _mm256_setzero_ps();
        x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK;
        d[c] = dx + (int64_t)(s0 + c) * nb;
    }
    for (int b = 0; b < nb; b++) {
        __m256i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {
            __m256i w = _mm256_load_si256((const void*) (qs + g * 32));
            for (int c = 0; c < JAM_VNNI_NR; c++)
                acc[c] = _mm256_dpbusd_avx_epi32(acc[c], _mm256_set1_epi32(((const int*) x[c])[g] ^ 0x80808080), w);
        }
        __m256 dwv = _mm256_load_ps(dw), cwv = _mm256_load_ps(cw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            __m256 da = _mm256_set1_ps(d[c][b]);
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[c]), _mm256_mul_ps(dwv, da), f[c]);
            f[c] = _mm256_fnmadd_ps(cwv, da, f[c]);
            x[c] += JAM_QK;
        }
        qs += 256; dw += 8; cw += 8;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

static float q8_scalar_dot8(const uint8_t* w, int nb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += JAM_Q8B_BYTES, x += 32) {
        float d = q8b_h2f(*(const uint16_t*) w);
        const int8_t* q = (const int8_t*) (w + 2);
        float s = 0.0f;
        for (int e = 0; e < 32; e++) s += (float) q[e] * x[e];
        acc += d * s;
    }
    return acc;
}

void jam_q8_0_repack_band_avxvnni(void* arg, int t0, int t1, int tid) {
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
            repack_q8_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, cw);
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                q8_block8_nr(qs, dw, cw, J->xq, J->dx, s, nb, ldc, J->out, r);
            for (; s < seq; s++)
                _mm256_storeu_ps(J->out + (int64_t) s * ldc + r,
                                 q8_block8(qs, dw, cw, J->xq + (int64_t) s * nb * JAM_QK, J->dx + (int64_t) s * nb, nb));
        }
        for (int r = row + group * 8; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] =
                    q8_scalar_dot8(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride);
    }
}

/* Pure (no-AVX-512) phase-1 activation requant feeding jam_q8_0_repack_band_avxvnni: F32 [seq×k] -> s8 xq +
 * per-32-block scale dx (amax/127). Per-token task (jam_run over seq). Scalar today — correct and portable;
 * AVX2-vectorize the max-abs + convert if it ever shows up in a profile. Q8_0 ignores xsum. */
void jam_q8_0_requant_256(void* arg, int s0, int s1, int tid) {
    (void) tid;
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    int nb = J->kblocks;
    for (int s = s0; s < s1; s++) {
        const float* row = J->rhs + (int64_t) s * J->rhs_stride;
        int8_t* xq = J->xq + (int64_t) s * nb * JAM_QK;
        float*  dx = J->dx + (int64_t) s * nb;
        float*  xs = J->xsum ? J->xsum + (int64_t) s * nb * 2 : 0;
        for (int b = 0; b < nb; b++, row += JAM_QK, xq += JAM_QK) {
            float amax = 0.0f;
            for (int e = 0; e < JAM_QK; e++) { float a = row[e] < 0 ? -row[e] : row[e]; if (a > amax) amax = a; }
            float d  = amax / 127.0f;
            float id = d > 0.0f ? 1.0f / d : 0.0f;
            for (int e = 0; e < JAM_QK; e++) {
                float fv = row[e] * id;
                int v = (int)(fv + (fv >= 0 ? 0.5f : -0.5f));
                xq[e] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
            }
            dx[b] = d;
            if (xs) { int h0=0,h1=0; for (int e=0;e<16;e++) h0+=xq[e]; for (int e=16;e<JAM_QK;e++) h1+=xq[e]; xs[2*b]=(float)h0*d; xs[2*b+1]=(float)h1*d; }   /* SCALED sum (dx·Σxq): Q4_0 -8 term is in activation units */
        }
    }
}

/* ================= Q4_0 8-row 256-bit AVX-VNNI repack band (mirror of jam_q4_0_repack_band). PACKED
 * nibbles (128 B/block for 8 rows); nibble (0..15) is the UNSIGNED vpdpbusd operand, activation the signed
 * one, -8 offset corrected via mw=8·d and the per-16 activation sums (xsum). ================= */
static void repack_q4_0_group8(const uint8_t* wbase, int64_t w_stride, int nb,
                               uint8_t* qs, float* dw, float* mw) {
    for (int r = 0; r < 8; r++) {
        const uint8_t* w = wbase + r * w_stride;
        for (int B = 0; B < nb; B++, w += 18) {                  /* Q4_0 block = 18 bytes */
            float d = q8b_h2f(*(const uint16_t*) w);
            const uint8_t* q = w + 2;
            #define Q40N(idx) ((idx) < 16 ? (q[idx] & 0xF) : (q[(idx) - 16] >> 4))
            for (int i = 0; i < 4; i++)
                for (int e = 0; e < 4; e++) {
                    uint8_t lo = Q40N(i * 8 + e), hi = Q40N(i * 8 + 4 + e);
                    qs[(int64_t) B * 128 + i * 32 + r * 4 + e] = (uint8_t)(lo | (hi << 4));   /* 8 rows × 4 = 32 B/plane */
                }
            #undef Q40N
            dw[(int64_t) B * 8 + r] = d;
            mw[(int64_t) B * 8 + r] = 8.0f * d;
        }
    }
}

static inline __m256 q4_0_block8(const uint8_t* qs, const float* dw, const float* mw,
                                 const int8_t* x, const float* dx, const float* xs, int nb) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 f = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256i acc = _mm256_setzero_si256();
        for (int i = 0; i < 4; i++) {
            __m256i pk = _mm256_load_si256((const void*) (qs + i * 32));
            acc = _mm256_dpbusd_avx_epi32(acc, _mm256_and_si256(pk, m4), _mm256_set1_epi32(((const int*) x)[2 * i]));
            acc = _mm256_dpbusd_avx_epi32(acc, _mm256_and_si256(_mm256_srli_epi16(pk, 4), m4), _mm256_set1_epi32(((const int*) x)[2 * i + 1]));
        }
        __m256 da = _mm256_set1_ps(dx[b]);
        f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(_mm256_load_ps(dw), da), f);
        f = _mm256_fnmadd_ps(_mm256_load_ps(mw), _mm256_set1_ps(xs[2 * b] + xs[2 * b + 1]), f);
        qs += 128; dw += 8; mw += 8; x += JAM_QK;
    }
    return f;
}

static inline void q4_0_block8_nr(const uint8_t* qs, const float* dw, const float* mw, const int8_t* xq,
                                  const float* dx, const float* xs, int s0, int nb, int64_t ldc, float* out, int r) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 f[JAM_VNNI_NR];
    const int8_t* x[JAM_VNNI_NR];
    for (int c = 0; c < JAM_VNNI_NR; c++) { f[c] = _mm256_setzero_ps(); x[c] = xq + (int64_t)(s0 + c) * nb * JAM_QK; }
    for (int b = 0; b < nb; b++) {
        __m256i acc[JAM_VNNI_NR];
        for (int c = 0; c < JAM_VNNI_NR; c++) acc[c] = _mm256_setzero_si256();
        for (int i = 0; i < 4; i++) {
            __m256i pk = _mm256_load_si256((const void*) (qs + i * 32));
            __m256i lo = _mm256_and_si256(pk, m4), hi = _mm256_and_si256(_mm256_srli_epi16(pk, 4), m4);
            for (int c = 0; c < JAM_VNNI_NR; c++) {
                acc[c] = _mm256_dpbusd_avx_epi32(acc[c], lo, _mm256_set1_epi32(((const int*) x[c])[2 * i]));
                acc[c] = _mm256_dpbusd_avx_epi32(acc[c], hi, _mm256_set1_epi32(((const int*) x[c])[2 * i + 1]));
            }
        }
        __m256 dwv = _mm256_load_ps(dw), mwv = _mm256_load_ps(mw);
        for (int c = 0; c < JAM_VNNI_NR; c++) {
            const float* xc = xs + (int64_t)(s0 + c) * nb * 2;
            __m256 da = _mm256_set1_ps(dx[(int64_t)(s0 + c) * nb + b]);
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[c]), _mm256_mul_ps(dwv, da), f[c]);
            f[c] = _mm256_fnmadd_ps(mwv, _mm256_set1_ps(xc[2 * b] + xc[2 * b + 1]), f[c]);
            x[c] += JAM_QK;
        }
        qs += 128; dw += 8; mw += 8;
    }
    for (int c = 0; c < JAM_VNNI_NR; c++) _mm256_storeu_ps(out + (int64_t)(s0 + c) * ldc + r, f[c]);
}

static float q4_0_scalar_dot8(const uint8_t* w, int nb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += 18, x += 32) {
        float d = q8b_h2f(*(const uint16_t*) w); const uint8_t* q = w + 2; float s = 0.0f;
        for (int e = 0; e < 16; e++) { s += (float)((q[e] & 0xF) - 8) * x[e]; s += (float)((q[e] >> 4) - 8) * x[e + 16]; }
        acc += d * s;
    }
    return acc;
}

void jam_q4_0_repack_band_avxvnni(void* arg, int t0, int t1, int tid) {
    const jam_q4k_job* J = (const jam_q4k_job*) arg;
    const int nb = J->kblocks, seq = J->seq; const int64_t ldc = J->out_stride;
    jam_repack* rp = &J->repack[tid];
    for (int tile = t0; tile < t1; tile++) {
        int row = tile * JAM_VNNI_BAND, row_end = row + JAM_VNNI_BAND;
        if (row_end > J->dim0) row_end = J->dim0;
        int group = 0;
        for (int r = row; r + 7 < row_end; r += 8, group++) {
            uint8_t* qs = rp->qs + (int64_t) group * nb * 128;
            float* dw = rp->dw + (int64_t) group * nb * 8;
            float* mw = rp->mw + (int64_t) group * nb * 8;
            repack_q4_0_group8(J->w + (int64_t) r * J->w_stride, J->w_stride, nb, qs, dw, mw);
            int s = 0;
            for (; s + JAM_VNNI_NR <= seq; s += JAM_VNNI_NR)
                q4_0_block8_nr(qs, dw, mw, J->xq, J->dx, J->xsum, s, nb, ldc, J->out, r);
            for (; s < seq; s++)
                _mm256_storeu_ps(J->out + (int64_t) s * ldc + r,
                                 q4_0_block8(qs, dw, mw, J->xq + (int64_t) s * nb * JAM_QK,
                                             J->dx + (int64_t) s * nb, J->xsum + (int64_t) s * nb * 2, nb));
        }
        for (int r = row + group * 8; r < row_end; r++)
            for (int s = 0; s < seq; s++)
                J->out[(int64_t) s * ldc + r] = q4_0_scalar_dot8(J->w + (int64_t) r * J->w_stride, nb, J->rhs + (int64_t) s * J->rhs_stride);
    }
}
