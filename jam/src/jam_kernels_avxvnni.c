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
#include "jam_decode_x86_256.h"

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
    const int mrows = ldc;
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
    const int mrows = ldc;
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
    const int mrows = ldc;
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
                    __m256i sb[4]; for (int t = 0; t < nt; ++t) sb[t] = _mm256_setzero_si256();
                    for (int g4 = 0; g4 < 4; ++g4) {
                        __m256i w = _mm256_loadu_si256((const __m256i*)(qs + (size_t)(s16*4 + g4)*32));
                        for (int t = 0; t < nt; ++t)
                            sb[t] = VDOT(sb[t], w, _mm256_set1_epi32(*(const int*)(AQ + (size_t)(j0+t)*k + (size_t) B*JAM_QKK + ebase + g4*4)));
                    }
                    for (int t = 0; t < nt; ++t) sumi[t] = _mm256_add_epi32(sumi[t], _mm256_mullo_epi32(sb[t], sc_v));
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
    const int mrows = ldc;
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
