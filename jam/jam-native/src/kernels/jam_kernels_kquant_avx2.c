/* AVX2 K-quant + Q8_0 cached-repack kernels (@ F32 -> F32) — the int8 path for x86 BELOW AVX-512-VNNI, so
 * these weights don't hit the generic float floor on the (very common) avx2-only / avx512-without-VNNI CPUs.
 * The AVX-512-VNNI repack (jam_kernels_q4k_avx512.c) remains the fast path above this. Built -mavx2 -mfma
 * -mf16c. Each kernel is 8-feature-WIDE (8 output features in the 8 int32 lanes, so no per-feature hsum) over
 * a cached weight repack; activations are pre-requantized to int8 (J->aq) + per-32 scales (J->ad), with the
 * dmin·min / Q6_K-(-32) / Q8_0 terms corrected in float via Σx ≈ ad·Σaq. */
#include <immintrin.h>
#include "jam_internal.h"
#include "jam_kquant.h"

/* ---- Q4_K CACHED-REPACK avx2 (8-feature-WIDE, hand-tuned maddubs). Weight repacked ONCE (cached by ctx,
 * keyed on the weight ptr); the GEMM puts 8 features in the 8 int32 lanes (one feature/lane) so the dot
 * reduces IN the lanes — no per-feature hsum. Activation broadcasts across the 8 features; weight reuses
 * across 4 tokens. Needs the per-256 jam_q8k_requant. ---- Repacked per super-block (1216 B): f32 d[8],
 * f32 dmin[8], u8 sc[8sub][8feat], u8 mn[8sub][8feat], u8 qs[8sub][8group][16]: each 16-byte group packs
 * 8 rows × 4 elements as [lo16=rows0..3 | hi16=rows4..7], byte b = nib(row b>>2, elem g*4+(b&3)). */
void jam_q4k_repack8(const void* Wv, int rows0, int re, int sblocks, size_t w_stride, void* outv) {
    const uint8_t* W = (const uint8_t*) Wv; uint8_t* out = (uint8_t*) outv;
    int nf = re - rows0 < 8 ? re - rows0 : 8;
    for (int B = 0; B < sblocks; ++B) {
        jam_q4k_rpblock* blk = (jam_q4k_rpblock*) out + B;
        float* d = blk->d; float* dmin = blk->dmin; uint8_t* sc = blk->sc; uint8_t* mn = blk->mn; uint8_t* qs = blk->qs;
        const uint8_t* qrow[8];
        for (int f = 0; f < 8; ++f) {
            if (f >= nf) { d[f] = dmin[f] = 0.0f; for (int s = 0; s < 8; ++s) { sc[s*8+f] = mn[s*8+f] = 0; } qrow[f] = 0; continue; }
            const uint8_t* wb = W + (size_t)(rows0+f)*w_stride + (size_t) B*JAM_Q4K_BYTES;
            d[f] = _cvtsh_ss(*(const uint16_t*) wb); dmin[f] = _cvtsh_ss(*(const uint16_t*)(wb+2));
            uint8_t scf[8], mnf[8]; jam_q4k_scales_mins(wb + 4, scf, mnf);
            for (int s = 0; s < 8; ++s) { sc[s*8+f] = scf[s]; mn[s*8+f] = mnf[s]; }
            qrow[f] = wb + 16;
        }
        for (int s = 0; s < 8; ++s) {
            const uint8_t* qp[8]; for (int f = 0; f < 8; ++f) qp[f] = qrow[f] ? qrow[f] + (s/2)*32 : 0;
            int odd = s & 1;
            for (int gp = 0; gp < 8; ++gp) {
                uint8_t* dst = qs + (size_t)(s*8 + gp)*16;
                for (int b = 0; b < 16; ++b) {
                    int rl = b >> 2, e = b & 3, elem = gp*4 + e, rh = rl + 4;
                    int lo = qp[rl] ? (odd ? (qp[rl][elem] >> 4) : (qp[rl][elem] & 0xF)) : 0;
                    int hi = qp[rh] ? (odd ? (qp[rh][elem] >> 4) : (qp[rh][elem] & 0xF)) : 0;
                    dst[b] = (uint8_t)(lo | (hi << 4));
                }
            }
        }
    }
}

/* The repacked GEMM: rb..re = 8-feature GROUP indices; J->a = the cached repacked weight. */
void jam_mm_q4k_rp_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* RP = (const uint8_t*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t grp_bytes = (size_t) sblocks * sizeof(jam_q4k_rpblock);
    const int mrows = J->m;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    for (int grp = rb; grp < re; ++grp) {
        int i0 = grp * 8;
        int nf = mrows - i0 < 8 ? mrows - i0 : 8;
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
                    /* sc16 = [sc0,sc0,sc1,sc1,...,sc7,sc7] so madd_epi16 folds the per-sub-block scale into the dot (no vpmulld) */
                    __m128i sc8b = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(sc + s*8)));
                    __m256i sc16 = _mm256_set_m128i(_mm_unpackhi_epi16(sc8b, sc8b), _mm_unpacklo_epi16(sc8b, sc8b));
                    __m256i sb[4]; for (int t = 0; t < nt; ++t) sb[t] = _mm256_setzero_si256();
                    for (int gp = 0; gp < 8; ++gp) {
                        __m128i x = _mm_loadu_si128((const __m128i*)(qs + (size_t)(s*8 + gp)*16));
                        __m256i w = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(x, 4), m4), _mm_and_si128(x, m4));
                        for (int t = 0; t < nt; ++t) {
                            const int8_t* a = AQ + (size_t)(j0+t)*k + (size_t) B*JAM_QKK + s*32 + gp*4;
                            __m256i aw = _mm256_set1_epi32(*(const int*) a);
                            sb[t] = _mm256_add_epi32(sb[t], _mm256_madd_epi16(_mm256_maddubs_epi16(w, aw), sc16));
                        }
                    }
                    for (int t = 0; t < nt; ++t) sumi[t] = _mm256_add_epi32(sumi[t], sb[t]);
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
            for (int t = 0; t < nt; ++t) {
                float tmp[8]; _mm256_storeu_ps(tmp, acc[t]);
                for (int f = 0; f < nf; ++f) C[(size_t)(j0+t)*ldc + (i0+f)] = tmp[f];
            }
        }
    }
}

/* ---- Q5_K cached-repack (8-feature-wide). Same as the Q4_K rp kernel, but the repack stores the FULL 5-bit
 * value (nibble | qh-bit<<4, 0..31) one byte per element, so the gemm loads w directly — no in-register decode
 * (qs is 2x Q4_K). All scale/min handling is identical to Q4_K. ---- */
void jam_q5k_repack8(const void* Wv, int rows0, int re, int sblocks, size_t w_stride, void* outv) {
    const uint8_t* W = (const uint8_t*) Wv; jam_q5k_rpblock* out = (jam_q5k_rpblock*) outv;
    int nf = re - rows0 < 8 ? re - rows0 : 8;
    for (int B = 0; B < sblocks; ++B) {
        jam_q5k_rpblock* blk = out + B;
        const uint8_t *qhrow[8], *qsrow[8];
        for (int f = 0; f < 8; ++f) {
            if (f >= nf) { blk->d[f] = blk->dmin[f] = 0.0f; for (int s = 0; s < 8; ++s) { blk->sc[s*8+f] = blk->mn[s*8+f] = 0; } qhrow[f] = qsrow[f] = 0; continue; }
            const uint8_t* wb = W + (size_t)(rows0+f)*w_stride + (size_t) B*JAM_Q5K_BYTES;
            blk->d[f] = _cvtsh_ss(*(const uint16_t*) wb); blk->dmin[f] = _cvtsh_ss(*(const uint16_t*)(wb+2));
            uint8_t scf[8], mnf[8]; jam_q4k_scales_mins(wb + 4, scf, mnf);
            for (int s = 0; s < 8; ++s) { blk->sc[s*8+f] = scf[s]; blk->mn[s*8+f] = mnf[s]; }
            qhrow[f] = wb + 16; qsrow[f] = wb + 48;
        }
        for (int s = 0; s < 8; ++s) {
            int odd = s & 1;
            for (int gp = 0; gp < 8; ++gp) {
                uint8_t* dst = blk->qs + (size_t)(s*8 + gp)*32;
                for (int f = 0; f < 8; ++f)
                    for (int e = 0; e < 4; ++e) {
                        int p = gp*4 + e, v = 0;
                        if (qsrow[f]) {
                            int nib = odd ? (qsrow[f][(s/2)*32 + p] >> 4) : (qsrow[f][(s/2)*32 + p] & 0xF);
                            v = nib | (((qhrow[f][p] >> s) & 1) << 4);
                        }
                        dst[f*4 + e] = (uint8_t) v;
                    }
            }
        }
    }
}

void jam_mm_q5k_rp_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* RP = (const uint8_t*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t grp_bytes = (size_t) sblocks * sizeof(jam_q5k_rpblock);
    const int mrows = J->m;
    for (int grp = rb; grp < re; ++grp) {
        int i0 = grp * 8;
        int nf = mrows - i0 < 8 ? mrows - i0 : 8;
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
                    __m128i sc8b = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(sc + s*8)));
                    __m256i sc16 = _mm256_set_m128i(_mm_unpackhi_epi16(sc8b, sc8b), _mm_unpacklo_epi16(sc8b, sc8b));
                    __m256i sb[4]; for (int t = 0; t < nt; ++t) sb[t] = _mm256_setzero_si256();
                    for (int gp = 0; gp < 8; ++gp) {
                        __m256i w = _mm256_loadu_si256((const __m256i*)(qs + (size_t)(s*8 + gp)*32));  /* 5-bit vals, direct */
                        for (int t = 0; t < nt; ++t) {
                            const int8_t* a = AQ + (size_t)(j0+t)*k + (size_t) B*JAM_QKK + s*32 + gp*4;
                            __m256i aw = _mm256_set1_epi32(*(const int*) a);
                            sb[t] = _mm256_add_epi32(sb[t], _mm256_madd_epi16(_mm256_maddubs_epi16(w, aw), sc16));
                        }
                    }
                    for (int t = 0; t < nt; ++t) sumi[t] = _mm256_add_epi32(sumi[t], sb[t]);
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
            for (int t = 0; t < nt; ++t) {
                float tmp[8]; _mm256_storeu_ps(tmp, acc[t]);
                for (int f = 0; f < nf; ++f) C[(size_t)(j0+t)*ldc + (i0+f)] = tmp[f];
            }
        }
    }
}

/* ---- Q6_K cached-repack (8-feature-wide). Q6_K has 16 per-16 INT8 (signed) scales, a signed 6-bit weight
 * (qv-32) and NO min. Store qv UNSIGNED (0..63, one byte/elem, group-interleaved) and fold the -32 as a
 * 32·Σaq bias (per-16 sums computed in-kernel), mirroring Q4_K's min term. ---- Repacked per super-block:
 * f32 d[8], i8 sc[16sub][8feat], u8 qs[16sub][4grp][32] (qv); sub16 s -> chunk (s/8)*4+(s%8)/2, half s&1. */
void jam_q6k_repack8(const void* Wv, int rows0, int re, int sblocks, size_t w_stride, void* outv) {
    const uint8_t* W = (const uint8_t*) Wv; jam_q6k_rpblock* out = (jam_q6k_rpblock*) outv;
    int nf = re - rows0 < 8 ? re - rows0 : 8;
    for (int B = 0; B < sblocks; ++B) {
        jam_q6k_rpblock* blk = out + B;
        const uint8_t* qlrow[8]; const uint8_t* qhrow[8];
        for (int f = 0; f < 8; ++f) {
            if (f >= nf) { blk->d[f] = 0.0f; for (int s = 0; s < 16; ++s) blk->sc[s*8+f] = 0; qlrow[f] = qhrow[f] = 0; continue; }
            const uint8_t* wb = W + (size_t)(rows0+f)*w_stride + (size_t) B*JAM_Q6K_BYTES;
            blk->d[f] = _cvtsh_ss(*(const uint16_t*)(wb + 208));
            const int8_t* scr = (const int8_t*)(wb + 192);
            for (int s = 0; s < 16; ++s) blk->sc[s*8+f] = scr[s];
            qlrow[f] = wb; qhrow[f] = wb + 128;
        }
        for (int s16 = 0; s16 < 16; ++s16) {
            int h = s16 / 8, rem = s16 % 8, g = rem / 2, half = rem & 1;
            for (int g4 = 0; g4 < 4; ++g4) {
                uint8_t* dst = blk->qs + (size_t)(s16*4 + g4)*32;
                for (int f = 0; f < 8; ++f)
                    for (int e = 0; e < 4; ++e) {
                        int lane = half*16 + g4*4 + e, v = 0;
                        if (qlrow[f]) {
                            int lq = qlrow[f][h*64 + (g&1)*32 + lane];
                            int nib = (g < 2) ? (lq & 0xF) : ((lq >> 4) & 0xF);
                            int hib = (qhrow[f][h*32 + lane] >> (2*g)) & 3;
                            v = nib | (hib << 4);                /* qv unsigned 0..63 */
                        }
                        dst[f*4 + e] = (uint8_t) v;
                    }
            }
        }
    }
}

void jam_mm_q6k_rp_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* RP = (const uint8_t*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t grp_bytes = (size_t) sblocks * sizeof(jam_q6k_rpblock);
    const int mrows = J->m;
    for (int grp = rb; grp < re; ++grp) {
        int i0 = grp * 8;
        int nf = mrows - i0 < 8 ? mrows - i0 : 8;
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
                    __m128i scb = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(sc + s16*8)));   /* signed scale */
                    __m256i sc16 = _mm256_set_m128i(_mm_unpackhi_epi16(scb, scb), _mm_unpacklo_epi16(scb, scb));
                    __m256i sb[4]; for (int t = 0; t < nt; ++t) sb[t] = _mm256_setzero_si256();
                    for (int g4 = 0; g4 < 4; ++g4) {
                        __m256i w = _mm256_loadu_si256((const __m256i*)(qs + (size_t)(s16*4 + g4)*32));
                        for (int t = 0; t < nt; ++t) {
                            int ai = *(const int*)(AQ + (size_t)(j0+t)*k + (size_t) B*JAM_QKK + ebase + g4*4);
                            sb[t] = _mm256_add_epi32(sb[t], _mm256_madd_epi16(_mm256_maddubs_epi16(w, _mm256_set1_epi32(ai)), sc16));
                        }
                    }
                    for (int t = 0; t < nt; ++t) sumi[t] = _mm256_add_epi32(sumi[t], sb[t]);
                }
                __m256 d_v = _mm256_loadu_ps(d);
                __m256 bias[4]; for (int t = 0; t < nt; ++t) bias[t] = _mm256_setzero_ps();
                for (int s16 = 0; s16 < 16; ++s16) {       /* load each scale once, fan out over tokens; per-16 Σaq from requant */
                    __m256 scf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(sc + s16*8))));
                    for (int t = 0; t < nt; ++t) bias[t] = _mm256_fmadd_ps(scf, _mm256_set1_ps(J->asum[(size_t)(j0+t)*(sblocks*16) + B*16 + s16]), bias[t]);
                }
                for (int t = 0; t < nt; ++t) {
                    __m256 inner = _mm256_sub_ps(_mm256_cvtepi32_ps(sumi[t]), _mm256_mul_ps(_mm256_set1_ps(32.0f), bias[t]));
                    acc[t] = _mm256_fmadd_ps(_mm256_mul_ps(d_v, inner), _mm256_set1_ps(AD[(size_t)(j0+t)*nb + B]), acc[t]);
                }
            }
            for (int t = 0; t < nt; ++t) {
                float tmp[8]; _mm256_storeu_ps(tmp, acc[t]);
                for (int f = 0; f < nf; ++f) C[(size_t)(j0+t)*ldc + (i0+f)] = tmp[f];
            }
        }
    }
}

/* ---- Q8_0 cached-repack (8-feature-wide, sign-trick maddubs). Q8_0 weight is int8 (signed), f16 per-32-block
 * scale, no min. maddubs needs a u8 operand, so put |a| (<=127, no int16 overflow) there and the sign on the
 * weight via sign_epi8(w,a): maddubs(|a|, w*sign(a)) = Σ a*w. Repacked per 32-block: f32 d[8], int8 qs[8grp][8feat*4]. */
void jam_q8_0_repack8(const void* Wv, int rows0, int re, int nblocks, size_t w_stride, void* outv) {
    const uint8_t* W = (const uint8_t*) Wv; jam_q8_0_rpblock* out = (jam_q8_0_rpblock*) outv;
    int nf = re - rows0 < 8 ? re - rows0 : 8;
    for (int blk = 0; blk < nblocks; ++blk) {
        jam_q8_0_rpblock* rp = out + blk;
        for (int f = 0; f < 8; ++f) {
            if (f >= nf) { rp->d[f] = 0.0f; for (int g = 0; g < 8; ++g) for (int e = 0; e < 4; ++e) rp->qs[g*32 + f*4 + e] = 0; continue; }
            const uint8_t* wb = W + (size_t)(rows0+f)*w_stride + (size_t) blk*34;
            rp->d[f] = _cvtsh_ss(*(const uint16_t*) wb);
            const int8_t* q = (const int8_t*)(wb + 2);
            /* The downstream sign-trick (abs(a)·sign(w,a) via maddubs/vpdpbusd) needs weights in [-127,127]:
             * _mm256_sign_epi8(-128, a<0) computes -(-128)=+128 which overflows int8 back to -128, flipping
             * that term's sign. Real GGUF Q8_0 quantizes into [-127,127]; clamp here to stay correct on any
             * input (1 LSB on the rare -128). */
            for (int g = 0; g < 8; ++g) for (int e = 0; e < 4; ++e) {
                int8_t v = q[g*4 + e]; rp->qs[g*32 + f*4 + e] = v < -127 ? -127 : v;
            }
        }
    }
}

/* ---- Q4_0 cached-repack (8-feature-wide). Q4_0 weight is a 4-bit nibble (0..15) with a fixed -8 offset and
 * an f16 per-32-block scale (18-byte block, no min). Dequant each nibble to the SIGNED value (q-8) ∈ [-8,7]
 * here, so the downstream kernel is the bias-free Q8_0 sign-trick maddubs gemm verbatim (-8 folded into the
 * weight, no separate min/bias term). Standard ggml nibble order: low nibble of byte j = elem j, high = j+16. */
void jam_q4_0_repack8(const void* Wv, int rows0, int re, int nblocks, size_t w_stride, void* outv) {
    const uint8_t* W = (const uint8_t*) Wv; jam_q8_0_rpblock* out = (jam_q8_0_rpblock*) outv;
    int nf = re - rows0 < 8 ? re - rows0 : 8;
    for (int blk = 0; blk < nblocks; ++blk) {
        jam_q8_0_rpblock* rp = out + blk;
        for (int f = 0; f < 8; ++f) {
            if (f >= nf) { rp->d[f] = 0.0f; for (int g = 0; g < 8; ++g) for (int e = 0; e < 4; ++e) rp->qs[g*32 + f*4 + e] = 0; continue; }
            const uint8_t* wb = W + (size_t)(rows0+f)*w_stride + (size_t) blk*18;
            rp->d[f] = _cvtsh_ss(*(const uint16_t*) wb);
            const uint8_t* q = wb + 2;
            for (int g = 0; g < 8; ++g) for (int e = 0; e < 4; ++e) {
                int idx = g*4 + e;                                    /* natural element index 0..31 within the block */
                int v = idx < 16 ? (q[idx] & 0x0F) : (q[idx-16] >> 4);
                rp->qs[g*32 + f*4 + e] = (int8_t)(v - 8);             /* fold the -8 offset: range [-8,7] */
            }
        }
    }
}

void jam_mm_q8_0_rp_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* RP = (const uint8_t*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;   /* nb = k/32 */
    const size_t grp_bytes = (size_t) nb * sizeof(jam_q8_0_rpblock);
    const int mrows = J->m;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int grp = rb; grp < re; ++grp) {
        int i0 = grp * 8;
        int nf = mrows - i0 < 8 ? mrows - i0 : 8;
        const uint8_t* gbase = RP + (size_t) grp * grp_bytes;
        for (int j0 = 0; j0 < n; j0 += 4) {
            int nt = n - j0 < 4 ? n - j0 : 4;
            __m256 acc[4]; for (int t = 0; t < 4; ++t) acc[t] = _mm256_setzero_ps();
            for (int blk = 0; blk < nb; ++blk) {
                const jam_q8_0_rpblock* rpb = (const jam_q8_0_rpblock*) gbase + blk;
                __m256 d8 = _mm256_loadu_ps(rpb->d);
                __m256i sb[4]; for (int t = 0; t < nt; ++t) sb[t] = _mm256_setzero_si256();
                for (int g = 0; g < 8; ++g) {
                    __m256i w = _mm256_loadu_si256((const __m256i*)(rpb->qs + g*32));   /* 8 feat × 4 elem, signed */
                    for (int t = 0; t < nt; ++t) {
                        __m256i a4 = _mm256_set1_epi32(*(const int*)(AQ + (size_t)(j0+t)*k + (size_t) blk*32 + g*4));
                        __m256i prod = _mm256_maddubs_epi16(_mm256_abs_epi8(a4), _mm256_sign_epi8(w, a4));
                        sb[t] = _mm256_add_epi32(sb[t], _mm256_madd_epi16(prod, ones));
                    }
                }
                for (int t = 0; t < nt; ++t) {
                    __m256 dad = _mm256_mul_ps(d8, _mm256_set1_ps(AD[(size_t)(j0+t)*nb + blk]));
                    acc[t] = _mm256_fmadd_ps(dad, _mm256_cvtepi32_ps(sb[t]), acc[t]);
                }
            }
            for (int t = 0; t < nt; ++t) {
                float tmp[8]; _mm256_storeu_ps(tmp, acc[t]);
                for (int f = 0; f < nf; ++f) C[(size_t)(j0+t)*ldc + (i0+f)] = tmp[f];
            }
        }
    }
}

