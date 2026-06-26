/* AVX2 K-quant kernels (Q4_K/Q5_K/Q6_K @ F32 -> F32) — the int8 path for x86 BELOW AVX-512-VNNI, so K-quant
 * weights stop hitting the generic float floor on the (very common) avx2-only / avx512-without-VNNI CPUs.
 * The AVX-512-VNNI repack (jam_kernels_q4k_avx512.c) remains the fast path above this. Built -mavx2 -mfma
 * -mf16c. Reuses the avx2 int8 dot (maddubs+madd, KDOT below) + jam_hsum8_256. Activations are pre-
 * requantized to int8 (J->aq) + per-32 scales (J->ad); the dmin·min / Q6_K-(-32) terms are corrected in
 * float via Σx ≈ ad·Σaq. Decode mirrors jam_kernels_generic.c bit-for-bit. */
#include <immintrin.h>
#include "jam_internal.h"
#include "jam_kquant.h"
#include "jam_decode_x86_256.h"   /* jam_hsum8_256 */

/* signed int8 dot of an unsigned (0..255) operand u and a signed operand s, as 8 floats (lanes 0-3 cover
 * elems 0..15, lanes 4-7 cover 16..31). For signed weights, pass abs(w) and sign(a,w). */
#define KDOT(u, s) _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)))

/* Q4_K: value = d·sc·nibble - dmin·mn. 8 sub-blocks/super-block; sub-block s -> elems s*32, nibbles
 * q[(s/2)*32+e] (low if s even, else high). Weights 0..15 (non-negative): KDOT direct, no abs/sign. */
/* Columns processed per weight decode in the avx2 K-quant kernels. acc[] are scalars and the weight
 * (qb/wq) is the only shared ymm, so NR isn't register-bound here; 4 amortizes the qb load + nibble
 * decode + per-super-block scale unpack (jam_q4k_scales_mins, h2f) across the columns of one weight row. */
#define JAM_KQ_AVX2_NR 4

void jam_mm_q4k_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q4K_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        int j = 0;
        for (; j + JAM_KQ_AVX2_NR <= n; j += JAM_KQ_AVX2_NR) {                 /* NR columns per weight decode */
            __m256 dotv[JAM_KQ_AVX2_NR]; float macc[JAM_KQ_AVX2_NR];          /* deferred-hsum dot accum + scalar min accum */
            const int8_t* aqc[JAM_KQ_AVX2_NR]; const float* adc[JAM_KQ_AVX2_NR]; const float* asc[JAM_KQ_AVX2_NR];
            for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) {
                dotv[c] = _mm256_setzero_ps(); macc[c] = 0.0f;
                aqc[c] = AQ + (size_t)(j+c)*k; adc[c] = AD + (size_t)(j+c)*nb; asc[c] = J->asum + (size_t)(j+c)*nb;
            }
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q4K_BYTES) {
                float d = _cvtsh_ss(*(const uint16_t*) w), dmin = _cvtsh_ss(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);       /* unpacked ONCE per super-block */
                const uint8_t* q = w + 16;
                for (int s = 0; s < 8; ++s) {
                    __m256i qb = _mm256_loadu_si256((const __m256i*) (q + (s/2)*32));
                    __m256i wq = (s & 1) ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4)
                                         : _mm256_and_si256(qb, m4);             /* decoded ONCE, reused NR cols */
                    int blk = B*8 + s;
                    float dsc = d*sc[s], dmn = dmin*mn[s];
                    for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) {
                        __m256i av = _mm256_loadu_si256((const __m256i*) (aqc[c] + (size_t) blk*32));
                        dotv[c] = _mm256_fmadd_ps(KDOT(wq, av), _mm256_set1_ps(adc[c][blk]*dsc), dotv[c]);   /* vec accum */
                        macc[c] += adc[c][blk] * dmn * asc[c][blk];
                    }
                }
            }
            for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) C[(size_t)(j+c)*ldc + i] = jam_hsum8_256(dotv[c]) - macc[c];  /* ONE hsum/col */
        }
        for (; j < n; ++j) {                                                    /* column tail (< NR) */
            const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
            const float* as = J->asum + (size_t) j * nb;
            const uint8_t* w = wrow; float acc = 0.0f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q4K_BYTES) {
                float d = _cvtsh_ss(*(const uint16_t*) w), dmin = _cvtsh_ss(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* q = w + 16;
                for (int s = 0; s < 8; ++s) {
                    __m256i qb = _mm256_loadu_si256((const __m256i*) (q + (s/2)*32));
                    __m256i wq = (s & 1) ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4)
                                         : _mm256_and_si256(qb, m4);
                    int blk = B*8 + s;
                    __m256i av = _mm256_loadu_si256((const __m256i*) (aq + (size_t) blk*32));
                    float dot = jam_hsum8_256(KDOT(wq, av));
                    acc += ad[blk] * (d*sc[s]*dot - dmin*mn[s]*as[blk]);
                }
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}

/* Q4_K int-scale variant (needs the per-256 jam_q8k_requant: one activation scale ad[B] per super-block).
 * llama.cpp's scheme: apply the 6-bit weight scale in INT (madd_epi16), accumulate int32 across the 8
 * sub-blocks, convert to float ONCE per super-block (not per sub-block). Keeps jam's NR column tiling on
 * top. C = Σ_B ad[B]·(d·Σ_s sc[s]·intdot - dmin·Σ_s mn[s]·Σaq). */
void jam_mm_q4k_avx2_is(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q4K_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        int j = 0;
        for (; j + JAM_KQ_AVX2_NR <= n; j += JAM_KQ_AVX2_NR) {
            __m256 accv[JAM_KQ_AVX2_NR];
            const int8_t* aqc[JAM_KQ_AVX2_NR]; const float* adc[JAM_KQ_AVX2_NR]; const float* asc[JAM_KQ_AVX2_NR];
            for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) {
                accv[c] = _mm256_setzero_ps();
                aqc[c] = AQ + (size_t)(j+c)*k; adc[c] = AD + (size_t)(j+c)*nb; asc[c] = J->asum + (size_t)(j+c)*nb;
            }
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q4K_BYTES) {
                float d = _cvtsh_ss(*(const uint16_t*) w), dmin = _cvtsh_ss(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                __m256 mnv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) mn)));  /* 8 mn -> f32 */
                const uint8_t* q = w + 16;
                __m256i sumi[JAM_KQ_AVX2_NR];
                for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) sumi[c] = _mm256_setzero_si256();
                for (int s = 0; s < 8; ++s) {
                    __m256i qb = _mm256_loadu_si256((const __m256i*) (q + (s/2)*32));
                    __m256i wq = (s & 1) ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4)
                                         : _mm256_and_si256(qb, m4);             /* decode ONCE */
                    __m256i scv = _mm256_set1_epi16((short) sc[s]);
                    int blk = B*8 + s;
                    for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) {
                        __m256i av = _mm256_loadu_si256((const __m256i*) (aqc[c] + (size_t) blk*32));
                        sumi[c] = _mm256_add_epi32(sumi[c], _mm256_madd_epi16(scv, _mm256_maddubs_epi16(wq, av)));
                    }
                }
                for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) {
                    float ad = adc[c][B];                            /* per-super-block activation scale */
                    accv[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[c]), _mm256_set1_ps(d * ad), accv[c]);
                    __m256 asv = _mm256_loadu_ps(asc[c] + (size_t) B*8);    /* min term: vectorized + folded into accv */
                    accv[c] = _mm256_fnmadd_ps(_mm256_mul_ps(mnv, asv), _mm256_set1_ps(dmin * ad), accv[c]);
                }
            }
            for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) C[(size_t)(j+c)*ldc + i] = jam_hsum8_256(accv[c]);
        }
        for (; j < n; ++j) {                                                    /* column tail (< NR) */
            const int8_t* aq = AQ + (size_t) j * k; const float* ad256 = AD + (size_t) j * nb;
            const float* as = J->asum + (size_t) j * nb;
            const uint8_t* w = wrow; __m256 accv = _mm256_setzero_ps();
            for (int B = 0; B < sblocks; ++B, w += JAM_Q4K_BYTES) {
                float d = _cvtsh_ss(*(const uint16_t*) w), dmin = _cvtsh_ss(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                __m256 mnv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) mn)));
                const uint8_t* q = w + 16;
                __m256i sumi = _mm256_setzero_si256();
                for (int s = 0; s < 8; ++s) {
                    __m256i qb = _mm256_loadu_si256((const __m256i*) (q + (s/2)*32));
                    __m256i wq = (s & 1) ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4)
                                         : _mm256_and_si256(qb, m4);
                    int blk = B*8 + s;
                    __m256i av = _mm256_loadu_si256((const __m256i*) (aq + (size_t) blk*32));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_set1_epi16((short) sc[s]), _mm256_maddubs_epi16(wq, av)));
                }
                float ad = ad256[B];
                accv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi), _mm256_set1_ps(d * ad), accv);
                __m256 asv = _mm256_loadu_ps(as + (size_t) B*8);
                accv = _mm256_fnmadd_ps(_mm256_mul_ps(mnv, asv), _mm256_set1_ps(dmin * ad), accv);
            }
            C[(size_t) j*ldc + i] = jam_hsum8_256(accv);
        }
    }
}

/* ---- Q4_K CACHED-REPACK avx2 (8-feature-WIDE, hand-tuned maddubs). Weight repacked ONCE (cached by ctx,
 * keyed on the weight ptr); the GEMM puts 8 features in the 8 int32 lanes (one feature/lane) so the dot
 * reduces IN the lanes — no per-feature hsum. Activation broadcasts across the 8 features; weight reuses
 * across 4 tokens. Needs the per-256 jam_q8k_requant. ---- Repacked per super-block (1216 B): f32 d[8],
 * f32 dmin[8], u8 sc[8sub][8feat], u8 mn[8sub][8feat], u8 qs[8sub][8group][16]: each 16-byte group packs
 * 8 rows × 4 elements as [lo16=rows0..3 | hi16=rows4..7], byte b = nib(row b>>2, elem g*4+(b&3)). */
void jam_q4k_repack8(const void* Wv, int rows0, int re, int sblocks, void* outv) {
    const uint8_t* W = (const uint8_t*) Wv; uint8_t* out = (uint8_t*) outv;
    const size_t w_stride = (size_t) sblocks * JAM_Q4K_BYTES;
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
    const int mrows = ldc;
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
                for (int t = 0; t < nt; ++t) {
                    __m256 minsum = _mm256_setzero_ps();
                    for (int s = 0; s < 8; ++s) {
                        __m256 mnf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(mn + s*8))));
                        minsum = _mm256_fmadd_ps(mnf, _mm256_set1_ps(J->asum[(size_t)(j0+t)*nb + B*8 + s]), minsum);
                    }
                    __m256 contrib = _mm256_sub_ps(_mm256_mul_ps(d_v, _mm256_cvtepi32_ps(sumi[t])), _mm256_mul_ps(dmin_v, minsum));
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
void jam_q5k_repack8(const void* Wv, int rows0, int re, int sblocks, void* outv) {
    const uint8_t* W = (const uint8_t*) Wv; jam_q5k_rpblock* out = (jam_q5k_rpblock*) outv;
    const size_t w_stride = (size_t) sblocks * JAM_Q5K_BYTES;
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
    const int mrows = ldc;
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
                for (int t = 0; t < nt; ++t) {
                    __m256 minsum = _mm256_setzero_ps();
                    for (int s = 0; s < 8; ++s) {
                        __m256 mnf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(mn + s*8))));
                        minsum = _mm256_fmadd_ps(mnf, _mm256_set1_ps(J->asum[(size_t)(j0+t)*nb + B*8 + s]), minsum);
                    }
                    __m256 contrib = _mm256_sub_ps(_mm256_mul_ps(d_v, _mm256_cvtepi32_ps(sumi[t])), _mm256_mul_ps(dmin_v, minsum));
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

/* Q5_K: like Q4_K + a 5th weight bit from qh (q5 = nibble | (bit s of qh[e] << 4), 0..31). */
void jam_mm_q5k_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q5K_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        int j = 0;
        for (; j + JAM_KQ_AVX2_NR <= n; j += JAM_KQ_AVX2_NR) {                 /* NR cols/decode + deferred hsum */
            __m256 dotv[JAM_KQ_AVX2_NR]; float macc[JAM_KQ_AVX2_NR];
            const int8_t* aqc[JAM_KQ_AVX2_NR]; const float* adc[JAM_KQ_AVX2_NR]; const float* asc[JAM_KQ_AVX2_NR];
            for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) {
                dotv[c] = _mm256_setzero_ps(); macc[c] = 0.0f;
                aqc[c] = AQ + (size_t)(j+c)*k; adc[c] = AD + (size_t)(j+c)*nb; asc[c] = J->asum + (size_t)(j+c)*nb;
            }
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q5K_BYTES) {
                float d = _cvtsh_ss(*(const uint16_t*) w), dmin = _cvtsh_ss(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* qh = w + 16; const uint8_t* qs = w + 48;
                __m256i h = _mm256_loadu_si256((const __m256i*) qh);
                for (int s = 0; s < 8; ++s) {
                    __m256i qb = _mm256_loadu_si256((const __m256i*) (qs + (s/2)*32));
                    __m256i nib = (s & 1) ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4)
                                          : _mm256_and_si256(qb, m4);
                    __m256i hb = _mm256_and_si256(_mm256_srl_epi16(h, _mm_cvtsi32_si128(s)), one);
                    __m256i wq = _mm256_or_si256(nib, _mm256_slli_epi16(hb, 4));   /* decode ONCE */
                    int blk = B*8 + s; float dsc = d*sc[s], dmn = dmin*mn[s];
                    for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) {
                        __m256i av = _mm256_loadu_si256((const __m256i*) (aqc[c] + (size_t) blk*32));
                        dotv[c] = _mm256_fmadd_ps(KDOT(wq, av), _mm256_set1_ps(adc[c][blk]*dsc), dotv[c]);
                        macc[c] += adc[c][blk] * dmn * asc[c][blk];
                    }
                }
            }
            for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) C[(size_t)(j+c)*ldc + i] = jam_hsum8_256(dotv[c]) - macc[c];
        }
        for (; j < n; ++j) {                                                    /* column tail (< NR) */
            const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
            const float* as = J->asum + (size_t) j * nb;
            const uint8_t* w = wrow; float acc = 0.0f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q5K_BYTES) {
                float d = _cvtsh_ss(*(const uint16_t*) w), dmin = _cvtsh_ss(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* qh = w + 16; const uint8_t* qs = w + 48;
                __m256i h = _mm256_loadu_si256((const __m256i*) qh);
                for (int s = 0; s < 8; ++s) {
                    __m256i qb = _mm256_loadu_si256((const __m256i*) (qs + (s/2)*32));
                    __m256i nib = (s & 1) ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4)
                                          : _mm256_and_si256(qb, m4);
                    __m256i hb = _mm256_and_si256(_mm256_srl_epi16(h, _mm_cvtsi32_si128(s)), one);
                    __m256i wq = _mm256_or_si256(nib, _mm256_slli_epi16(hb, 4));
                    int blk = B*8 + s;
                    __m256i av = _mm256_loadu_si256((const __m256i*) (aq + (size_t) blk*32));
                    acc += ad[blk] * (d*sc[s]*jam_hsum8_256(KDOT(wq, av)) - dmin*mn[s]*as[blk]);
                }
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}

/* ---- Q6_K cached-repack (8-feature-wide). Q6_K has 16 per-16 INT8 (signed) scales, a signed 6-bit weight
 * (qv-32) and NO min. Store qv UNSIGNED (0..63, one byte/elem, group-interleaved) and fold the -32 as a
 * 32·Σaq bias (per-16 sums computed in-kernel), mirroring Q4_K's min term. ---- Repacked per super-block:
 * f32 d[8], i8 sc[16sub][8feat], u8 qs[16sub][4grp][32] (qv); sub16 s -> chunk (s/8)*4+(s%8)/2, half s&1. */
void jam_q6k_repack8(const void* Wv, int rows0, int re, int sblocks, void* outv) {
    const uint8_t* W = (const uint8_t*) Wv; jam_q6k_rpblock* out = (jam_q6k_rpblock*) outv;
    const size_t w_stride = (size_t) sblocks * JAM_Q6K_BYTES;
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
    const int mrows = ldc;
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

/* Q6_K: value = d·sc·(qv-32), qv 6-bit (ql nibble | qh 2-bit << 4). -32 folds into a signed weight (so
 * KDOT needs abs/sign); int8 scale per 16 elems -> scale the 8-lane dot by [sc0×4, sc1×4] before hsum. */
void jam_mm_q6k_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a; const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q6K_BYTES;
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(0x03), bias = _mm256_set1_epi8(32);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        int j = 0;
        for (; j + JAM_KQ_AVX2_NR <= n; j += JAM_KQ_AVX2_NR) {                 /* NR cols/decode + deferred hsum */
            __m256 dotv[JAM_KQ_AVX2_NR];
            const int8_t* aqc[JAM_KQ_AVX2_NR]; const float* adc[JAM_KQ_AVX2_NR];
            for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) { dotv[c]=_mm256_setzero_ps(); aqc[c]=AQ+(size_t)(j+c)*k; adc[c]=AD+(size_t)(j+c)*nb; }
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q6K_BYTES) {
                const uint8_t* ql = w; const uint8_t* qh = w + 128;
                const int8_t* sc = (const int8_t*) (w + 192);
                float d = _cvtsh_ss(*(const uint16_t*) (w + 208));
                for (int h = 0; h < 2; ++h) {
                    const uint8_t* qlb = ql + h*64; const uint8_t* qhb = qh + h*32;
                    for (int g = 0; g < 4; ++g) {
                        __m256i lq = _mm256_loadu_si256((const __m256i*) (qlb + (g & 1)*32));
                        __m256i lo = (g < 2) ? _mm256_and_si256(lq, m4)
                                             : _mm256_and_si256(_mm256_srli_epi16(lq, 4), m4);
                        __m256i hq = _mm256_loadu_si256((const __m256i*) qhb);
                        __m256i hi = _mm256_and_si256(_mm256_srl_epi16(hq, _mm_cvtsi32_si128(2*g)), m2);
                        __m256i wq = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hi, 4)), bias);  /* decode ONCE */
                        __m256i absw = _mm256_abs_epi8(wq);
                        int blk = B*8 + h*4 + g;
                        float s0 = (float) sc[h*8 + g*2], s1 = (float) sc[h*8 + g*2 + 1];
                        __m256 scv = _mm256_setr_ps(s0,s0,s0,s0, s1,s1,s1,s1);
                        for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) {
                            __m256i av = _mm256_loadu_si256((const __m256i*) (aqc[c] + (size_t) blk*32));
                            __m256 prod = _mm256_mul_ps(KDOT(absw, _mm256_sign_epi8(av, wq)), scv);
                            dotv[c] = _mm256_fmadd_ps(prod, _mm256_set1_ps(d * adc[c][blk]), dotv[c]);
                        }
                    }
                }
            }
            for (int c = 0; c < JAM_KQ_AVX2_NR; ++c) C[(size_t)(j+c)*ldc + i] = jam_hsum8_256(dotv[c]);
        }
        for (; j < n; ++j) {                                                    /* column tail (< NR) */
            const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
            const uint8_t* w = wrow; float acc = 0.0f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q6K_BYTES) {
                const uint8_t* ql = w; const uint8_t* qh = w + 128;
                const int8_t* sc = (const int8_t*) (w + 192);
                float d = _cvtsh_ss(*(const uint16_t*) (w + 208));
                for (int h = 0; h < 2; ++h) {
                    const uint8_t* qlb = ql + h*64; const uint8_t* qhb = qh + h*32;
                    for (int g = 0; g < 4; ++g) {
                        __m256i lq = _mm256_loadu_si256((const __m256i*) (qlb + (g & 1)*32));
                        __m256i lo = (g < 2) ? _mm256_and_si256(lq, m4)
                                             : _mm256_and_si256(_mm256_srli_epi16(lq, 4), m4);
                        __m256i hq = _mm256_loadu_si256((const __m256i*) qhb);
                        __m256i hi = _mm256_and_si256(_mm256_srl_epi16(hq, _mm_cvtsi32_si128(2*g)), m2);
                        __m256i wq = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hi, 4)), bias);
                        int blk = B*8 + h*4 + g;
                        __m256i av = _mm256_loadu_si256((const __m256i*) (aq + (size_t) blk*32));
                        __m256 prod = KDOT(_mm256_abs_epi8(wq), _mm256_sign_epi8(av, wq));
                        float s0 = (float) sc[h*8 + g*2], s1 = (float) sc[h*8 + g*2 + 1];
                        __m256 scv = _mm256_setr_ps(s0,s0,s0,s0, s1,s1,s1,s1);
                        acc += d * ad[blk] * jam_hsum8_256(_mm256_mul_ps(prod, scv));
                    }
                }
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}
