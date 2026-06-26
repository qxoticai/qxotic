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
