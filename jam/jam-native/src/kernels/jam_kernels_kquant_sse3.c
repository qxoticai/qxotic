/* SSE3 K-quant GEMM (Q4_K/Q5_K/Q6_K @ F32 -> F32) — the pre-AVX2 x86 floor for K-quants, replacing the
 * generic dequant-to-float path on SSE3-without-AVX2 machines. A *true* SSE3 floor: no SSSE3 maddubs and no
 * F16C, so weights decode to int8 with and/shift/or (software fp16 scale) and the int8 dot is sign-extend
 * (unpack + arithmetic-shift) + madd_epi16, reduced with a final hsum. Mirrors the ARM kquant engine math
 * (jam_kquant_engine.inc): per (weight row, 4-column tile), decode each sub-block ONCE and dot all 4 columns;
 * the dmin·min term is corrected in float via Σx ≈ ad·Σaq (Q6_K folds -32 into a signed weight, no min term).
 * Consumes jam_q8_0_requant output (J->aq int8, J->ad per-32 scale, J->asum per-32 Σaq). Built -msse3. */
#include <pmmintrin.h>   /* SSE3 (pulls in SSE/SSE2) */
#include <stdint.h>
#include "jam_internal.h"
#include "jam_kquant.h"   /* JAM_QKK, JAM_Q4K/5K/6K_BYTES, jam_q4k_scales_mins */
#include "jam_fp16.h"     /* jam_half2float (software fp16 -> fp32; SSE3 has no F16C) */

#define JAM_KTN 4         /* activation columns tiled per decoded weight sub-block (matches the SSE3 4-col tile) */

/* signed int8 -> int16 (low / high 8 lanes) via duplicate + arithmetic shift; K-quant nibbles are <128 so the
 * sign bit is clear (zero-extend), Q6_K weights are genuinely signed (qv-32) — both correct under sign-extend. */
#define SEXT_LO(x) _mm_srai_epi16(_mm_unpacklo_epi8((x), (x)), 8)
#define SEXT_HI(x) _mm_srai_epi16(_mm_unpackhi_epi8((x), (x)), 8)
#define DOT16(w, a) _mm_add_epi32(_mm_madd_epi16(SEXT_LO(w), SEXT_LO(a)), _mm_madd_epi16(SEXT_HI(w), SEXT_HI(a)))

static inline int jam_hsum4_epi32(__m128i v) {
    __m128i s = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
/* 32-wide signed dot: two 16-byte weight halves vs 32 int8 activations -> scalar int. */
static inline int jam_kdot32_sse3(__m128i wlo, __m128i whi, const int8_t* p) {
    return jam_hsum4_epi32(_mm_add_epi32(DOT16(wlo, _mm_loadu_si128((const __m128i*) p)),
                                         DOT16(whi, _mm_loadu_si128((const __m128i*) (p + 16)))));
}
/* 16-wide signed dot (Q6_K): 16 int8 weights vs 16 int8 activations -> scalar int. */
static inline int jam_kdot16_sse3(__m128i w, const int8_t* p) {
    return jam_hsum4_epi32(DOT16(w, _mm_loadu_si128((const __m128i*) p)));
}

void jam_mm_q4k_sse3(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t)(J->lda / JAM_QKK) * JAM_Q4K_BYTES;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j0 = 0; j0 < n; j0 += JAM_KTN) {
            float acc[JAM_KTN]; for (int c = 0; c < JAM_KTN; ++c) acc[c] = 0.0f;
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q4K_BYTES) {
                float d = jam_half2float(*(const uint16_t*) w), dmin = jam_half2float(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* q = w + 16;
                for (int g = 0; g < 4; ++g) {
                    __m128i q0 = _mm_loadu_si128((const __m128i*) (q + g*32));
                    __m128i q1 = _mm_loadu_si128((const __m128i*) (q + g*32 + 16));
                    __m128i wl0 = _mm_and_si128(q0, m4), wl1 = _mm_and_si128(q1, m4);                     /* sub-block 2g */
                    __m128i wh0 = _mm_and_si128(_mm_srli_epi16(q0, 4), m4), wh1 = _mm_and_si128(_mm_srli_epi16(q1, 4), m4); /* 2g+1 */
                    int bl = B*8 + 2*g, bh = bl + 1;
                    float dl = d*sc[2*g], ml = dmin*mn[2*g], dh = d*sc[2*g+1], mh = dmin*mn[2*g+1];
                    for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j >= n) break;
                        const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
                        const float* as = J->asum + (size_t) j * nb;
                        int dlo = jam_kdot32_sse3(wl0, wl1, aq + (size_t) bl*32);
                        int dhi = jam_kdot32_sse3(wh0, wh1, aq + (size_t) bh*32);
                        acc[c] += ad[bl] * (dl*dlo - ml*as[bl]) + ad[bh] * (dh*dhi - mh*as[bh]);
                    }
                }
            }
            for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j < n) C[(size_t) j*ldc + i] = acc[c]; }
        }
    }
}

/* Q5_K: Q4_K plus a 5th weight bit from qh (q5 = nibble | (bit s of qh[e] << 4), 0..31). */
void jam_mm_q5k_sse3(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t)(J->lda / JAM_QKK) * JAM_Q5K_BYTES;
    const __m128i m4 = _mm_set1_epi8(0x0F), m1 = _mm_set1_epi8(1);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j0 = 0; j0 < n; j0 += JAM_KTN) {
            float acc[JAM_KTN]; for (int c = 0; c < JAM_KTN; ++c) acc[c] = 0.0f;
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q5K_BYTES) {
                float d = jam_half2float(*(const uint16_t*) w), dmin = jam_half2float(*(const uint16_t*) (w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* qh = w + 16; const uint8_t* qs = w + 48;
                __m128i h0 = _mm_loadu_si128((const __m128i*) qh), h1 = _mm_loadu_si128((const __m128i*) (qh + 16));
                for (int g = 0; g < 4; ++g) {
                    __m128i q0 = _mm_loadu_si128((const __m128i*) (qs + g*32));
                    __m128i q1 = _mm_loadu_si128((const __m128i*) (qs + g*32 + 16));
                    /* high bit: bit 2g (low sub-block) / 2g+1 (high) of qh[e], shifted into nibble bit 4 */
                    __m128i bl0 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h0, 2*g), m1), 4);
                    __m128i bl1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h1, 2*g), m1), 4);
                    __m128i bh0 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h0, 2*g+1), m1), 4);
                    __m128i bh1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h1, 2*g+1), m1), 4);
                    __m128i wl0 = _mm_or_si128(_mm_and_si128(q0, m4), bl0), wl1 = _mm_or_si128(_mm_and_si128(q1, m4), bl1);
                    __m128i wh0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q0, 4), m4), bh0);
                    __m128i wh1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q1, 4), m4), bh1);
                    int bl = B*8 + 2*g, bh = bl + 1;
                    float dl = d*sc[2*g], ml = dmin*mn[2*g], dh = d*sc[2*g+1], mh = dmin*mn[2*g+1];
                    for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j >= n) break;
                        const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
                        const float* as = J->asum + (size_t) j * nb;
                        int dlo = jam_kdot32_sse3(wl0, wl1, aq + (size_t) bl*32);
                        int dhi = jam_kdot32_sse3(wh0, wh1, aq + (size_t) bh*32);
                        acc[c] += ad[bl] * (dl*dlo - ml*as[bl]) + ad[bh] * (dh*dhi - mh*as[bh]);
                    }
                }
            }
            for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j < n) C[(size_t) j*ldc + i] = acc[c]; }
        }
    }
}

/* Q6_K: value = d·sc·(qv-32), qv 6-bit (ql nibble | qh 2-bit << 4). -32 folds into a signed weight; int8
 * scales, one per 16 elements (two per 32-sub-block), so each sub-block is two 16-wide dots. No min term. */
void jam_mm_q6k_sse3(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t)(J->lda / JAM_QKK) * JAM_Q6K_BYTES;
    const __m128i m4 = _mm_set1_epi8(0x0F), m2 = _mm_set1_epi8(3), bias = _mm_set1_epi8(32);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j0 = 0; j0 < n; j0 += JAM_KTN) {
            float acc[JAM_KTN]; for (int c = 0; c < JAM_KTN; ++c) acc[c] = 0.0f;
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q6K_BYTES) {
                const uint8_t* ql = w; const uint8_t* qh = w + 128;
                const int8_t* sc = (const int8_t*) (w + 192);
                float d = jam_half2float(*(const uint16_t*) (w + 208));
                for (int h = 0; h < 2; ++h) {
                    const uint8_t* qlb = ql + h*64; const uint8_t* qhb = qh + h*32;
                    for (int g = 0; g < 4; ++g) {
                        __m128i l0 = _mm_loadu_si128((const __m128i*) (qlb + (g&1)*32));
                        __m128i l1 = _mm_loadu_si128((const __m128i*) (qlb + (g&1)*32 + 16));
                        __m128i lo0 = (g < 2) ? _mm_and_si128(l0, m4) : _mm_and_si128(_mm_srli_epi16(l0, 4), m4);
                        __m128i lo1 = (g < 2) ? _mm_and_si128(l1, m4) : _mm_and_si128(_mm_srli_epi16(l1, 4), m4);
                        __m128i hb0 = _mm_loadu_si128((const __m128i*) qhb), hb1 = _mm_loadu_si128((const __m128i*) (qhb + 16));
                        __m128i hi0 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hb0, 2*g), m2), 4);
                        __m128i hi1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hb1, 2*g), m2), 4);
                        __m128i w0 = _mm_sub_epi8(_mm_or_si128(lo0, hi0), bias);   /* qv-32 (signed) */
                        __m128i w1 = _mm_sub_epi8(_mm_or_si128(lo1, hi1), bias);
                        int blk = B*8 + h*4 + g;
                        float s0 = d * (float) sc[h*8 + g*2], s1 = d * (float) sc[h*8 + g*2 + 1];
                        for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j >= n) break;
                            const int8_t* a = AQ + (size_t) j * k + (size_t) blk*32; const float* ad = AD + (size_t) j * nb;
                            int dot0 = jam_kdot16_sse3(w0, a), dot1 = jam_kdot16_sse3(w1, a + 16);
                            acc[c] += ad[blk] * (s0 * dot0 + s1 * dot1);
                        }
                    }
                }
            }
            for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j < n) C[(size_t) j*ldc + i] = acc[c]; }
        }
    }
}
