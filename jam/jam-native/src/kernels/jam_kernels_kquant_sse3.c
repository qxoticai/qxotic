/* SSE3 K-quant GEMM (Q4_K/Q5_K/Q6_K @ F32 -> F32) - the pre-AVX2 x86 floor for K-quants, replacing the
 * generic dequant-to-float path on SSE3-without-AVX2 machines. A *true* SSE3 floor: no SSSE3 maddubs and no
 * F16C, so weights decode to int8 with and/shift/or (software fp16 scale) and the int8 dot is sign-extend
 * (unpack with the sign mask) + madd_epi16, reduced with a final hsum. Mirrors the ARM kquant engine math
 * (jam_kquant_engine.inc): per (weight row, 4-column tile), decode each sub-block ONCE and dot all 4 columns;
 * the dmin·min term is corrected in float via Σx ≈ ad·Σaq (Q6_K folds -32 into a signed weight, no min term).
 * Consumes jam_q8_0_requant output (J->aq int8, J->ad per-32 scale, J->asum per-32 Σaq). Built -msse3.
 *
 * The tiles keep 4 outputs in the 4 LANES of one float vector - 4 columns of a row, or for the column tail
 * (decode) 4 rows of a column: the per-sub-block scale math is then 4-wide mulps/subps/addps, each lane
 * doing exactly its output's scalar operations in the same order (SSE3 has no FMA, so nothing contracts and
 * the results are the scalar form's bit for bit). Written out because only clang found that SLP form on its
 * own (gcc ran it as 4x scalar, ~20% slower) and gcc then spilled the scalar accumulators. The 4 dots
 * reduce together with one transpose-add instead of 4 hsums. */
#include <pmmintrin.h>   /* SSE3 (pulls in SSE/SSE2) */
#include <stdint.h>
#include "jam_internal.h"
#include "jam_kquant.h"          /* JAM_QKK, JAM_Q4K/5K/6K_BYTES, jam_q4k_scales_mins */
#include "jam_fp16.h"            /* jam_half2float (software fp16 -> fp32; SSE3 has no F16C) */
#include "jam_decode_x86_128.h"  /* JAM_DOT16 (shared int8 madd dot) */

/* Q4_K/Q5_K weights are 0..31, so they ZERO-extend (unpack with 0); spelled out because gcc does not see
 * the range through the and-mask and would sign-extend them with a compare per half. */
#define JAM_ZDOT16(w, a) _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi8((w), _mm_setzero_si128()), JAM_SEXT_LO(a)), \
                                       _mm_madd_epi16(_mm_unpackhi_epi8((w), _mm_setzero_si128()), JAM_SEXT_HI(a)))
/* 32-wide dot of non-negative weights (two 16-byte halves) vs 32 int8 activations -> 4 int32 partials. */
static inline __m128i jam_kdot32_sse3(__m128i wlo, __m128i whi, const int8_t* p) {
    return _mm_add_epi32(JAM_ZDOT16(wlo, _mm_loadu_si128((const __m128i*) p)),
                         JAM_ZDOT16(whi, _mm_loadu_si128((const __m128i*) (p + 16))));
}
/* 16-wide signed dot (Q6_K: qv-32 is genuinely signed) -> 4 int32 partials. */
static inline __m128i jam_kdot16_sse3(__m128i w, const int8_t* p) {
    return JAM_DOT16(w, _mm_loadu_si128((const __m128i*) p));
}
/* one dot's partials -> float (int32 sum exact in float: |dot| < 2^24). */
static inline float jam_hsum4_f(__m128i v) {
    __m128i s = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtss_f32(_mm_cvtepi32_ps(s));
}
/* four dots' partials -> {Σv0, Σv1, Σv2, Σv3} as floats (integer sums: any order is exact). */
static inline __m128 jam_hsum4x4_f(__m128i v0, __m128i v1, __m128i v2, __m128i v3) {
    __m128i s01 = _mm_add_epi32(_mm_unpacklo_epi32(v0, v1), _mm_unpackhi_epi32(v0, v1));
    __m128i s23 = _mm_add_epi32(_mm_unpacklo_epi32(v2, v3), _mm_unpackhi_epi32(v2, v3));
    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_unpacklo_epi64(s01, s23), _mm_unpackhi_epi64(s01, s23)));
}

/* The tile's per-column operand pointers (int8 activations, per-32 scales, per-32 Σaq) and lane gathers. */
typedef struct { const int8_t* aq[4]; const float* ad[4]; const float* as[4]; } kq_cols;
static inline kq_cols kq_cols_at(const jam_q8_job* J, int j) {
    kq_cols t;
    for (int c = 0; c < 4; ++c) {
        t.aq[c] = J->aq + (size_t) (j + c) * J->k;
        t.ad[c] = J->ad + (size_t) (j + c) * J->nb;
        t.as[c] = J->asum + (size_t) (j + c) * J->nb;
    }
    return t;
}
#define KQ_LANES(p, b) _mm_setr_ps((p)[0][b], (p)[1][b], (p)[2][b], (p)[3][b])
static inline void kq_store4(const jam_q8_job* J, int i, int j, __m128 acc) {
    float* C = (float*) J->c; const size_t ldc = (size_t) J->ldc;
    float o[4]; _mm_storeu_ps(o, acc);
    C[(size_t) j * ldc + i] = o[0]; C[(size_t) (j+1) * ldc + i] = o[1];
    C[(size_t) (j+2) * ldc + i] = o[2]; C[(size_t) (j+3) * ldc + i] = o[3];
}

/* ---- Q4_K / Q5_K: sub-block pair (2g, 2g+1) of super-block B decoded to wl (2g) and wh (2g+1) halves ---- */

typedef struct { float d, dmin; uint8_t sc[8], mn[8]; } kq_sb;   /* super-block scales */
static inline kq_sb kq45_scales(const uint8_t* w) {
    kq_sb s; s.d = jam_half2float(*(const uint16_t*) w); s.dmin = jam_half2float(*(const uint16_t*) (w + 2));
    jam_q4k_scales_mins(w + 4, s.sc, s.mn);
    return s;
}
static inline void q4k_decode(const uint8_t* w, int g, __m128i* wl0, __m128i* wl1, __m128i* wh0, __m128i* wh1) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const uint8_t* q = w + 16;
    __m128i q0 = _mm_loadu_si128((const __m128i*) (q + g*32)), q1 = _mm_loadu_si128((const __m128i*) (q + g*32 + 16));
    *wl0 = _mm_and_si128(q0, m4); *wl1 = _mm_and_si128(q1, m4);                                    /* sub-block 2g */
    *wh0 = _mm_and_si128(_mm_srli_epi16(q0, 4), m4); *wh1 = _mm_and_si128(_mm_srli_epi16(q1, 4), m4); /* 2g+1 */
}
/* Q5_K: Q4_K plus a 5th weight bit from qh (q5 = nibble | (bit s of qh[e] << 4), 0..31). */
static inline void q5k_decode(const uint8_t* w, int g, __m128i* wl0, __m128i* wl1, __m128i* wh0, __m128i* wh1) {
    const __m128i m4 = _mm_set1_epi8(0x0F), m1 = _mm_set1_epi8(1);
    const uint8_t* qh = w + 16; const uint8_t* qs = w + 48;
    __m128i h0 = _mm_loadu_si128((const __m128i*) qh), h1 = _mm_loadu_si128((const __m128i*) (qh + 16));
    __m128i q0 = _mm_loadu_si128((const __m128i*) (qs + g*32)), q1 = _mm_loadu_si128((const __m128i*) (qs + g*32 + 16));
    /* high bit: bit 2g (low sub-block) / 2g+1 (high) of qh[e], shifted into nibble bit 4 */
    __m128i bl0 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h0, 2*g), m1), 4);
    __m128i bl1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h1, 2*g), m1), 4);
    __m128i bh0 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h0, 2*g+1), m1), 4);
    __m128i bh1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h1, 2*g+1), m1), 4);
    *wl0 = _mm_or_si128(_mm_and_si128(q0, m4), bl0); *wl1 = _mm_or_si128(_mm_and_si128(q1, m4), bl1);
    *wh0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q0, 4), m4), bh0);
    *wh1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q1, 4), m4), bh1);
}


/* Per column, per sub-block pair: acc += ad[bl]·(dl·dot_lo - ml·Σaq[bl]) + ad[bh]·(dh·dot_hi - mh·Σaq[bh]).
 * Three shapes compute it: one row x 4 columns (columns in the lanes), 4 rows x one column (rows in the
 * lanes: the column tail, so all of decode) and the scalar one row x one column for what is left. Lane
 * by lane they run the scalar form's operations in its order. */
#define INLINE static inline __attribute__((always_inline))
#define KQ_UNROLL4 _Pragma("GCC unroll 4")   /* g constant: clang kept the loop, with its decode selects */
typedef void (*kq45_decode)(const uint8_t* w, int g, __m128i* wl0, __m128i* wl1, __m128i* wh0, __m128i* wh1);

INLINE __m128 kq45_cols4(const jam_q8_job* J, const uint8_t* w, int bytes, kq45_decode dec, int j) {
    const kq_cols t = kq_cols_at(J, j);
    __m128 acc = _mm_setzero_ps();
    for (int B = 0; B < J->k / JAM_QKK; ++B, w += bytes) {
        const kq_sb s = kq45_scales(w);
        KQ_UNROLL4 for (int g = 0; g < 4; ++g) {
            __m128i wl0, wl1, wh0, wh1; dec(w, g, &wl0, &wl1, &wh0, &wh1);
            const int bl = B*8 + 2*g, bh = bl + 1;
            const size_t ol = (size_t) bl*32, oh = (size_t) bh*32;
            __m128 dlo = jam_hsum4x4_f(jam_kdot32_sse3(wl0, wl1, t.aq[0] + ol), jam_kdot32_sse3(wl0, wl1, t.aq[1] + ol),
                                       jam_kdot32_sse3(wl0, wl1, t.aq[2] + ol), jam_kdot32_sse3(wl0, wl1, t.aq[3] + ol));
            __m128 dhi = jam_hsum4x4_f(jam_kdot32_sse3(wh0, wh1, t.aq[0] + oh), jam_kdot32_sse3(wh0, wh1, t.aq[1] + oh),
                                       jam_kdot32_sse3(wh0, wh1, t.aq[2] + oh), jam_kdot32_sse3(wh0, wh1, t.aq[3] + oh));
            __m128 lo = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(s.d * s.sc[2*g]), dlo),
                                   _mm_mul_ps(_mm_set1_ps(s.dmin * s.mn[2*g]), KQ_LANES(t.as, bl)));
            __m128 hi = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(s.d * s.sc[2*g+1]), dhi),
                                   _mm_mul_ps(_mm_set1_ps(s.dmin * s.mn[2*g+1]), KQ_LANES(t.as, bh)));
            acc = _mm_add_ps(acc, _mm_add_ps(_mm_mul_ps(KQ_LANES(t.ad, bl), lo), _mm_mul_ps(KQ_LANES(t.ad, bh), hi)));
        }
    }
    return acc;
}

INLINE __m128 kq45_rows4(const jam_q8_job* J, const uint8_t* w0, size_t w_stride, int bytes, kq45_decode dec, int j) {
    const int8_t* aq = J->aq + (size_t) j * J->k;
    const float* ad = J->ad + (size_t) j * J->nb; const float* as = J->asum + (size_t) j * J->nb;
    __m128 acc = _mm_setzero_ps();
    for (int B = 0; B < J->k / JAM_QKK; ++B) {
        const uint8_t* w[4];
        for (int r = 0; r < 4; ++r) w[r] = w0 + r * w_stride + (size_t) B * bytes;
        const kq_sb s[4] = { kq45_scales(w[0]), kq45_scales(w[1]), kq45_scales(w[2]), kq45_scales(w[3]) };
        const __m128 d = _mm_setr_ps(s[0].d, s[1].d, s[2].d, s[3].d);
        const __m128 dmin = _mm_setr_ps(s[0].dmin, s[1].dmin, s[2].dmin, s[3].dmin);
        KQ_UNROLL4 for (int g = 0; g < 4; ++g) {
            const int bl = B*8 + 2*g, bh = bl + 1;
            __m128i lo[4], hi[4];
            for (int r = 0; r < 4; ++r) {
                __m128i wl0, wl1, wh0, wh1; dec(w[r], g, &wl0, &wl1, &wh0, &wh1);
                lo[r] = jam_kdot32_sse3(wl0, wl1, aq + (size_t) bl*32);
                hi[r] = jam_kdot32_sse3(wh0, wh1, aq + (size_t) bh*32);
            }
            const __m128 dlo = jam_hsum4x4_f(lo[0], lo[1], lo[2], lo[3]), dhi = jam_hsum4x4_f(hi[0], hi[1], hi[2], hi[3]);
            const __m128 dl = _mm_mul_ps(d, _mm_setr_ps(s[0].sc[2*g], s[1].sc[2*g], s[2].sc[2*g], s[3].sc[2*g]));
            const __m128 ml = _mm_mul_ps(dmin, _mm_setr_ps(s[0].mn[2*g], s[1].mn[2*g], s[2].mn[2*g], s[3].mn[2*g]));
            const __m128 dh = _mm_mul_ps(d, _mm_setr_ps(s[0].sc[2*g+1], s[1].sc[2*g+1], s[2].sc[2*g+1], s[3].sc[2*g+1]));
            const __m128 mh = _mm_mul_ps(dmin, _mm_setr_ps(s[0].mn[2*g+1], s[1].mn[2*g+1], s[2].mn[2*g+1], s[3].mn[2*g+1]));
            const __m128 lo4 = _mm_sub_ps(_mm_mul_ps(dl, dlo), _mm_mul_ps(ml, _mm_set1_ps(as[bl])));
            const __m128 hi4 = _mm_sub_ps(_mm_mul_ps(dh, dhi), _mm_mul_ps(mh, _mm_set1_ps(as[bh])));
            acc = _mm_add_ps(acc, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(ad[bl]), lo4), _mm_mul_ps(_mm_set1_ps(ad[bh]), hi4)));
        }
    }
    return acc;
}

INLINE float kq45_one(const jam_q8_job* J, const uint8_t* w, int bytes, kq45_decode dec, int j) {
    const int8_t* aq = J->aq + (size_t) j * J->k;
    const float* ad = J->ad + (size_t) j * J->nb; const float* as = J->asum + (size_t) j * J->nb;
    float acc = 0.0f;
    for (int B = 0; B < J->k / JAM_QKK; ++B, w += bytes) {
        const kq_sb s = kq45_scales(w);
        KQ_UNROLL4 for (int g = 0; g < 4; ++g) {
            __m128i wl0, wl1, wh0, wh1; dec(w, g, &wl0, &wl1, &wh0, &wh1);
            const int bl = B*8 + 2*g, bh = bl + 1;
            float dl = s.d*s.sc[2*g], ml = s.dmin*s.mn[2*g], dh = s.d*s.sc[2*g+1], mh = s.dmin*s.mn[2*g+1];
            float dlo = jam_hsum4_f(jam_kdot32_sse3(wl0, wl1, aq + (size_t) bl*32));
            float dhi = jam_hsum4_f(jam_kdot32_sse3(wh0, wh1, aq + (size_t) bh*32));
            acc += ad[bl] * (dl*dlo - ml*as[bl]) + ad[bh] * (dh*dhi - mh*as[bh]);
        }
    }
    return acc;
}

/* rows in 4s x columns in 4s, the column tail 4 rows at a time, then the row tail */
INLINE void kq45_sweep(const jam_q8_job* J, int rb, int re, int bytes, kq45_decode dec) {
    const size_t w_stride = (size_t) (J->lda / JAM_QKK) * bytes;
    const int n = J->n, n4 = n & ~3;
    float* C = (float*) J->c; const size_t ldc = (size_t) J->ldc;
    int i = rb;
    for (; i + 4 <= re; i += 4) {
        const uint8_t* w0 = (const uint8_t*) J->a + (size_t) i * w_stride;
        for (int j = 0; j < n4; j += 4)
            for (int r = 0; r < 4; ++r) kq_store4(J, i + r, j, kq45_cols4(J, w0 + r * w_stride, bytes, dec, j));
        for (int j = n4; j < n; ++j) _mm_storeu_ps(C + (size_t) j * ldc + i, kq45_rows4(J, w0, w_stride, bytes, dec, j));
    }
    for (; i < re; ++i) {
        const uint8_t* w = (const uint8_t*) J->a + (size_t) i * w_stride;
        for (int j = 0; j < n4; j += 4) kq_store4(J, i, j, kq45_cols4(J, w, bytes, dec, j));
        for (int j = n4; j < n; ++j) C[(size_t) j * ldc + i] = kq45_one(J, w, bytes, dec, j);
    }
}

void jam_mm_q4k_sse3(void* arg, int rb, int re, int tid) {
    (void) tid;
    kq45_sweep((const jam_q8_job*) arg, rb, re, JAM_Q4K_BYTES, q4k_decode);
}
void jam_mm_q5k_sse3(void* arg, int rb, int re, int tid) {
    (void) tid;
    kq45_sweep((const jam_q8_job*) arg, rb, re, JAM_Q5K_BYTES, q5k_decode);
}

/* Q6_K: value = d·sc·(qv-32), qv 6-bit (ql nibble | qh 2-bit << 4). -32 folds into a signed weight; int8
 * scales, one per 16 elements (two per 32-sub-block), so each sub-block is two 16-wide dots. No min term.
 * Per column: acc += ad[blk]·(s0·dot0 + s1·dot1). Same three shapes. */
static inline void q6k_decode(const uint8_t* w, int h, int g, __m128i* w0, __m128i* w1) {
    const __m128i m4 = _mm_set1_epi8(0x0F), m2 = _mm_set1_epi8(3), bias = _mm_set1_epi8(32);
    const uint8_t* qlb = w + h*64; const uint8_t* qhb = w + 128 + h*32;
    __m128i l0 = _mm_loadu_si128((const __m128i*) (qlb + (g&1)*32));
    __m128i l1 = _mm_loadu_si128((const __m128i*) (qlb + (g&1)*32 + 16));
    __m128i lo0 = (g < 2) ? _mm_and_si128(l0, m4) : _mm_and_si128(_mm_srli_epi16(l0, 4), m4);
    __m128i lo1 = (g < 2) ? _mm_and_si128(l1, m4) : _mm_and_si128(_mm_srli_epi16(l1, 4), m4);
    __m128i hb0 = _mm_loadu_si128((const __m128i*) qhb), hb1 = _mm_loadu_si128((const __m128i*) (qhb + 16));
    __m128i hi0 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hb0, 2*g), m2), 4);
    __m128i hi1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hb1, 2*g), m2), 4);
    *w0 = _mm_sub_epi8(_mm_or_si128(lo0, hi0), bias);   /* qv-32 (signed) */
    *w1 = _mm_sub_epi8(_mm_or_si128(lo1, hi1), bias);
}

INLINE __m128 q6k_cols4(const jam_q8_job* J, const uint8_t* w, int j) {
    const kq_cols t = kq_cols_at(J, j);
    __m128 acc = _mm_setzero_ps();
    for (int B = 0; B < J->k / JAM_QKK; ++B, w += JAM_Q6K_BYTES) {
        const int8_t* sc = (const int8_t*) (w + 192);
        const float d = jam_half2float(*(const uint16_t*) (w + 208));
        for (int h = 0; h < 2; ++h)
            KQ_UNROLL4 for (int g = 0; g < 4; ++g) {
                __m128i w0, w1; q6k_decode(w, h, g, &w0, &w1);
                const int blk = B*8 + h*4 + g;
                const size_t o = (size_t) blk*32;
                __m128 d0 = jam_hsum4x4_f(jam_kdot16_sse3(w0, t.aq[0] + o), jam_kdot16_sse3(w0, t.aq[1] + o),
                                          jam_kdot16_sse3(w0, t.aq[2] + o), jam_kdot16_sse3(w0, t.aq[3] + o));
                __m128 d1 = jam_hsum4x4_f(jam_kdot16_sse3(w1, t.aq[0] + o + 16), jam_kdot16_sse3(w1, t.aq[1] + o + 16),
                                          jam_kdot16_sse3(w1, t.aq[2] + o + 16), jam_kdot16_sse3(w1, t.aq[3] + o + 16));
                __m128 s0 = _mm_set1_ps(d * (float) sc[h*8 + g*2]), s1 = _mm_set1_ps(d * (float) sc[h*8 + g*2 + 1]);
                acc = _mm_add_ps(acc, _mm_mul_ps(KQ_LANES(t.ad, blk), _mm_add_ps(_mm_mul_ps(s0, d0), _mm_mul_ps(s1, d1))));
            }
    }
    return acc;
}

INLINE __m128 q6k_rows4(const jam_q8_job* J, const uint8_t* w0, size_t w_stride, int j) {
    const int8_t* aq = J->aq + (size_t) j * J->k; const float* ad = J->ad + (size_t) j * J->nb;
    __m128 acc = _mm_setzero_ps();
    for (int B = 0; B < J->k / JAM_QKK; ++B) {
        const uint8_t* w[4];
        for (int r = 0; r < 4; ++r) w[r] = w0 + r * w_stride + (size_t) B * JAM_Q6K_BYTES;
        const __m128 d = _mm_setr_ps(jam_half2float(*(const uint16_t*) (w[0] + 208)), jam_half2float(*(const uint16_t*) (w[1] + 208)),
                                     jam_half2float(*(const uint16_t*) (w[2] + 208)), jam_half2float(*(const uint16_t*) (w[3] + 208)));
        for (int h = 0; h < 2; ++h)
            KQ_UNROLL4 for (int g = 0; g < 4; ++g) {
                const int blk = B*8 + h*4 + g, e0 = h*8 + g*2;
                const size_t o = (size_t) blk*32;
                __m128i p0[4], p1[4];
                for (int r = 0; r < 4; ++r) {
                    __m128i w0r, w1r; q6k_decode(w[r], h, g, &w0r, &w1r);
                    p0[r] = jam_kdot16_sse3(w0r, aq + o); p1[r] = jam_kdot16_sse3(w1r, aq + o + 16);
                }
                const __m128 d0 = jam_hsum4x4_f(p0[0], p0[1], p0[2], p0[3]), d1 = jam_hsum4x4_f(p1[0], p1[1], p1[2], p1[3]);
                const int8_t* sc[4] = { (const int8_t*) (w[0] + 192), (const int8_t*) (w[1] + 192),
                                        (const int8_t*) (w[2] + 192), (const int8_t*) (w[3] + 192) };
                const __m128 s0 = _mm_mul_ps(d, _mm_setr_ps(sc[0][e0], sc[1][e0], sc[2][e0], sc[3][e0]));
                const __m128 s1 = _mm_mul_ps(d, _mm_setr_ps(sc[0][e0 + 1], sc[1][e0 + 1], sc[2][e0 + 1], sc[3][e0 + 1]));
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(ad[blk]), _mm_add_ps(_mm_mul_ps(s0, d0), _mm_mul_ps(s1, d1))));
            }
    }
    return acc;
}

INLINE float q6k_one(const jam_q8_job* J, const uint8_t* w, int j) {
    const int8_t* aq = J->aq + (size_t) j * J->k; const float* ad = J->ad + (size_t) j * J->nb;
    float acc = 0.0f;
    for (int B = 0; B < J->k / JAM_QKK; ++B, w += JAM_Q6K_BYTES) {
        const int8_t* sc = (const int8_t*) (w + 192);
        const float d = jam_half2float(*(const uint16_t*) (w + 208));
        for (int h = 0; h < 2; ++h)
            KQ_UNROLL4 for (int g = 0; g < 4; ++g) {
                __m128i w0, w1; q6k_decode(w, h, g, &w0, &w1);
                const int blk = B*8 + h*4 + g;
                float s0 = d * (float) sc[h*8 + g*2], s1 = d * (float) sc[h*8 + g*2 + 1];
                float dot0 = jam_hsum4_f(jam_kdot16_sse3(w0, aq + (size_t) blk*32));
                float dot1 = jam_hsum4_f(jam_kdot16_sse3(w1, aq + (size_t) blk*32 + 16));
                acc += ad[blk] * (s0 * dot0 + s1 * dot1);
            }
    }
    return acc;
}

void jam_mm_q6k_sse3(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const size_t w_stride = (size_t) (J->lda / JAM_QKK) * JAM_Q6K_BYTES;
    const int n = J->n, n4 = n & ~3;
    float* C = (float*) J->c; const size_t ldc = (size_t) J->ldc;
    int i = rb;
    for (; i + 4 <= re; i += 4) {   /* the kq45_sweep order */
        const uint8_t* w0 = (const uint8_t*) J->a + (size_t) i * w_stride;
        for (int j = 0; j < n4; j += 4)
            for (int r = 0; r < 4; ++r) kq_store4(J, i + r, j, q6k_cols4(J, w0 + r * w_stride, j));
        for (int j = n4; j < n; ++j) _mm_storeu_ps(C + (size_t) j * ldc + i, q6k_rows4(J, w0, w_stride, j));
    }
    for (; i < re; ++i) {
        const uint8_t* w = (const uint8_t*) J->a + (size_t) i * w_stride;
        for (int j = 0; j < n4; j += 4) kq_store4(J, i, j, q6k_cols4(J, w, j));
        for (int j = n4; j < n; ++j) C[(size_t) j * ldc + i] = q6k_one(J, w, j);
    }
}
