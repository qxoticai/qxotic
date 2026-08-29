/* ARM DOTPROD K-quant kernels (Q4_K/Q5_K/Q6_K @ F32 -> F32) - same decode as the NEON baseline, but the
 * int8 dot uses the dot-product extension (vdotq_s32 / sdot): one instruction does a 16-wide int8 dot into
 * 4 i32 lanes, so 32-wide = two sdots. Built with -march=armv8.2-a+dotprod. i8mm cores reuse this (sdot is
 * available there too); an smmla 2x2 variant would be a further step but barely helps the decode-bound K-quants. */
#include "jam_internal.h"
#include "jam_kquant.h"
#include <arm_neon.h>

/* 32-wide signed int8 dot via two sdots (Σ activations is precomputed in requant). */
static inline int jam_kdot32_dotprod(uint8x16_t w0u, uint8x16_t w1u, const int8_t* a) {
    int8x16_t w0 = vreinterpretq_s8_u8(w0u), w1 = vreinterpretq_s8_u8(w1u);
    int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
    int32x4_t d = vdotq_s32(vdupq_n_s32(0), w0, a0);
    d = vdotq_s32(d, w1, a1);
    return (int) vaddvq_s32(d);
}

/* 16-wide signed int8 dot (Q6_K; weight already holds qv-32). */
static inline int jam_kdot16_dotprod(int8x16_t w, const int8_t* a) {
    return (int) vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(a)));
}

#define JAM_KDOT32   jam_kdot32_dotprod
#define JAM_KDOT16   jam_kdot16_dotprod
#define JAM_Q4K_NAME jam_mm_q4k_dotprod
#define JAM_Q5K_NAME jam_mm_q5k_dotprod
#define JAM_Q6K_NAME jam_mm_q6k_dotprod
#include "jam_kquant_engine.inc"

/* ---- 4-row decode (n==1) GEMVs ----
 * The engine above amortizes decode across KTN activation columns; at n==1 that is 1x1 with a
 * serial vaddvq per dot. These process FOUR weight rows against the one activation column: four
 * independent decode+sdot chains, activation chunks loaded once, and the per-row sums packed into
 * one float32x4 lane-per-row via a vpaddq tree. Leftover rows delegate to the engine kernel. */

/* [r0 r1 r2 r3] row sums from four per-row sdot partials. */
static inline int32x4_t jam_krow4(int32x4_t p0, int32x4_t p1, int32x4_t p2, int32x4_t p3) {
    return vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
}

void jam_gemv_q6k_dotprod_4x1(void* arg, int rb, int re, int tid) {
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int sblocks = J->k / JAM_QKK;
    const size_t ws = (size_t)(J->lda / JAM_QKK) * JAM_Q6K_BYTES;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m2 = vdupq_n_u8(3);
    const int8x16_t bias = vdupq_n_s8(32);
    int i = rb;
    for (; i + 4 <= re; i += 4) {
        const uint8_t* wr[4];
        for (int r = 0; r < 4; ++r) wr[r] = (const uint8_t*) (W + (size_t)(i + r) * ws);
        float32x4_t facc = vdupq_n_f32(0);
        for (int B = 0; B < sblocks; ++B) {
            float dv[4];
            for (int r = 0; r < 4; ++r) dv[r] = (float) *(const __fp16*) (wr[r] + (size_t) B * JAM_Q6K_BYTES + 208);
            for (int h = 0; h < 2; ++h)
                for (int g = 0; g < 4; ++g) {
                    int blk = B*8 + h*4 + g;
                    const int8_t* a = AQ + (size_t) blk * 32;
                    const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
                    const int8x16_t sh = vdupq_n_s8(-(int8_t)(2*g));
                    int32x4_t p0[4], p1[4];
                    float s0a[4], s1a[4];
                    for (int r = 0; r < 4; ++r) {
                        const uint8_t* w = wr[r] + (size_t) B * JAM_Q6K_BYTES;
                        const uint8_t* qlb = w + h*64 + (g&1)*32;
                        const uint8_t* qhb = w + 128 + h*32;
                        uint8x16_t l0 = vld1q_u8(qlb), l1 = vld1q_u8(qlb + 16);
                        uint8x16_t lo0 = (g < 2) ? vandq_u8(l0, m4) : vshrq_n_u8(l0, 4);
                        uint8x16_t lo1 = (g < 2) ? vandq_u8(l1, m4) : vshrq_n_u8(l1, 4);
                        uint8x16_t hb0 = vld1q_u8(qhb), hb1 = vld1q_u8(qhb + 16);
                        uint8x16_t hi0 = vshlq_n_u8(vandq_u8(vshlq_u8(hb0, sh), m2), 4);
                        uint8x16_t hi1 = vshlq_n_u8(vandq_u8(vshlq_u8(hb1, sh), m2), 4);
                        int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo0, hi0)), bias);
                        int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo1, hi1)), bias);
                        p0[r] = vdotq_s32(vdupq_n_s32(0), w0, a0);
                        p1[r] = vdotq_s32(vdupq_n_s32(0), w1, a1);
                        const int8_t* sc = (const int8_t*) (w + 192);
                        s0a[r] = dv[r] * (float) sc[h*8 + g*2];
                        s1a[r] = dv[r] * (float) sc[h*8 + g*2 + 1];
                    }
                    float adb = AD[blk];
                    facc = vfmaq_f32(facc, vcvtq_f32_s32(jam_krow4(p0[0], p0[1], p0[2], p0[3])),
                                     vmulq_n_f32(vld1q_f32(s0a), adb));
                    facc = vfmaq_f32(facc, vcvtq_f32_s32(jam_krow4(p1[0], p1[1], p1[2], p1[3])),
                                     vmulq_n_f32(vld1q_f32(s1a), adb));
                }
        }
        vst1q_f32(C + i, facc);
    }
    if (i < re) jam_mm_q6k_dotprod(arg, i, re, tid);
}

void jam_gemv_q4k_dotprod_4x1(void* arg, int rb, int re, int tid) {
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad; const float* AS = J->asum;
    float* C = (float*) J->c;
    const int sblocks = J->k / JAM_QKK;
    const size_t ws = (size_t)(J->lda / JAM_QKK) * JAM_Q4K_BYTES;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    int i = rb;
    for (; i + 4 <= re; i += 4) {
        const uint8_t* wr[4];
        for (int r = 0; r < 4; ++r) wr[r] = (const uint8_t*) (W + (size_t)(i + r) * ws);
        float32x4_t facc = vdupq_n_f32(0);
        for (int B = 0; B < sblocks; ++B) {
            float d[4], dmin[4];
            uint8_t sc[4][8], mn[4][8];
            for (int r = 0; r < 4; ++r) {
                const uint8_t* w = wr[r] + (size_t) B * JAM_Q4K_BYTES;
                d[r] = (float) *(const __fp16*) w; dmin[r] = (float) *(const __fp16*) (w + 2);
                jam_q4k_scales_mins(w + 4, sc[r], mn[r]);
            }
            for (int g = 0; g < 4; ++g) {
                int bl = B*8 + 2*g, bh = bl + 1;
                const int8_t* al = AQ + (size_t) bl * 32;
                const int8_t* ah = AQ + (size_t) bh * 32;
                const int8x16_t al0 = vld1q_s8(al), al1 = vld1q_s8(al + 16);
                const int8x16_t ah0 = vld1q_s8(ah), ah1 = vld1q_s8(ah + 16);
                int32x4_t plo[4], phi[4];
                float sl[4], ml[4], sh[4], mh[4];
                for (int r = 0; r < 4; ++r) {
                    const uint8_t* q = wr[r] + (size_t) B * JAM_Q4K_BYTES + 16 + g*32;
                    uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
                    int8x16_t wl0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
                    int8x16_t wl1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
                    int8x16_t wh0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                    int8x16_t wh1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                    plo[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl0, al0), wl1, al1);
                    phi[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wh0, ah0), wh1, ah1);
                    sl[r] = d[r] * (float) sc[r][2*g];     ml[r] = dmin[r] * (float) mn[r][2*g];
                    sh[r] = d[r] * (float) sc[r][2*g + 1]; mh[r] = dmin[r] * (float) mn[r][2*g + 1];
                }
                facc = vfmaq_f32(facc, vcvtq_f32_s32(jam_krow4(plo[0], plo[1], plo[2], plo[3])),
                                 vmulq_n_f32(vld1q_f32(sl), AD[bl]));
                facc = vfmsq_f32(facc, vld1q_f32(ml), vdupq_n_f32(AD[bl] * AS[bl]));
                facc = vfmaq_f32(facc, vcvtq_f32_s32(jam_krow4(phi[0], phi[1], phi[2], phi[3])),
                                 vmulq_n_f32(vld1q_f32(sh), AD[bh]));
                facc = vfmsq_f32(facc, vld1q_f32(mh), vdupq_n_f32(AD[bh] * AS[bh]));
            }
        }
        vst1q_f32(C + i, facc);
    }
    if (i < re) jam_mm_q4k_dotprod(arg, i, re, tid);
}

void jam_gemv_q5k_dotprod_4x1(void* arg, int rb, int re, int tid) {
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad; const float* AS = J->asum;
    float* C = (float*) J->c;
    const int sblocks = J->k / JAM_QKK;
    const size_t ws = (size_t)(J->lda / JAM_QKK) * JAM_Q5K_BYTES;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), one = vdupq_n_u8(1);
    int i = rb;
    for (; i + 4 <= re; i += 4) {
        const uint8_t* wr[4];
        for (int r = 0; r < 4; ++r) wr[r] = (const uint8_t*) (W + (size_t)(i + r) * ws);
        float32x4_t facc = vdupq_n_f32(0);
        for (int B = 0; B < sblocks; ++B) {
            float d[4], dmin[4];
            uint8_t sc[4][8], mn[4][8];
            uint8x16_t h0[4], h1[4];
            for (int r = 0; r < 4; ++r) {
                const uint8_t* w = wr[r] + (size_t) B * JAM_Q5K_BYTES;
                d[r] = (float) *(const __fp16*) w; dmin[r] = (float) *(const __fp16*) (w + 2);
                jam_q4k_scales_mins(w + 4, sc[r], mn[r]);
                h0[r] = vld1q_u8(w + 16); h1[r] = vld1q_u8(w + 32);
            }
            for (int g = 0; g < 4; ++g) {
                int bl = B*8 + 2*g, bh = bl + 1;
                const int8_t* al = AQ + (size_t) bl * 32;
                const int8_t* ah = AQ + (size_t) bh * 32;
                const int8x16_t al0 = vld1q_s8(al), al1 = vld1q_s8(al + 16);
                const int8x16_t ah0 = vld1q_s8(ah), ah1 = vld1q_s8(ah + 16);
                const int8x16_t shL = vdupq_n_s8(-(int8_t)(2*g)), shH = vdupq_n_s8(-(int8_t)(2*g + 1));
                int32x4_t plo[4], phi[4];
                float sl[4], ml[4], sh[4], mh[4];
                for (int r = 0; r < 4; ++r) {
                    const uint8_t* q = wr[r] + (size_t) B * JAM_Q5K_BYTES + 48 + g*32;
                    uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
                    int8x16_t wl0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q0, m4), vshlq_n_u8(vandq_u8(vshlq_u8(h0[r], shL), one), 4)));
                    int8x16_t wl1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q1, m4), vshlq_n_u8(vandq_u8(vshlq_u8(h1[r], shL), one), 4)));
                    int8x16_t wh0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q0, 4), vshlq_n_u8(vandq_u8(vshlq_u8(h0[r], shH), one), 4)));
                    int8x16_t wh1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q1, 4), vshlq_n_u8(vandq_u8(vshlq_u8(h1[r], shH), one), 4)));
                    plo[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl0, al0), wl1, al1);
                    phi[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wh0, ah0), wh1, ah1);
                    sl[r] = d[r] * (float) sc[r][2*g];     ml[r] = dmin[r] * (float) mn[r][2*g];
                    sh[r] = d[r] * (float) sc[r][2*g + 1]; mh[r] = dmin[r] * (float) mn[r][2*g + 1];
                }
                facc = vfmaq_f32(facc, vcvtq_f32_s32(jam_krow4(plo[0], plo[1], plo[2], plo[3])),
                                 vmulq_n_f32(vld1q_f32(sl), AD[bl]));
                facc = vfmsq_f32(facc, vld1q_f32(ml), vdupq_n_f32(AD[bl] * AS[bl]));
                facc = vfmaq_f32(facc, vcvtq_f32_s32(jam_krow4(phi[0], phi[1], phi[2], phi[3])),
                                 vmulq_n_f32(vld1q_f32(sh), AD[bh]));
                facc = vfmsq_f32(facc, vld1q_f32(mh), vdupq_n_f32(AD[bh] * AS[bh]));
            }
        }
        vst1q_f32(C + i, facc);
    }
    if (i < re) jam_mm_q5k_dotprod(arg, i, re, tid);
}

/* Q6_K packed GEMV (caller-packed per-group layout, jam.h JAM_PACK_ABI): int8 payload lines - the
 * 6-bit unpack is gone entirely - plus original int8 scale pairs and f32 d. Two SDOTs per row per
 * 32-block, scales widened 8-at-a-time. Handles arbitrary row ranges (partial groups per-row). */
void jam_gemv_q6k_packed_4x1(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int nb = J->nb, sblocks = nb / 8;
    const size_t GB = (size_t) nb * 136 + (size_t) sblocks * 16;   /* jam.h JAM_PACK_ABI 1 */
    const int8_t* P = (const int8_t*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    int i = rb;
    while (i < re) {
        const int g = i / 4;
        const int r0 = i - g * 4;
        const int r1 = re - g * 4 < 4 ? re - g * 4 : 4;
        const int8_t* pg = P + (size_t) g * GB;
        const int8_t* scg = pg + (size_t) nb * 128;
        const float*  dg = (const float*) (scg + (size_t) nb * 8);
        if (r0 == 0 && r1 == 4) {
            float32x4_t facc = vdupq_n_f32(0);
            for (int B = 0; B < sblocks; ++B) {
                const float32x4_t d4 = vld1q_f32(dg + (size_t) B * 4);
                for (int b8 = 0; b8 < 8; ++b8) {
                    int blk = B * 8 + b8;
                    const int8_t* p = pg + (size_t) blk * 128;
                    const int8_t* a = AQ + (size_t) blk * 32;
                    const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
                    int32x4_t p0[4], p1[4];
                    for (int r = 0; r < 4; ++r) {
                        p0[r] = vdotq_s32(vdupq_n_s32(0), vld1q_s8(p + r * 32), a0);
                        p1[r] = vdotq_s32(vdupq_n_s32(0), vld1q_s8(p + r * 32 + 16), a1);
                    }
                    const int16x8_t sw = vmovl_s8(vld1_s8(scg + (size_t) blk * 8));
                    const float32x4_t s0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(sw))), d4);
                    const float32x4_t s1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(sw))), d4);
                    const float adb = AD[blk];
                    facc = vfmaq_f32(facc, vcvtq_f32_s32(jam_krow4(p0[0], p0[1], p0[2], p0[3])),
                                     vmulq_n_f32(s0, adb));
                    facc = vfmaq_f32(facc, vcvtq_f32_s32(jam_krow4(p1[0], p1[1], p1[2], p1[3])),
                                     vmulq_n_f32(s1, adb));
                }
            }
            vst1q_f32(C + i, facc);
        } else {
            for (int r = r0; r < r1; ++r) {
                float32x4_t f = vdupq_n_f32(0);
                for (int B = 0; B < sblocks; ++B) {
                    const float dr = dg[(size_t) B * 4 + r];
                    for (int b8 = 0; b8 < 8; ++b8) {
                        int blk = B * 8 + b8;
                        const int8_t* p = pg + (size_t) blk * 128 + (size_t) r * 32;
                        const int8_t* a = AQ + (size_t) blk * 32;
                        int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(p), vld1q_s8(a));
                        int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(p + 16), vld1q_s8(a + 16));
                        float s0 = dr * (float) scg[(size_t) blk * 8 + r];
                        float s1 = dr * (float) scg[(size_t) blk * 8 + 4 + r];
                        f = vfmaq_n_f32(f, vcvtq_f32_s32(d0), s0 * AD[blk]);
                        f = vfmaq_n_f32(f, vcvtq_f32_s32(d1), s1 * AD[blk]);
                    }
                }
                C[g * 4 + r] = vaddvq_f32(f);
            }
        }
        i = g * 4 + r1;
    }
}

/* Shared scale/min application for the split K-quant GEMVs: SM holds [4x sc | 4x mn] u8 per block,
 * DD [4x d | 4x dmin] f32 per super-block. acc += ad*(d*sc)*dot - (ad*asum)*(dmin*mn), vectorized
 * lane-per-row. */
static inline float32x4_t jam_ksplit_acc(float32x4_t facc, int32x4_t dots,
                                         const uint8_t* sm, float32x4_t d4, float32x4_t dmin4,
                                         float adb, float adas) {
    const uint16x8_t w16 = vmovl_u8(vld1_u8(sm));
    const float32x4_t scv = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16)));
    const float32x4_t mnv = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16)));
    facc = vfmaq_f32(facc, vcvtq_f32_s32(dots), vmulq_n_f32(vmulq_f32(scv, d4), adb));
    return vfmsq_f32(facc, vmulq_f32(mnv, dmin4), vdupq_n_f32(adas));
}

/* Q4_K packed GEMV (jam.h JAM_PACK_ABI layout): re-nibbled 4-bit payload, Q4_0-style decode,
 * per-32 sc/mn applied lane-per-row. */
void jam_gemv_q4k_packed_4x1(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int nb = J->nb, sblocks = nb / 8;
    const size_t GB = (size_t) nb * 72 + (size_t) sblocks * 32;    /* jam.h JAM_PACK_ABI 1 */
    const uint8_t* P = (const uint8_t*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad; const float* AS = J->asum;
    float* C = (float*) J->c;
    const int8x16_t zero = vdupq_n_s8(0);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    int i = rb;
    while (i < re) {
        const int g = i / 4;
        const int r0 = i - g * 4;
        const int r1 = re - g * 4 < 4 ? re - g * 4 : 4;
        const uint8_t* pg = P + (size_t) g * GB;
        const uint8_t* smg = pg + (size_t) nb * 64;
        const float*   ddg = (const float*) (smg + (size_t) nb * 8);
        if (r0 == 0 && r1 == 4) {
            float32x4_t facc = vdupq_n_f32(0);
            for (int B = 0; B < sblocks; ++B) {
                const float32x4_t d4 = vld1q_f32(ddg + (size_t) B * 8);
                const float32x4_t dmin4 = vld1q_f32(ddg + (size_t) B * 8 + 4);
                for (int b8 = 0; b8 < 8; ++b8) {
                    int blk = B * 8 + b8;
                    const uint8_t* p = pg + (size_t) blk * 64;
                    const int8_t* a = AQ + (size_t) blk * 32;
                    const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
                    int32x4_t pr[4];
                    for (int r = 0; r < 4; ++r) {
                        uint8x16_t q = vld1q_u8(p + r * 16);
                        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(q, m4));
                        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
                        pr[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, a0), hi, a1);
                    }
                    facc = jam_ksplit_acc(facc, jam_krow4(pr[0], pr[1], pr[2], pr[3]),
                                          smg + (size_t) blk * 8, d4, dmin4,
                                          AD[blk], AD[blk] * AS[blk]);
                }
            }
            vst1q_f32(C + i, facc);
        } else {
            for (int r = r0; r < r1; ++r) {
                float32x4_t f = vdupq_n_f32(0);
                float minsum = 0;
                for (int B = 0; B < sblocks; ++B) {
                    const float dr = ddg[(size_t) B * 8 + r], dm = ddg[(size_t) B * 8 + 4 + r];
                    for (int b8 = 0; b8 < 8; ++b8) {
                        int blk = B * 8 + b8;
                        const uint8_t* p = pg + (size_t) blk * 64 + (size_t) r * 16;
                        const int8_t* a = AQ + (size_t) blk * 32;
                        uint8x16_t q = vld1q_u8(p);
                        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(q, m4));
                        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
                        int32x4_t d0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(a)), hi, vld1q_s8(a + 16));
                        const uint8_t* sm = smg + (size_t) blk * 8;
                        f = vfmaq_n_f32(f, vcvtq_f32_s32(d0), dr * (float) sm[r] * AD[blk]);
                        minsum += dm * (float) sm[4 + r] * AD[blk] * AS[blk];
                    }
                }
                C[g * 4 + r] = vaddvq_f32(f) - minsum;
            }
        }
        i = g * 4 + r1;
    }
    (void) zero;
}

/* Q5_K packed GEMV (jam.h JAM_PACK_ABI layout): int8-expanded payload, zero decode. */
void jam_gemv_q5k_packed_4x1(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int nb = J->nb, sblocks = nb / 8;
    const size_t GB = (size_t) nb * 136 + (size_t) sblocks * 32;   /* jam.h JAM_PACK_ABI 1 */
    const int8_t* P = (const int8_t*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad; const float* AS = J->asum;
    float* C = (float*) J->c;
    int i = rb;
    while (i < re) {
        const int g = i / 4;
        const int r0 = i - g * 4;
        const int r1 = re - g * 4 < 4 ? re - g * 4 : 4;
        const int8_t* pg = P + (size_t) g * GB;
        const uint8_t* smg = (const uint8_t*) pg + (size_t) nb * 128;
        const float*   ddg = (const float*) (smg + (size_t) nb * 8);
        if (r0 == 0 && r1 == 4) {
            float32x4_t facc = vdupq_n_f32(0);
            for (int B = 0; B < sblocks; ++B) {
                const float32x4_t d4 = vld1q_f32(ddg + (size_t) B * 8);
                const float32x4_t dmin4 = vld1q_f32(ddg + (size_t) B * 8 + 4);
                for (int b8 = 0; b8 < 8; ++b8) {
                    int blk = B * 8 + b8;
                    const int8_t* p = pg + (size_t) blk * 128;
                    const int8_t* a = AQ + (size_t) blk * 32;
                    const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
                    int32x4_t pr[4];
                    for (int r = 0; r < 4; ++r)
                        pr[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(p + r * 32), a0),
                                          vld1q_s8(p + r * 32 + 16), a1);
                    facc = jam_ksplit_acc(facc, jam_krow4(pr[0], pr[1], pr[2], pr[3]),
                                          smg + (size_t) blk * 8, d4, dmin4,
                                          AD[blk], AD[blk] * AS[blk]);
                }
            }
            vst1q_f32(C + i, facc);
        } else {
            for (int r = r0; r < r1; ++r) {
                float32x4_t f = vdupq_n_f32(0);
                float minsum = 0;
                for (int B = 0; B < sblocks; ++B) {
                    const float dr = ddg[(size_t) B * 8 + r], dm = ddg[(size_t) B * 8 + 4 + r];
                    for (int b8 = 0; b8 < 8; ++b8) {
                        int blk = B * 8 + b8;
                        const int8_t* p = pg + (size_t) blk * 128 + (size_t) r * 32;
                        const int8_t* a = AQ + (size_t) blk * 32;
                        int32x4_t d0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(p), vld1q_s8(a)),
                                                 vld1q_s8(p + 16), vld1q_s8(a + 16));
                        const uint8_t* sm = smg + (size_t) blk * 8;
                        f = vfmaq_n_f32(f, vcvtq_f32_s32(d0), dr * (float) sm[r] * AD[blk]);
                        minsum += dm * (float) sm[4 + r] * AD[blk] * AS[blk];
                    }
                }
                C[g * 4 + r] = vaddvq_f32(f) - minsum;
            }
        }
        i = g * 4 + r1;
    }
}

/* ---- packed prefill (n > 1) kernels ----
 * Row-GROUP ranges [gb, ge): jam.c fans whole 4-row groups, so no partial-group paths here
 * (m % 4 == 0 is the packed-layout contract). Structure = the split GEMVs above with the
 * engine's KTN=4 column tile: one weight pass feeds 4 requant'd activation columns, so the
 * payload streams n/4 times and the (already cheap) decode amortizes 4x on top. */

void jam_mm_q4_0_packed_dotprod(void* arg, int gb, int ge, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int nb = J->nb, n = J->n, k = J->k, ldc = J->ldc;
    const size_t GB = (size_t) nb * 80;
    const uint8_t* base = (const uint8_t*) J->a;
    float* C = (float*) J->c;
    const int8x16_t e8 = vdupq_n_s8(8);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int g = gb; g < ge; ++g) {
        const uint8_t* pg = base + (size_t) g * GB;
        const float*   sg = (const float*) (pg + (size_t) nb * 64);
        const int i = g * 4;
        for (int j0 = 0; j0 < n; j0 += 4) {
            const int jn = n - j0 < 4 ? n - j0 : 4;
            float32x4_t facc[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            for (int b = 0; b < nb; ++b) {
                const uint8_t* p = pg + (size_t) b * 64;
                int8x16_t lo[4], hi[4];
                for (int r = 0; r < 4; ++r) {
                    uint8x16_t q = vld1q_u8(p + r * 16);
                    lo[r] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4)), e8);
                    hi[r] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), e8);
                }
                const float32x4_t s4 = vld1q_f32(sg + (size_t) b * 4);
                for (int c = 0; c < jn; ++c) {
                    const int j = j0 + c;
                    const int8_t* a = J->aq + (size_t) j * k + (size_t) b * 32;
                    const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
                    int32x4_t pr[4];
                    for (int r = 0; r < 4; ++r)
                        pr[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[r], a0), hi[r], a1);
                    facc[c] = vfmaq_f32(facc[c], vcvtq_f32_s32(jam_krow4(pr[0], pr[1], pr[2], pr[3])),
                                        vmulq_n_f32(s4, J->ad[(size_t) j * nb + b]));
                }
            }
            for (int c = 0; c < jn; ++c) vst1q_f32(C + (size_t) (j0 + c) * ldc + i, facc[c]);
        }
    }
}

void jam_mm_q4k_packed_dotprod(void* arg, int gb, int ge, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int nb = J->nb, sblocks = nb / 8, n = J->n, k = J->k, ldc = J->ldc;
    const size_t GB = (size_t) nb * 72 + (size_t) sblocks * 32;
    const uint8_t* base = (const uint8_t*) J->a;
    float* C = (float*) J->c;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int g = gb; g < ge; ++g) {
        const uint8_t* pg = base + (size_t) g * GB;
        const uint8_t* smg = pg + (size_t) nb * 64;
        const float*   ddg = (const float*) (smg + (size_t) nb * 8);
        const int i = g * 4;
        for (int j0 = 0; j0 < n; j0 += 4) {
            const int jn = n - j0 < 4 ? n - j0 : 4;
            float32x4_t facc[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            for (int B = 0; B < sblocks; ++B) {
                const float32x4_t d4 = vld1q_f32(ddg + (size_t) B * 8);
                const float32x4_t dmin4 = vld1q_f32(ddg + (size_t) B * 8 + 4);
                for (int b8 = 0; b8 < 8; ++b8) {
                    const int blk = B * 8 + b8;
                    const uint8_t* p = pg + (size_t) blk * 64;
                    int8x16_t lo[4], hi[4];
                    for (int r = 0; r < 4; ++r) {
                        uint8x16_t q = vld1q_u8(p + r * 16);
                        lo[r] = vreinterpretq_s8_u8(vandq_u8(q, m4));
                        hi[r] = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
                    }
                    const uint16x8_t w16 = vmovl_u8(vld1_u8(smg + (size_t) blk * 8));
                    const float32x4_t scd = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16))), d4);
                    const float32x4_t mnd = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16))), dmin4);
                    for (int c = 0; c < jn; ++c) {
                        const int j = j0 + c;
                        const int8_t* a = J->aq + (size_t) j * k + (size_t) blk * 32;
                        const float adb = J->ad[(size_t) j * nb + blk];
                        const float ads = adb * J->asum[(size_t) j * nb + blk];
                        const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
                        int32x4_t pr[4];
                        for (int r = 0; r < 4; ++r)
                            pr[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[r], a0), hi[r], a1);
                        facc[c] = vfmaq_f32(facc[c], vcvtq_f32_s32(jam_krow4(pr[0], pr[1], pr[2], pr[3])),
                                            vmulq_n_f32(scd, adb));
                        facc[c] = vfmsq_f32(facc[c], mnd, vdupq_n_f32(ads));
                    }
                }
            }
            for (int c = 0; c < jn; ++c) vst1q_f32(C + (size_t) (j0 + c) * ldc + i, facc[c]);
        }
    }
}

void jam_mm_q5k_packed_dotprod(void* arg, int gb, int ge, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int nb = J->nb, sblocks = nb / 8, n = J->n, k = J->k, ldc = J->ldc;
    const size_t GB = (size_t) nb * 136 + (size_t) sblocks * 32;
    const uint8_t* base = (const uint8_t*) J->a;
    float* C = (float*) J->c;
    for (int g = gb; g < ge; ++g) {
        const int8_t*  pg = (const int8_t*) (base + (size_t) g * GB);
        const uint8_t* smg = (const uint8_t*) pg + (size_t) nb * 128;
        const float*   ddg = (const float*) (smg + (size_t) nb * 8);
        const int i = g * 4;
        for (int j0 = 0; j0 < n; j0 += 4) {
            const int jn = n - j0 < 4 ? n - j0 : 4;
            float32x4_t facc[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            for (int B = 0; B < sblocks; ++B) {
                const float32x4_t d4 = vld1q_f32(ddg + (size_t) B * 8);
                const float32x4_t dmin4 = vld1q_f32(ddg + (size_t) B * 8 + 4);
                for (int b8 = 0; b8 < 8; ++b8) {
                    const int blk = B * 8 + b8;
                    const int8_t* p = pg + (size_t) blk * 128;
                    int8x16_t w0[4], w1[4];
                    for (int r = 0; r < 4; ++r) {
                        w0[r] = vld1q_s8(p + r * 32);
                        w1[r] = vld1q_s8(p + r * 32 + 16);
                    }
                    const uint16x8_t w16 = vmovl_u8(vld1_u8(smg + (size_t) blk * 8));
                    const float32x4_t scd = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16))), d4);
                    const float32x4_t mnd = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16))), dmin4);
                    for (int c = 0; c < jn; ++c) {
                        const int j = j0 + c;
                        const int8_t* a = J->aq + (size_t) j * k + (size_t) blk * 32;
                        const float adb = J->ad[(size_t) j * nb + blk];
                        const float ads = adb * J->asum[(size_t) j * nb + blk];
                        const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
                        int32x4_t pr[4];
                        for (int r = 0; r < 4; ++r)
                            pr[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0[r], a0), w1[r], a1);
                        facc[c] = vfmaq_f32(facc[c], vcvtq_f32_s32(jam_krow4(pr[0], pr[1], pr[2], pr[3])),
                                            vmulq_n_f32(scd, adb));
                        facc[c] = vfmsq_f32(facc[c], mnd, vdupq_n_f32(ads));
                    }
                }
            }
            for (int c = 0; c < jn; ++c) vst1q_f32(C + (size_t) (j0 + c) * ldc + i, facc[c]);
        }
    }
}

void jam_mm_q6k_packed_dotprod(void* arg, int gb, int ge, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int nb = J->nb, sblocks = nb / 8, n = J->n, k = J->k, ldc = J->ldc;
    const size_t GB = (size_t) nb * 136 + (size_t) sblocks * 16;
    const uint8_t* base = (const uint8_t*) J->a;
    float* C = (float*) J->c;
    for (int g = gb; g < ge; ++g) {
        const int8_t* pg = (const int8_t*) (base + (size_t) g * GB);
        const int8_t* scg = pg + (size_t) nb * 128;
        const float*  dg = (const float*) (scg + (size_t) nb * 8);
        const int i = g * 4;
        for (int j0 = 0; j0 < n; j0 += 4) {
            const int jn = n - j0 < 4 ? n - j0 : 4;
            float32x4_t facc[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            for (int B = 0; B < sblocks; ++B) {
                const float32x4_t d4 = vld1q_f32(dg + (size_t) B * 4);
                for (int b8 = 0; b8 < 8; ++b8) {
                    const int blk = B * 8 + b8;
                    const int8_t* p = pg + (size_t) blk * 128;
                    int8x16_t w0[4], w1[4];
                    for (int r = 0; r < 4; ++r) {
                        w0[r] = vld1q_s8(p + r * 32);
                        w1[r] = vld1q_s8(p + r * 32 + 16);
                    }
                    const int16x8_t sw = vmovl_s8(vld1_s8(scg + (size_t) blk * 8));
                    const float32x4_t s0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(sw))), d4);
                    const float32x4_t s1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(sw))), d4);
                    for (int c = 0; c < jn; ++c) {
                        const int j = j0 + c;
                        const int8_t* a = J->aq + (size_t) j * k + (size_t) blk * 32;
                        const float adb = J->ad[(size_t) j * nb + blk];
                        const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
                        int32x4_t p0[4], p1[4];
                        for (int r = 0; r < 4; ++r) {
                            p0[r] = vdotq_s32(vdupq_n_s32(0), w0[r], a0);
                            p1[r] = vdotq_s32(vdupq_n_s32(0), w1[r], a1);
                        }
                        facc[c] = vfmaq_f32(facc[c], vcvtq_f32_s32(jam_krow4(p0[0], p0[1], p0[2], p0[3])),
                                            vmulq_n_f32(s0, adb));
                        facc[c] = vfmaq_f32(facc[c], vcvtq_f32_s32(jam_krow4(p1[0], p1[1], p1[2], p1[3])),
                                            vmulq_n_f32(s1, adb));
                    }
                }
            }
            for (int c = 0; c < jn; ++c) vst1q_f32(C + (size_t) (j0 + c) * ldc + i, facc[c]);
        }
    }
}
