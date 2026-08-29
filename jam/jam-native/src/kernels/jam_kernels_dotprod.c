/* DOTPROD simple-block kernels (Q8_0/Q4_0/MXFP4 @ F32 -> F32; built -march=armv8.2-a+dotprod). The int8
 * block-dot is two vdotq_s32 (signed 8-bit dot-product, 4 elems/lane) - the int8 workhorse on modern ARM
 * (all Apple M-series, Graviton2+, recent phones). Decode per quant in jam_decode_neon.h; engine shared. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"

/* vdotq_s32(acc,a,b): acc[l] += Σ_{4} a[4l+..]·b[4l+..]. Two of them cover the 32-elem block -> int32x4. */
#define JAM_BLKDOT(wlo,whi,blo,bhi) vdotq_s32(vdotq_s32(vdupq_n_s32(0), wlo, blo), whi, bhi)

/* Decode (n==1) GEMV specializes across output rows: one activation load feeds four independent
 * weight streams - cuts activation/scale loads 4x and gives the core 4 independent SDOT chains. */
#define JAM_BLK       jam_q8_blk
#define JAM_DECODE    jam_decode_q8_0_neon
#define JAM_GEMV_NAME jam_gemv_q8_0_dotprod_4x1
#include "jam_gemv4_dotprod.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_GEMV_NAME

#define JAM_BLK       jam_q4_0_blk
#define JAM_DECODE    jam_decode_q4_0_neon
#define JAM_GEMV_NAME jam_gemv_q4_0_dotprod_4x1
#include "jam_gemv4_dotprod.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_GEMV_NAME

/* Packed Q4_0 GEMV (caller-packed per-group layout, jam.h JAM_PACK_ABI): each k-block of a 4-row
 * group is one aligned 64B line of nibble payloads, with the fp16 scales decoded to f32 behind
 * them. Same SDOT math and accumulation order as jam_gemv_q4_0_dotprod_4x1 (bit-identical output);
 * the layout alone measures 30-70% more GB/s on M3 Pro decode shapes. */
void jam_gemv_q4_0_packed_4x1(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int nb = J->nb;
    const size_t GB = (size_t) nb * 80;    /* jam.h JAM_PACK_ABI 1 */
    const uint8_t* P = (const uint8_t*) J->a;
    float* C = (float*) J->c;
    const int8_t* q = J->aq;
    const float* d = J->ad;
    const int8x16_t e8 = vdupq_n_s8(8);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);

    int i = rb;
    while (i < re) {
        const int g = i / 4;
        const int r0 = i - g * 4;
        const int r1 = re - g * 4 < 4 ? re - g * 4 : 4;
        const uint8_t* pg = P + (size_t) g * GB;
        const float*   sg = (const float*) (pg + (size_t) nb * 64);
        if (r0 == 0 && r1 == 4) {
            float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0);
            float32x4_t f2 = vdupq_n_f32(0), f3 = vdupq_n_f32(0);
            int b = 0;
            /* two k-blocks per iteration, all loads issued up front: the second block's payload
             * streams in while the first block's SDOT chain retires (~12% on the short-k expert
             * down projection, the weakest decode shape) */
            for (; b + 2 <= nb; b += 2) {
                const uint8_t* g0 = pg + (size_t) b * 64;
                const uint8_t* g1 = g0 + 64;
                const int8_t* p = q + (size_t) b * 32;
                uint8x16_t q0 = vld1q_u8(g0), q1 = vld1q_u8(g0 + 16);
                uint8x16_t q2 = vld1q_u8(g0 + 32), q3 = vld1q_u8(g0 + 48);
                uint8x16_t w0 = vld1q_u8(g1), w1 = vld1q_u8(g1 + 16);
                uint8x16_t w2 = vld1q_u8(g1 + 32), w3 = vld1q_u8(g1 + 48);
                const int8x16_t alo = vld1q_s8(p), ahi = vld1q_s8(p + 16);
                const int8x16_t blo = vld1q_s8(p + 32), bhi = vld1q_s8(p + 48);
                const float32x4_t s0 = vmulq_n_f32(vld1q_f32(sg + (size_t) b * 4), d[b]);
                const float32x4_t s1 = vmulq_n_f32(vld1q_f32(sg + (size_t) b * 4 + 4), d[b + 1]);
                int8x16_t lo, hi;
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q0, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), e8);
                f0 = vfmaq_n_f32(f0, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), vgetq_lane_f32(s0, 0));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q1, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)), e8);
                f1 = vfmaq_n_f32(f1, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), vgetq_lane_f32(s0, 1));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q2, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q2, 4)), e8);
                f2 = vfmaq_n_f32(f2, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), vgetq_lane_f32(s0, 2));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q3, 4)), e8);
                f3 = vfmaq_n_f32(f3, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), vgetq_lane_f32(s0, 3));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w0, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w0, 4)), e8);
                f0 = vfmaq_n_f32(f0, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, blo, bhi)), vgetq_lane_f32(s1, 0));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w1, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w1, 4)), e8);
                f1 = vfmaq_n_f32(f1, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, blo, bhi)), vgetq_lane_f32(s1, 1));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w2, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w2, 4)), e8);
                f2 = vfmaq_n_f32(f2, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, blo, bhi)), vgetq_lane_f32(s1, 2));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w3, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w3, 4)), e8);
                f3 = vfmaq_n_f32(f3, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, blo, bhi)), vgetq_lane_f32(s1, 3));
            }
            for (; b < nb; ++b) {
                const uint8_t* gq = pg + (size_t) b * 64;
                const int8_t* p = q + (size_t) b * 32;
                const int8x16_t alo = vld1q_s8(p), ahi = vld1q_s8(p + 16);
                const float32x4_t sc = vmulq_n_f32(vld1q_f32(sg + (size_t) b * 4), d[b]);
                uint8x16_t q0 = vld1q_u8(gq), q1 = vld1q_u8(gq + 16);
                uint8x16_t q2 = vld1q_u8(gq + 32), q3 = vld1q_u8(gq + 48);
                int8x16_t lo, hi;
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q0, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), e8);
                f0 = vfmaq_n_f32(f0, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), vgetq_lane_f32(sc, 0));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q1, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)), e8);
                f1 = vfmaq_n_f32(f1, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), vgetq_lane_f32(sc, 1));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q2, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q2, 4)), e8);
                f2 = vfmaq_n_f32(f2, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), vgetq_lane_f32(sc, 2));
                lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3, m4)), e8);
                hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q3, 4)), e8);
                f3 = vfmaq_n_f32(f3, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), vgetq_lane_f32(sc, 3));
            }
            C[i + 0] = vaddvq_f32(f0); C[i + 1] = vaddvq_f32(f1);
            C[i + 2] = vaddvq_f32(f2); C[i + 3] = vaddvq_f32(f3);
        } else {
            /* a worker's row range need not start or end on a group boundary */
            for (int r = r0; r < r1; ++r) {
                float32x4_t f = vdupq_n_f32(0);
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* gq = pg + (size_t) b * 64 + (size_t) r * 16;
                    const int8_t* p = q + (size_t) b * 32;
                    uint8x16_t qs = vld1q_u8(gq);
                    int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qs, m4)), e8);
                    int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qs, 4)), e8);
                    f = vfmaq_n_f32(f, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, vld1q_s8(p), vld1q_s8(p + 16))),
                                    sg[(size_t) b * 4 + r] * d[b]);
                }
                C[g * 4 + r] = vaddvq_f32(f);
            }
        }
        i = g * 4 + r1;
    }
}

#define JAM_BLK     jam_q8_blk
#define JAM_DECODE  jam_decode_q8_0_neon
#define JAM_MM_NAME jam_mm_q8_0_dotprod
#include "jam_gemm_neon.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_q4_0_blk
#define JAM_DECODE  jam_decode_q4_0_neon
#define JAM_MM_NAME jam_mm_q4_0_dotprod
#include "jam_gemm_neon.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_mxfp4_blk
#define JAM_DECODE  jam_decode_mxfp4_neon
#define JAM_MM_NAME jam_mm_mxfp4_dotprod
#include "jam_gemm_neon.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

/* ---- F16 / BF16 dense @ F32 (ARM) ----
 * Widen the weight row to f32 in registers and FMA against the f32 activations; k % 16 == 0
 * (gate in jam.c). n==1 is a 4-row shared-activation GEMV (four independent widen+fma chains);
 * n>1 tiles one row x two columns so each widened weight chunk is reused. */
static inline float32x4_t jam_f16w_lo(uint16x8_t w) { return vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(w))); }
static inline float32x4_t jam_f16w_hi(uint16x8_t w) { return vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(w))); }
static inline float32x4_t jam_bf16w_lo(uint16x8_t w) { return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w), 16)); }
static inline float32x4_t jam_bf16w_hi(uint16x8_t w) { return vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w), 16)); }

#define JAM_DENSE16_BODY(NAME, WLO, WHI)                                                              \
void NAME(void* arg, int rb, int re, int tid) {                                                       \
    (void) tid;                                                                                       \
    const jam_mm_job* J = (const jam_mm_job*) arg;                                                    \
    const uint16_t* W = (const uint16_t*) J->a;                                                       \
    const float* A = (const float*) J->b;                                                             \
    float* C = (float*) J->c;                                                                         \
    const int n = J->n, k = J->k, ldw = J->lda, ldb = J->ldb, ldc = J->ldc;                           \
    int i = rb;                                                                                       \
    if (n == 1) {                                                                                     \
        for (; i + 4 <= re; i += 4) {                                                                 \
            const uint16_t* w0 = W + (size_t)(i + 0) * ldw;                                           \
            const uint16_t* w1 = W + (size_t)(i + 1) * ldw;                                           \
            const uint16_t* w2 = W + (size_t)(i + 2) * ldw;                                           \
            const uint16_t* w3 = W + (size_t)(i + 3) * ldw;                                           \
            float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0);                                     \
            float32x4_t f2 = vdupq_n_f32(0), f3 = vdupq_n_f32(0);                                     \
            for (int e = 0; e < k; e += 8) {                                                          \
                const float32x4_t a0 = vld1q_f32(A + e), a1 = vld1q_f32(A + e + 4);                   \
                uint16x8_t u;                                                                         \
                u = vld1q_u16(w0 + e); f0 = vfmaq_f32(vfmaq_f32(f0, WLO(u), a0), WHI(u), a1);         \
                u = vld1q_u16(w1 + e); f1 = vfmaq_f32(vfmaq_f32(f1, WLO(u), a0), WHI(u), a1);         \
                u = vld1q_u16(w2 + e); f2 = vfmaq_f32(vfmaq_f32(f2, WLO(u), a0), WHI(u), a1);         \
                u = vld1q_u16(w3 + e); f3 = vfmaq_f32(vfmaq_f32(f3, WLO(u), a0), WHI(u), a1);         \
            }                                                                                         \
            C[i + 0] = vaddvq_f32(f0); C[i + 1] = vaddvq_f32(f1);                                     \
            C[i + 2] = vaddvq_f32(f2); C[i + 3] = vaddvq_f32(f3);                                     \
        }                                                                                             \
    }                                                                                                 \
    for (; i < re; ++i) {                                                                             \
        const uint16_t* w = W + (size_t) i * ldw;                                                     \
        int j = 0;                                                                                    \
        for (; j + 2 <= n; j += 2) {                                                                  \
            const float* b0 = A + (size_t) j * ldb;                                                   \
            const float* b1 = b0 + ldb;                                                               \
            float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0);                                     \
            for (int e = 0; e < k; e += 8) {                                                          \
                uint16x8_t u = vld1q_u16(w + e);                                                      \
                const float32x4_t wl = WLO(u), wh = WHI(u);                                           \
                f0 = vfmaq_f32(vfmaq_f32(f0, wl, vld1q_f32(b0 + e)), wh, vld1q_f32(b0 + e + 4));      \
                f1 = vfmaq_f32(vfmaq_f32(f1, wl, vld1q_f32(b1 + e)), wh, vld1q_f32(b1 + e + 4));      \
            }                                                                                         \
            C[(size_t)(j + 0) * ldc + i] = vaddvq_f32(f0);                                            \
            C[(size_t)(j + 1) * ldc + i] = vaddvq_f32(f1);                                            \
        }                                                                                             \
        for (; j < n; ++j) {                                                                          \
            const float* b0 = A + (size_t) j * ldb;                                                   \
            float32x4_t f = vdupq_n_f32(0);                                                           \
            for (int e = 0; e < k; e += 8) {                                                          \
                uint16x8_t u = vld1q_u16(w + e);                                                      \
                f = vfmaq_f32(vfmaq_f32(f, WLO(u), vld1q_f32(b0 + e)), WHI(u), vld1q_f32(b0 + e + 4)); \
            }                                                                                         \
            C[(size_t) j * ldc + i] = vaddvq_f32(f);                                                  \
        }                                                                                             \
    }                                                                                                 \
}

JAM_DENSE16_BODY(jam_mm_f16_neon, jam_f16w_lo, jam_f16w_hi)
JAM_DENSE16_BODY(jam_mm_bf16_neon, jam_bf16w_lo, jam_bf16w_hi)
