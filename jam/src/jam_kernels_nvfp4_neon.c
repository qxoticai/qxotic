/* NEON NVFP4 kernel (E2M1 nibbles + a per-16 E4M3 scale, PLANAR layout). Decode = MXFP4's vqtbl1q LUT into
 * two contiguous 16-element halves; each half is dotted against the int8-requantized activations and scaled
 * by its E4M3 block scale. Baseline NEON int8 dot (vmull + vpadal) — bound for all aarch64 tiers for now
 * (DOTPROD/I8MM can add an sdot/mmla variant later). The per-tensor FP32 global scale G is the caller's. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_nvfp4.h"   /* jam_nvfp4_dhalf; JAM_MXFP4_CODES via jam_mxfp4.h */

/* 16-element signed int8 dot -> int (widening multiply + pairwise accumulate). */
static inline int jam_nvfp4_dot16(int8x16_t w, int8x16_t a) {
    int32x4_t d = vdupq_n_s32(0);
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(w),  vget_low_s8(a)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(w), vget_high_s8(a)));
    return (int) vaddvq_s32(d);
}

void jam_mm_nvfp4_neon(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* DATA  = (const uint8_t*) J->a;
    const uint8_t* SCALE = (const uint8_t*) J->wscale;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;   /* nb = k/32 */
    const size_t drow = (size_t) nb * 16, srow = (size_t) nb * 2;
    static const int8_t codes[16] = { JAM_MXFP4_CODES };
    const int8x16_t lut = vld1q_s8(codes);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int i = rb; i < re; ++i) {
        const uint8_t* dr = DATA + (size_t) i * drow;
        const uint8_t* sr = SCALE + (size_t) i * srow;
        for (int j = 0; j < n; ++j) {
            const int8_t* aq = AQ + (size_t) j * k;
            const float* ad = AD + (size_t) j * nb;
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                uint8x16_t qs = vld1q_u8(dr + (size_t) b * 16);
                int8x16_t wlo = vqtbl1q_s8(lut, vandq_u8(qs, m4));     /* elements 0-15 */
                int8x16_t whi = vqtbl1q_s8(lut, vshrq_n_u8(qs, 4));    /* elements 16-31 */
                const int8_t* a = aq + (size_t) b * 32;
                int dlo = jam_nvfp4_dot16(wlo, vld1q_s8(a));
                int dhi = jam_nvfp4_dot16(whi, vld1q_s8(a + 16));
                float s0 = jam_nvfp4_dhalf(sr[2*b]), s1 = jam_nvfp4_dhalf(sr[2*b + 1]);
                acc += ad[b] * (s0 * (float) dlo + s1 * (float) dhi);
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}
