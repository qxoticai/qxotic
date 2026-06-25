/* NEON NVFP4 kernel — GGUF block_nvfp4 ({d[4] UE4M3; qs[32]}, 64 elems = 4 sub-blocks of 16, no global
 * scale). Decode = MXFP4's vqtbl1q LUT; the sub-block nibble order (byte s*8+j: low=elem j, high=elem j+8)
 * is reassembled per sub-block via vcombine. Each 16-element sub-block is dotted against the int8-requantized
 * activations and scaled by its UE4M3 scale. Baseline NEON (vmull+vpadal), bound for all aarch64 tiers. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_nvfp4.h"   /* jam_ue4m3_to_float; JAM_MXFP4_CODES via jam_mxfp4.h */

/* 16-element signed int8 dot -> int. */
static inline int jam_nvfp4_dot16(int8x16_t w, int8x16_t a) {
    int32x4_t d = vdupq_n_s32(0);
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(w),  vget_low_s8(a)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(w), vget_high_s8(a)));
    return (int) vaddvq_s32(d);
}

void jam_mm_nvfp4_neon(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k;
    const int nblk = k / JAM_NVFP4_QK;
    const size_t wrow = (size_t) nblk * sizeof(jam_nvfp4_blk);
    static const int8_t codes[16] = { JAM_MXFP4_CODES };
    const int8x16_t lut = vld1q_s8(codes);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int i = rb; i < re; ++i) {
        const jam_nvfp4_blk* wr = (const jam_nvfp4_blk*) (W + (size_t) i * wrow);
        for (int j = 0; j < n; ++j) {
            const int8_t* aq = AQ + (size_t) j * k;
            const float* ad = AD + (size_t) j * (k / 32);
            float acc = 0.0f;
            for (int bb = 0; bb < nblk; ++bb) {
                const jam_nvfp4_blk* w = &wr[bb];
                for (int sp = 0; sp < 2; ++sp) {              /* 2 spans of 32 = sub-blocks (2sp, 2sp+1) */
                    uint8x16_t qs = vld1q_u8(w->qs + sp * 16);
                    int8x16_t lo = vqtbl1q_s8(lut, vandq_u8(qs, m4));      /* [e0-7, e16-23] */
                    int8x16_t hi = vqtbl1q_s8(lut, vshrq_n_u8(qs, 4));     /* [e8-15, e24-31] */
                    int8x16_t wl = vcombine_s8(vget_low_s8(lo), vget_low_s8(hi));    /* elems 0-15  */
                    int8x16_t wh = vcombine_s8(vget_high_s8(lo), vget_high_s8(hi));  /* elems 16-31 */
                    int blk32 = bb * 2 + sp;
                    const int8_t* a = aq + (size_t) blk32 * 32;
                    int d0 = jam_nvfp4_dot16(wl, vld1q_s8(a));
                    int d1 = jam_nvfp4_dot16(wh, vld1q_s8(a + 16));
                    float s0 = jam_ue4m3_to_float(w->d[2*sp]), s1 = jam_ue4m3_to_float(w->d[2*sp + 1]);
                    acc += ad[blk32] * (s0 * (float) d0 + s1 * (float) d1);
                }
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}
