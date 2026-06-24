/* NEON-baseline simple-block kernels (Q8_0/Q4_0/MXFP4 @ F32 -> F32; built -march=armv8-a). The int8
 * block-dot uses widening multiply (vmull_s8) + pairwise-accumulate (vpadalq_s16) — the fallback for
 * ARMv8.0/8.1 cores without the dot-product extension. The decode (per quant) lives in jam_decode_neon.h;
 * the shared engine in jam_gemm_neon.inc. DOTPROD/I8MM cores get faster dots (jam_kernels_dotprod/i8mm.c). */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"

static inline int32x4_t jam_blkdot_neon(int8x16_t wlo, int8x16_t whi, int8x16_t blo, int8x16_t bhi) {
    int32x4_t d = vdupq_n_s32(0);
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(wlo),  vget_low_s8(blo)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(wlo), vget_high_s8(blo)));
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(whi),  vget_low_s8(bhi)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(whi), vget_high_s8(bhi)));
    return d;
}
#define JAM_BLKDOT(wlo,whi,blo,bhi) jam_blkdot_neon(wlo,whi,blo,bhi)

#define JAM_BLK     jam_q8_blk
#define JAM_DECODE  jam_decode_q8_0_neon
#define JAM_MM_NAME jam_mm_q8_0_neon
#include "jam_gemm_neon.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_q4_0_blk
#define JAM_DECODE  jam_decode_q4_0_neon
#define JAM_MM_NAME jam_mm_q4_0_neon
#include "jam_gemm_neon.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_mxfp4_blk
#define JAM_DECODE  jam_decode_mxfp4_neon
#define JAM_MM_NAME jam_mm_mxfp4_neon
#include "jam_gemm_neon.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME
