/* NEON-baseline Q8_0 @ F32 -> F32 (this TU built with -march=armv8-a). The int8 block-dot uses widening
 * multiply (vmull_s8) + pairwise-accumulate (vpadalq_s16) — the fallback for ARMv8.0/8.1 cores without
 * the dot-product extension. DOTPROD/I8MM cores get faster kernels (jam_kernels_dotprod/i8mm.c). */
#include "jam_internal.h"
#include <string.h>
#include <arm_neon.h>

static inline int32x4_t jam_blkdot_neon(int8x16_t wlo, int8x16_t whi, int8x16_t blo, int8x16_t bhi) {
    int32x4_t d = vdupq_n_s32(0);
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(wlo),  vget_low_s8(blo)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(wlo), vget_high_s8(blo)));
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(whi),  vget_low_s8(bhi)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(whi), vget_high_s8(bhi)));
    return d;
}

#define JAM_Q8_BLKDOT(wlo,whi,blo,bhi) jam_blkdot_neon(wlo,whi,blo,bhi)
#define JAM_Q8_MM_NAME jam_mm_q8_0_neon
#include "jam_q8_neon.inc"
