/* NEON-baseline NVFP4 kernel (vmull + vpadal int8 dot). The shared body is in jam_gemm_nvfp4_neon.inc;
 * DOTPROD gets an sdot variant in jam_kernels_nvfp4_dotprod.c. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_nvfp4.h"   /* jam_nvfp4_blk, jam_ue4m3_to_float, JAM_MXFP4_CODES */

/* 16-element signed int8 dot -> int (widening multiply + pairwise accumulate). */
static inline int jam_nvfp4_dot16_neon(int8x16_t w, int8x16_t a) {
    int32x4_t d = vdupq_n_s32(0);
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(w),  vget_low_s8(a)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(w), vget_high_s8(a)));
    return (int) vaddvq_s32(d);
}

#define JAM_NVFP4_DOT16 jam_nvfp4_dot16_neon
#define JAM_MM_NAME     jam_mm_nvfp4_neon
#include "jam_gemm_nvfp4_neon.inc"
