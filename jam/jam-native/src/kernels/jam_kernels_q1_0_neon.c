/* NEON-baseline Q1_0 kernel (vmull + vpadal int8 dot). The shared body is in jam_gemm_q1_0_neon.inc;
 * DOTPROD gets an sdot variant in jam_kernels_q1_0_dotprod.c. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_fp16.h"

/* 16-element signed int8 dot -> int (widening multiply + pairwise accumulate). */
static inline int jam_q1_0_dot16_neon(int8x16_t w, int8x16_t a) {
    int32x4_t d = vdupq_n_s32(0);
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(w),  vget_low_s8(a)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(w), vget_high_s8(a)));
    return (int) vaddvq_s32(d);
}

#define JAM_Q1_0_DOT16 jam_q1_0_dot16_neon
#define JAM_MM_NAME    jam_mm_q1_0_neon
#include "jam_gemm_q1_0_neon.inc"
