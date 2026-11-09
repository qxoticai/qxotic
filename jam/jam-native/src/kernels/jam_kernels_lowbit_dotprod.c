/* DOTPROD kernel for the low-bit codes formats: the 16-element int8 dot is a single vdotq_s32 (sdot).
 * Shares the body in jam_gemm_lowbit_neon.inc. */
#include "jam_internal.h"
#include <arm_neon.h>

/* acc += a 16-element signed int8 dot via one vdotq_s32 (4 int32 lanes, each of 4 products). */
static inline int32x4_t jam_lowbit_dot16_dotprod(int32x4_t acc, int8x16_t w, int8x16_t a) {
    return vdotq_s32(acc, w, a);
}

#define JAM_LOWBIT_DOT16 jam_lowbit_dot16_dotprod
#define JAM_MM_NAME      jam_mm_lowbit_dotprod
#include "jam_gemm_lowbit_neon.inc"
