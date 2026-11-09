/* NEON-baseline kernel for the low-bit codes formats (vmull + vpadal int8 dot). The shared body is in
 * jam_gemm_lowbit_neon.inc; DOTPROD gets an sdot variant in jam_kernels_lowbit_dotprod.c. */
#include "jam_internal.h"
#include <arm_neon.h>

/* acc += a 16-element signed int8 dot, in 4 int32 lanes (widening multiply + pairwise accumulate). */
static inline int32x4_t jam_lowbit_dot16_neon(int32x4_t acc, int8x16_t w, int8x16_t a) {
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w),  vget_low_s8(a)));
    return vpadalq_s16(acc, vmull_s8(vget_high_s8(w), vget_high_s8(a)));
}

#define JAM_LOWBIT_DOT16 jam_lowbit_dot16_neon
#define JAM_MM_NAME      jam_mm_lowbit_neon
#include "jam_gemm_lowbit_neon.inc"
