/* DOTPROD Q1_0 kernel: the 16-element int8 dot is a single vdotq_s32 (sdot). Shares the body in
 * jam_gemm_q1_0_neon.inc. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_fp16.h"

/* 16-element signed int8 dot -> int via one vdotq_s32 (4 int32 lanes, each of 4 products). */
static inline int jam_q1_0_dot16_dotprod(int8x16_t w, int8x16_t a) {
    return (int) vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, a));
}

#define JAM_Q1_0_DOT16 jam_q1_0_dot16_dotprod
#define JAM_MM_NAME    jam_mm_q1_0_dotprod
#include "jam_gemm_q1_0_neon.inc"
