/* DOTPROD Q1_0 kernel: the 16-element int8 dot is a single vdotq_s32 (sdot). Shares the body in
 * jam_gemm_q1_0_neon.inc. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"   /* jam_q1_0.h, jam_neon_half_at, jam_dot16_dotprod */

#define JAM_Q1_0_DOT16 jam_dot16_dotprod
#define JAM_MM_NAME    jam_mm_q1_0_dotprod
#include "jam_gemm_q1_0_neon.inc"
