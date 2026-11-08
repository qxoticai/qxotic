/* NEON-baseline Q1_0 kernel (vmull + vpadal int8 dot). The shared body is in jam_gemm_q1_0_neon.inc;
 * DOTPROD gets an sdot variant in jam_kernels_q1_0_dotprod.c. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"   /* jam_q1_0.h, jam_neon_half_at, jam_dot16_neon */

#define JAM_Q1_0_DOT16 jam_dot16_neon
#define JAM_MM_NAME    jam_mm_q1_0_neon
#include "jam_gemm_q1_0_neon.inc"
