/* DOTPROD NVFP4 kernel: the 16-element int8 dot is a single vdotq_s32 (sdot) - the int8 workhorse on
 * modern ARM (Apple M-series, Graviton2+). Shares the body in jam_gemm_nvfp4_neon.inc. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"   /* jam_nvfp4.h, jam_dot16_dotprod */

#define JAM_NVFP4_DOT16 jam_dot16_dotprod
#define JAM_MM_NAME     jam_mm_nvfp4_dotprod
#include "jam_gemm_nvfp4_neon.inc"
