/* NEON-baseline NVFP4 kernel (vmull + vpadal int8 dot). The shared body is in jam_gemm_nvfp4_neon.inc;
 * DOTPROD gets an sdot variant in jam_kernels_nvfp4_dotprod.c. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"   /* jam_nvfp4.h (jam_nvfp4_blk, jam_ue4m3_lut, JAM_MXFP4_CODES), jam_dot16_neon */

#define JAM_NVFP4_DOT16 jam_dot16_neon
#define JAM_MM_NAME     jam_mm_nvfp4_neon
#include "jam_gemm_nvfp4_neon.inc"
