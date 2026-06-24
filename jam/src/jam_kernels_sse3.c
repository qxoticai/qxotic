/* SSE3 (true SSE3: no SSSE3 maddubs, no F16C) 128-bit int8 GEMM — the pre-AVX2 x86 floor, replacing the
 * generic dequant-to-float path on machines without AVX2. Built with -msse3, dispatched when the CPU has
 * SSE3 but not AVX2. Q8_0 only for now (the q128 engine + a 128-bit decoder generalize to other quants). */
#include <pmmintrin.h>
#include <stdint.h>
#include "jam_internal.h"
#include "jam_decode_x86_128.h"

#define JAM_BLK     jam_q8_blk
#define JAM_DECODE  jam_decode_q8_0_128
#define JAM_MM_NAME jam_mm_q8_0_sse3
#include "jam_gemm_q128.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_q4_0_blk
#define JAM_DECODE  jam_decode_q4_0_128
#define JAM_MM_NAME jam_mm_q4_0_sse3
#include "jam_gemm_q128.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_mxfp4_blk
#define JAM_DECODE  jam_decode_mxfp4_128
#define JAM_MM_NAME jam_mm_mxfp4_sse3
#include "jam_gemm_q128.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME
