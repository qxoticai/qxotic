/* SSSE3 (Core 2 2006+, still pre-AVX2) 128-bit int8 GEMM — Q8_0 / Q4_0 via the maddubs sign-trick, the
 * faster sibling of the true-SSE3 floor (jam_kernels_sse3.c). SSSE3 adds pmaddubsw/pabsb/psignb, so the
 * int8 dot drops the sign-extend (halving its op count); still no F16C (software fp16 in the decoders).
 * Built with -mssse3, dispatched when the CPU has SSSE3 but not AVX2. Reuses the SSE3 128-bit decoders. */
#include <tmmintrin.h>   /* SSSE3 (_mm_maddubs_epi16, _mm_sign_epi8, _mm_abs_epi8); pulls in SSE..SSE3 */
#include <stdint.h>
#include "jam_internal.h"
#include "jam_decode_x86_128.h"

#define JAM_BLK     jam_q8_blk
#define JAM_DECODE  jam_decode_q8_0_128
#define JAM_MM_NAME jam_mm_q8_0_ssse3
#include "jam_gemm_q128_ssse3.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_q4_0_blk
#define JAM_DECODE  jam_decode_q4_0_128
#define JAM_MM_NAME jam_mm_q4_0_ssse3
#include "jam_gemm_q128_ssse3.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME
