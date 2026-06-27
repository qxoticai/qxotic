/* DOTPROD simple-block kernels (Q8_0/Q4_0/MXFP4 @ F32 -> F32; built -march=armv8.2-a+dotprod). The int8
 * block-dot is two vdotq_s32 (signed 8-bit dot-product, 4 elems/lane) — the int8 workhorse on modern ARM
 * (all Apple M-series, Graviton2+, recent phones). Decode per quant in jam_decode_neon.h; engine shared. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"

/* vdotq_s32(acc,a,b): acc[l] += Σ_{4} a[4l+..]·b[4l+..]. Two of them cover the 32-elem block -> int32x4. */
#define JAM_BLKDOT(wlo,whi,blo,bhi) vdotq_s32(vdotq_s32(vdupq_n_s32(0), wlo, blo), whi, bhi)

#define JAM_BLK     jam_q8_blk
#define JAM_DECODE  jam_decode_q8_0_neon
#define JAM_MM_NAME jam_mm_q8_0_dotprod
#include "jam_gemm_neon.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_q4_0_blk
#define JAM_DECODE  jam_decode_q4_0_neon
#define JAM_MM_NAME jam_mm_q4_0_dotprod
#include "jam_gemm_neon.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_mxfp4_blk
#define JAM_DECODE  jam_decode_mxfp4_neon
#define JAM_MM_NAME jam_mm_mxfp4_dotprod
#include "jam_gemm_neon.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME
