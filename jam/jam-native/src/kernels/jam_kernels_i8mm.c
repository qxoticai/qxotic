/* I8MM block-quant @ F32 -> F32 (this TU built with -march=armv8.6-a+i8mm). SMMLA (vmmlaq_s32) computes
 * a 2x2 block of 8-deep signed int8 dots. The production kernels compose those instructions into direct-
 * layout 4x4 Q8_0/Q4_0 tiles. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"   /* jam_q8_blk + jam_neon_h2f */

/* Accumulate two 16-element row vectors against two 16-element column vectors.  Each SMMLA consumes
 * eight K elements, hence the low/high halves.  Lane order is r0c0,r0c1,r1c0,r1c1. */
static inline int32x4_t jam_i8mm_pair_acc(int32x4_t acc,
                                          int8x16_t w0, int8x16_t w1,
                                          int8x16_t a0, int8x16_t a1) {
    acc = vmmlaq_s32(acc, vcombine_s8(vget_low_s8(w0), vget_low_s8(w1)),
                     vcombine_s8(vget_low_s8(a0), vget_low_s8(a1)));
    return vmmlaq_s32(acc, vcombine_s8(vget_high_s8(w0), vget_high_s8(w1)),
                      vcombine_s8(vget_high_s8(a0), vget_high_s8(a1)));
}

static inline void jam_i8mm_store_2x2(float* c, int ldc, int row, int col, float32x4_t tile) {
    c[(size_t)(col + 0) * ldc + row + 0] = vgetq_lane_f32(tile, 0);
    c[(size_t)(col + 1) * ldc + row + 0] = vgetq_lane_f32(tile, 1);
    c[(size_t)(col + 0) * ldc + row + 1] = vgetq_lane_f32(tile, 2);
    c[(size_t)(col + 1) * ldc + row + 1] = vgetq_lane_f32(tile, 3);
}

/* The production direct-layout 4x4 kernels.  Q8 is load-only; Q4 decodes ordinary GGUF nibbles in
 * registers.  Both share the same SMMLA tiling and complete 2x2/1x1 tails without scalar int8 loops. */
#define JAM_BLK        jam_q8_blk
#define JAM_DECODE     jam_decode_q8_0_neon
#define JAM_EDGE_NAME  jam_q8_i8mm_edge
#define JAM_TILE2_NAME jam_q8_i8mm_tile2
#define JAM_MM_NAME    jam_mm_q8_0_i8mm_4x4
#include "jam_gemm_i8mm_4x4.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_EDGE_NAME
#undef JAM_TILE2_NAME
#undef JAM_MM_NAME

#define JAM_BLK        jam_q4_0_blk
#define JAM_DECODE     jam_decode_q4_0_neon
#define JAM_EDGE_NAME  jam_q4_i8mm_edge
#define JAM_TILE2_NAME jam_q4_i8mm_tile2
#define JAM_MM_NAME    jam_mm_q4_0_i8mm_4x4
#include "jam_gemm_i8mm_4x4.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_EDGE_NAME
#undef JAM_TILE2_NAME
#undef JAM_MM_NAME

