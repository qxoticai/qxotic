/* I8MM block-quant @ F32 -> F32 (this TU built with -mcpu=generic+dotprod+i8mm). SMMLA (vmmlaq_s32)
 * computes a 2x2 block of 8-deep signed int8 dots. The production kernels compose those instructions into
 * direct-layout 4x4 Q8_0/Q4_0/Q5_0 tiles. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"   /* Q8_0/Q4_0/Q5_0 blocks and decoders */

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

/* The production direct-layout 4x4 kernels.  Q8 is load-only; Q4 and Q5 decode ordinary GGUF nibbles
 * (and Q5's high bits) in registers.  All share the same SMMLA tiling and complete 2x2/1x1 tails without
 * scalar int8 loops. */
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

#define JAM_BLK        jam_q5_0_blk
#define JAM_DECODE     jam_decode_q5_0_neon
#define JAM_EDGE_NAME  jam_q5_i8mm_edge
#define JAM_TILE2_NAME jam_q5_i8mm_tile2
#define JAM_MM_NAME    jam_mm_q5_0_i8mm_4x4
#include "jam_gemm_i8mm_4x4.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_EDGE_NAME
#undef JAM_TILE2_NAME
#undef JAM_MM_NAME


/* The low-bit codes formats (jam_lowbit.h: Q2_0, PQ2_0, PTQ1_0) on I8MM. Prefill decodes each 4-row
 * group once into Q8_0 blocks - the codes are small ints and the source's fp16 scale carries over bit
 * for bit, so the conversion is exact - and runs the Q8_0 SMMLA tile on them. Decode (n == 1) keeps
 * the SDOT kernel, as Q8_0 itself does on I8MM. */
#include "jam_lowbit.h"
void jam_mm_lowbit_dotprod(void* job, int rb, int re, int tid);   /* every I8MM CPU has DOTPROD */
void jam_mm_lowbit_i8mm(void* arg, int rb, int re, int tid) {
    const jam_q8_job* J = (const jam_q8_job*) arg;
    if (J->n == 1) { jam_mm_lowbit_dotprod(arg, rb, re, tid); return; }
    const int k = J->k, nb32 = k / 32, qk = jam_lowbit_qk(J->wt), bytes = jam_lowbit_bytes(J->wt);
    const size_t wrow = (size_t) (J->lda / qk) * bytes;
    jam_q8_blk* q8 = (jam_q8_blk*) J->repack[tid].qs;        /* 4 rows of Q8_0 blocks */
    /* the job's fields as locals: the stores into q8 could alias *J, and the block loop would reload them */
    const uint8_t* const W = (const uint8_t*) J->a;
    const int ptq = J->wt == JAM_PTQ1_0;
    jam_q8_job group = *J;
    group.a = q8; group.lda = k;
    for (int r = rb; r < re; r += 4) {
        const int nr = re - r < 4 ? re - r : 4;
        for (int t = 0; t < nr; ++t) {
            const uint8_t* w = W + (size_t) (r + t) * wrow;
            for (int b = 0; b < k / qk; b++, w += bytes) {
                int8_t codes[128];
                uint16_t d;
                memcpy(&d, ptq ? w + JAM_PTQ1_0_D : w, 2);
                if (ptq) jam_ptq1_0_codes(w, codes);
                else     jam_q2_codes(w, qk, codes);
                jam_q8_blk* o = q8 + (size_t) t * nb32 + (size_t) b * (qk / 32);
                for (int s = 0; s < qk / 32; s++) { o[s].d = d; memcpy(o[s].qs, codes + 32 * s, 32); }
            }
        }
        group.c = (float*) J->c + r;                   /* C is token-major: rows r.. are columns r.. */
        jam_mm_q8_0_i8mm_4x4(&group, 0, nr, tid);
    }
}
