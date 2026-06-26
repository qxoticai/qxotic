/* I8MM Q8_0 @ F32 -> F32 (this TU built with -march=armv8.6-a+i8mm). Uses SMMLA (vmmlaq_s32): a single
 * instruction does a 2x2 block of 8-deep signed int8 dots, so we tile 2 weight rows × 2 activation cols.
 * Highest int8 throughput on ARM (Graviton3/4, Apple M-series). Odd row/col edges fall back to a scalar
 * int dot. Same requant-B scratch + deferred-float scaling as the other Q8_0 kernels. */
#include "jam_internal.h"
#include <string.h>
#include <arm_neon.h>

typedef struct __attribute__((packed)) { uint16_t d; int8_t qs[32]; } jam_q8_blk;

static inline float i8mm_h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float) x; }

/* scalar int8 block-dot of a weight row and an activation column, scaled — for the odd edges. */
static inline float i8mm_edge(const jam_q8_blk* wr, const int8_t* qc, const float* dc, int nb) {
    float acc = 0;
    for (int b = 0; b < nb; ++b) {
        const jam_q8_blk* w = &wr[b]; int dot = 0;
        for (int e = 0; e < 32; ++e) dot += (int) w->qs[e] * (int) qc[(size_t)b*32+e];
        acc += i8mm_h2f(w->d) * dc[b] * (float) dot;
    }
    return acc;
}

void jam_mm_q8_0_i8mm(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* A = (const char*) J->a;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, nb = J->nb, k = J->k;
    const size_t wrow = (size_t)(J->lda / 32);
    #define AROW(i) ((const jam_q8_blk*)(A + (size_t)(i)*wrow*sizeof(jam_q8_blk)))

    int i = rb;
    for (; i + 2 <= re; i += 2) {                             /* 2 weight rows at a time */
        const jam_q8_blk* w0 = AROW(i); const jam_q8_blk* w1 = AROW(i+1);
        int j = 0;
        for (; j + 2 <= n; j += 2) {                          /* 2 activation cols at a time -> 2x2 SMMLA */
            const int8_t* a0 = J->aq+(size_t)(j+0)*k; const int8_t* a1 = J->aq+(size_t)(j+1)*k;
            const float*  e0 = J->ad+(size_t)(j+0)*nb; const float* e1 = J->ad+(size_t)(j+1)*nb;
            float32x4_t f = vdupq_n_f32(0);                   /* [c(i,j) c(i,j+1) c(i+1,j) c(i+1,j+1)] */
            for (int blk = 0; blk < nb; ++blk) {
                int32x4_t r = vdupq_n_s32(0);
                for (int ch = 0; ch < 32; ch += 8) {          /* SMMLA is 8-deep; 4 chunks per block */
                    int8x16_t va = vcombine_s8(vld1_s8(w0[blk].qs+ch), vld1_s8(w1[blk].qs+ch));   /* rows wi, wi+1 */
                    int8x16_t vb = vcombine_s8(vld1_s8(a0+(size_t)blk*32+ch), vld1_s8(a1+(size_t)blk*32+ch)); /* rows aj, aj+1 */
                    r = vmmlaq_s32(r, va, vb);                /* += [wi·aj wi·aj1 wi1·aj wi1·aj1] */
                }
                float dA0=i8mm_h2f(w0[blk].d), dA1=i8mm_h2f(w1[blk].d);
                float sc[4] = { dA0*e0[blk], dA0*e1[blk], dA1*e0[blk], dA1*e1[blk] };
                f = vfmaq_f32(f, vcvtq_f32_s32(r), vld1q_f32(sc));
            }
            C[(size_t)(j+0)*ldc+i]     = vgetq_lane_f32(f,0); C[(size_t)(j+1)*ldc+i]     = vgetq_lane_f32(f,1);
            C[(size_t)(j+0)*ldc+(i+1)] = vgetq_lane_f32(f,2); C[(size_t)(j+1)*ldc+(i+1)] = vgetq_lane_f32(f,3);
        }
        for (; j < n; ++j) {                                  /* odd column: both rows, scalar */
            C[(size_t)j*ldc+i]     = i8mm_edge(w0, J->aq+(size_t)j*k, J->ad+(size_t)j*nb, nb);
            C[(size_t)j*ldc+(i+1)] = i8mm_edge(w1, J->aq+(size_t)j*k, J->ad+(size_t)j*nb, nb);
        }
    }
    for (; i < re; ++i)                                       /* odd weight row: scalar over all cols */
        for (int j = 0; j < n; ++j)
            C[(size_t)j*ldc+i] = i8mm_edge(AROW(i), J->aq+(size_t)j*k, J->ad+(size_t)j*nb, nb);
    #undef AROW
}
