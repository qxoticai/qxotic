/* Shared NEON Q8_0 (weight) @ F32 (activation) -> F32 matmul body. ARM has a native SIGNED int8 dot, so
 * no +128 / abs-sign trick: the block-dot is exact and we just scale + accumulate in float (reduced once
 * per output). The includer defines, BEFORE #include:
 *     JAM_Q8_BLKDOT(wlo,whi,blo,bhi) -> int32x4_t   signed dot of one 32-elem block (4 partial lanes)
 *     JAM_Q8_MM_NAME                                 the exported kernel symbol
 * and has included <arm_neon.h>, <string.h>, "jam_internal.h". NEON-baseline supplies vmull+vpadal;
 * DOTPROD supplies vdotq_s32. Tiles 4 activation columns to reuse each weight-block load. */

typedef struct __attribute__((packed)) { uint16_t d; int8_t qs[32]; } jam_q8_blk;

static inline float jam_q8_h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float) x; }  /* fp16->f32 (FCVT) */

void JAM_Q8_MM_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* A = (const char*) J->a;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, nb = J->nb, k = J->k;
    const size_t wrow = (size_t)(J->lda / 32);
    for (int i = rb; i < re; ++i) {
        const jam_q8_blk* arow = (const jam_q8_blk*) (A + (size_t) i * wrow * sizeof(jam_q8_blk));
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            const int8_t *q0=J->aq+(size_t)(j+0)*k,*q1=J->aq+(size_t)(j+1)*k,*q2=J->aq+(size_t)(j+2)*k,*q3=J->aq+(size_t)(j+3)*k;
            const float  *d0=J->ad+(size_t)(j+0)*nb,*d1=J->ad+(size_t)(j+1)*nb,*d2=J->ad+(size_t)(j+2)*nb,*d3=J->ad+(size_t)(j+3)*nb;
            float32x4_t f0=vdupq_n_f32(0),f1=vdupq_n_f32(0),f2=vdupq_n_f32(0),f3=vdupq_n_f32(0);
            for (int blk = 0; blk < nb; ++blk) {
                const jam_q8_blk* wb = &arow[blk];
                int8x16_t wlo = vld1q_s8(wb->qs), whi = vld1q_s8(wb->qs + 16);
                float dA = jam_q8_h2f(wb->d);
                const int8_t* p;
                p=q0+(size_t)blk*32; f0=vfmaq_n_f32(f0, vcvtq_f32_s32(JAM_Q8_BLKDOT(wlo,whi,vld1q_s8(p),vld1q_s8(p+16))), dA*d0[blk]);
                p=q1+(size_t)blk*32; f1=vfmaq_n_f32(f1, vcvtq_f32_s32(JAM_Q8_BLKDOT(wlo,whi,vld1q_s8(p),vld1q_s8(p+16))), dA*d1[blk]);
                p=q2+(size_t)blk*32; f2=vfmaq_n_f32(f2, vcvtq_f32_s32(JAM_Q8_BLKDOT(wlo,whi,vld1q_s8(p),vld1q_s8(p+16))), dA*d2[blk]);
                p=q3+(size_t)blk*32; f3=vfmaq_n_f32(f3, vcvtq_f32_s32(JAM_Q8_BLKDOT(wlo,whi,vld1q_s8(p),vld1q_s8(p+16))), dA*d3[blk]);
            }
            C[(size_t)(j+0)*ldc+i]=vaddvq_f32(f0); C[(size_t)(j+1)*ldc+i]=vaddvq_f32(f1);
            C[(size_t)(j+2)*ldc+i]=vaddvq_f32(f2); C[(size_t)(j+3)*ldc+i]=vaddvq_f32(f3);
        }
        for (; j < n; ++j) {                                  /* column tail */
            const int8_t* qj=J->aq+(size_t)j*k; const float* dj=J->ad+(size_t)j*nb;
            float32x4_t f = vdupq_n_f32(0);
            for (int blk = 0; blk < nb; ++blk) {
                const jam_q8_blk* wb = &arow[blk];
                const int8_t* p = qj+(size_t)blk*32;
                f = vfmaq_n_f32(f, vcvtq_f32_s32(JAM_Q8_BLKDOT(vld1q_s8(wb->qs),vld1q_s8(wb->qs+16),vld1q_s8(p),vld1q_s8(p+16))),
                                jam_q8_h2f(wb->d)*dj[blk]);
            }
            C[(size_t)j*ldc+i] = vaddvq_f32(f);
        }
    }
}
