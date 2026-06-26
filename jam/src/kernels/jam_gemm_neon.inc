/* Shared NEON simple-block (32-element) weight @ F32 -> F32 matmul body — the ARM analog of jam_gemm_q256.inc.
 * Decode each weight block to two signed int8x16 halves + a float scale, signed-dot it against the int8-
 * requantized activation, scale-accumulate in float (reduced once per output). The includer defines, before
 * #include:
 *     JAM_BLK                                   weight block struct (row stride = (k/32)*sizeof)   [per QUANT]
 *     JAM_DECODE(blkptr, &wlo, &whi, &dA)       decode one 32-block -> int8x16 wlo,whi + float scale [per QUANT]
 *     JAM_BLKDOT(wlo,whi,blo,bhi) -> int32x4_t  signed dot of one 32-elem block (4 partial lanes)   [per ISA]
 *     JAM_MM_NAME                               exported symbol
 * Requires <arm_neon.h>, "jam_internal.h", "jam_decode_neon.h". Tiles 4 activation columns, decoding each
 * weight block ONCE; activations are pre-requantized to int8 (J->aq) + per-32 scales (J->ad). */
void JAM_MM_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, nb = J->nb, k = J->k;
    const size_t wrow = (size_t)(J->lda / 32);
    for (int i = rb; i < re; ++i) {
        const JAM_BLK* arow = (const JAM_BLK*) (W + (size_t) i * wrow * sizeof(JAM_BLK));
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            const int8_t *q0=J->aq+(size_t)(j+0)*k,*q1=J->aq+(size_t)(j+1)*k,*q2=J->aq+(size_t)(j+2)*k,*q3=J->aq+(size_t)(j+3)*k;
            const float  *d0=J->ad+(size_t)(j+0)*nb,*d1=J->ad+(size_t)(j+1)*nb,*d2=J->ad+(size_t)(j+2)*nb,*d3=J->ad+(size_t)(j+3)*nb;
            float32x4_t f0=vdupq_n_f32(0),f1=vdupq_n_f32(0),f2=vdupq_n_f32(0),f3=vdupq_n_f32(0);
            for (int blk = 0; blk < nb; ++blk) {
                int8x16_t wlo, whi; float dA; JAM_DECODE(&arow[blk], &wlo, &whi, &dA);   /* decode ONCE */
                const int8_t* p;
                p=q0+(size_t)blk*32; f0=vfmaq_n_f32(f0, vcvtq_f32_s32(JAM_BLKDOT(wlo,whi,vld1q_s8(p),vld1q_s8(p+16))), dA*d0[blk]);
                p=q1+(size_t)blk*32; f1=vfmaq_n_f32(f1, vcvtq_f32_s32(JAM_BLKDOT(wlo,whi,vld1q_s8(p),vld1q_s8(p+16))), dA*d1[blk]);
                p=q2+(size_t)blk*32; f2=vfmaq_n_f32(f2, vcvtq_f32_s32(JAM_BLKDOT(wlo,whi,vld1q_s8(p),vld1q_s8(p+16))), dA*d2[blk]);
                p=q3+(size_t)blk*32; f3=vfmaq_n_f32(f3, vcvtq_f32_s32(JAM_BLKDOT(wlo,whi,vld1q_s8(p),vld1q_s8(p+16))), dA*d3[blk]);
            }
            C[(size_t)(j+0)*ldc+i]=vaddvq_f32(f0); C[(size_t)(j+1)*ldc+i]=vaddvq_f32(f1);
            C[(size_t)(j+2)*ldc+i]=vaddvq_f32(f2); C[(size_t)(j+3)*ldc+i]=vaddvq_f32(f3);
        }
        for (; j < n; ++j) {                                  /* column tail */
            const int8_t* qj=J->aq+(size_t)j*k; const float* dj=J->ad+(size_t)j*nb;
            float32x4_t f = vdupq_n_f32(0);
            for (int blk = 0; blk < nb; ++blk) {
                int8x16_t wlo, whi; float dA; JAM_DECODE(&arow[blk], &wlo, &whi, &dA);
                const int8_t* p = qj+(size_t)blk*32;
                f = vfmaq_n_f32(f, vcvtq_f32_s32(JAM_BLKDOT(wlo,whi,vld1q_s8(p),vld1q_s8(p+16))), dA*dj[blk]);
            }
            C[(size_t)j*ldc+i] = vaddvq_f32(f);
        }
    }
}
