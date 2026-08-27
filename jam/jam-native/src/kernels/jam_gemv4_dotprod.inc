/* Direct-layout 4-row x 1-column SDOT GEMV for 32-element block quants.
 *
 * The activation block is shared by four independent weight rows.  Besides cutting activation loads
 * and scale loads by 4x, this gives the core four independent accumulator chains to hide SDOT/load
 * latency.  Weights stay in their original GGUF layout; there is no packing or persistent scratch.
 *
 * The includer defines JAM_BLK, JAM_DECODE and JAM_GEMV_NAME. */
void JAM_GEMV_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    float* C = (float*) J->c;
    const int nb = J->nb;
    const size_t wrow = (size_t)(J->lda / 32);
    const int8_t* q = J->aq;
    const float* d = J->ad;

    int i = rb;
    for (; i + 4 <= re; i += 4) {
        const JAM_BLK* w0 = (const JAM_BLK*)(W + (size_t)(i + 0) * wrow * sizeof(JAM_BLK));
        const JAM_BLK* w1 = (const JAM_BLK*)(W + (size_t)(i + 1) * wrow * sizeof(JAM_BLK));
        const JAM_BLK* w2 = (const JAM_BLK*)(W + (size_t)(i + 2) * wrow * sizeof(JAM_BLK));
        const JAM_BLK* w3 = (const JAM_BLK*)(W + (size_t)(i + 3) * wrow * sizeof(JAM_BLK));
        float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0);
        float32x4_t f2 = vdupq_n_f32(0), f3 = vdupq_n_f32(0);

        for (int blk = 0; blk < nb; ++blk) {
            const int8_t* p = q + (size_t)blk * 32;
            const int8x16_t alo = vld1q_s8(p), ahi = vld1q_s8(p + 16);
            const float db = d[blk];
            int8x16_t lo, hi;
            float dw;

            JAM_DECODE(&w0[blk], &lo, &hi, &dw);
            f0 = vfmaq_n_f32(f0, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), dw * db);
            JAM_DECODE(&w1[blk], &lo, &hi, &dw);
            f1 = vfmaq_n_f32(f1, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), dw * db);
            JAM_DECODE(&w2[blk], &lo, &hi, &dw);
            f2 = vfmaq_n_f32(f2, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), dw * db);
            JAM_DECODE(&w3[blk], &lo, &hi, &dw);
            f3 = vfmaq_n_f32(f3, vcvtq_f32_s32(JAM_BLKDOT(lo, hi, alo, ahi)), dw * db);
        }
        C[i + 0] = vaddvq_f32(f0);
        C[i + 1] = vaddvq_f32(f1);
        C[i + 2] = vaddvq_f32(f2);
        C[i + 3] = vaddvq_f32(f3);
    }

    /* A worker's row range need not be a multiple of four. */
    for (; i < re; ++i) {
        const JAM_BLK* w = (const JAM_BLK*)(W + (size_t)i * wrow * sizeof(JAM_BLK));
        float32x4_t f = vdupq_n_f32(0);
        for (int blk = 0; blk < nb; ++blk) {
            const int8_t* p = q + (size_t)blk * 32;
            int8x16_t lo, hi;
            float dw;
            JAM_DECODE(&w[blk], &lo, &hi, &dw);
            f = vfmaq_n_f32(f,
                            vcvtq_f32_s32(JAM_BLKDOT(lo, hi, vld1q_s8(p), vld1q_s8(p + 16))),
                            dw * d[blk]);
        }
        C[i] = vaddvq_f32(f);
    }
}
