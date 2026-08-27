/* Direct-layout I8MM matmul for 32-element block quants.
 *
 * SMMLA computes a 2-row x 2-column x 8-element tile.  Four row/column pairings and four K chunks
 * therefore form a 4x4x32 block.  Weight and activation vectors are loaded/decompressed once and
 * reused across both opposite dimensions.  No weight repacking is required.
 *
 * The includer defines JAM_BLK, JAM_DECODE, JAM_EDGE_NAME, JAM_TILE2_NAME and JAM_MM_NAME. */
static inline float JAM_EDGE_NAME(const JAM_BLK* w, const int8_t* a, const float* da, int nb) {
    float out = 0;
    for (int blk = 0; blk < nb; ++blk) {
        int8x16_t wlo, whi;
        float dw;
        JAM_DECODE(&w[blk], &wlo, &whi, &dw);
        const int8_t* p = a + (size_t)blk * 32;
        int32x4_t r = vdupq_n_s32(0);
        r = jam_i8mm_pair_acc(r, wlo, wlo, vld1q_s8(p), vld1q_s8(p));
        r = jam_i8mm_pair_acc(r, whi, whi, vld1q_s8(p + 16), vld1q_s8(p + 16));
        out += (float)vgetq_lane_s32(r, 0) * dw * da[blk];
    }
    return out;
}

static inline float32x4_t JAM_TILE2_NAME(const JAM_BLK* w0, const JAM_BLK* w1,
                                         const int8_t* a0, const int8_t* a1,
                                         const float* da0, const float* da1, int nb) {
    float32x4_t out = vdupq_n_f32(0);
    for (int blk = 0; blk < nb; ++blk) {
        int8x16_t w0lo, w0hi, w1lo, w1hi;
        float dw0, dw1;
        JAM_DECODE(&w0[blk], &w0lo, &w0hi, &dw0);
        JAM_DECODE(&w1[blk], &w1lo, &w1hi, &dw1);
        const int8_t* p0 = a0 + (size_t)blk * 32;
        const int8_t* p1 = a1 + (size_t)blk * 32;
        int32x4_t r = vdupq_n_s32(0);
        r = jam_i8mm_pair_acc(r, w0lo, w1lo, vld1q_s8(p0), vld1q_s8(p1));
        r = jam_i8mm_pair_acc(r, w0hi, w1hi, vld1q_s8(p0 + 16), vld1q_s8(p1 + 16));

        const float32x2_t cols = { da0[blk], da1[blk] };
        const float32x4_t scales = vcombine_f32(vmul_n_f32(cols, dw0), vmul_n_f32(cols, dw1));
        out = vfmaq_f32(out, vcvtq_f32_s32(r), scales);
    }
    return out;
}

void JAM_MM_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, nb = J->nb, k = J->k;
    const size_t wrow = (size_t)(J->lda / 32);
    #define JAM_I8MM_ROW(row) ((const JAM_BLK*)(W + (size_t)(row) * wrow * sizeof(JAM_BLK)))

    int i = rb;
    for (; i + 4 <= re; i += 4) {
        const JAM_BLK* w0 = JAM_I8MM_ROW(i + 0);
        const JAM_BLK* w1 = JAM_I8MM_ROW(i + 1);
        const JAM_BLK* w2 = JAM_I8MM_ROW(i + 2);
        const JAM_BLK* w3 = JAM_I8MM_ROW(i + 3);
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            const int8_t* a0 = J->aq + (size_t)(j + 0) * k;
            const int8_t* a1 = J->aq + (size_t)(j + 1) * k;
            const int8_t* a2 = J->aq + (size_t)(j + 2) * k;
            const int8_t* a3 = J->aq + (size_t)(j + 3) * k;
            const float* d0 = J->ad + (size_t)(j + 0) * nb;
            const float* d1 = J->ad + (size_t)(j + 1) * nb;
            const float* d2 = J->ad + (size_t)(j + 2) * nb;
            const float* d3 = J->ad + (size_t)(j + 3) * nb;
            float32x4_t f00 = vdupq_n_f32(0), f01 = vdupq_n_f32(0);
            float32x4_t f10 = vdupq_n_f32(0), f11 = vdupq_n_f32(0);

            for (int blk = 0; blk < nb; ++blk) {
                int8x16_t w0lo, w0hi, w1lo, w1hi, w2lo, w2hi, w3lo, w3hi;
                float dw0, dw1, dw2, dw3;
                JAM_DECODE(&w0[blk], &w0lo, &w0hi, &dw0);
                JAM_DECODE(&w1[blk], &w1lo, &w1hi, &dw1);
                JAM_DECODE(&w2[blk], &w2lo, &w2hi, &dw2);
                JAM_DECODE(&w3[blk], &w3lo, &w3hi, &dw3);

                const int8_t* p0 = a0 + (size_t)blk * 32;
                const int8_t* p1 = a1 + (size_t)blk * 32;
                const int8_t* p2 = a2 + (size_t)blk * 32;
                const int8_t* p3 = a3 + (size_t)blk * 32;
                const int8x16_t a0lo = vld1q_s8(p0), a1lo = vld1q_s8(p1);
                const int8x16_t a2lo = vld1q_s8(p2), a3lo = vld1q_s8(p3);
                int32x4_t r00 = vdupq_n_s32(0), r01 = vdupq_n_s32(0);
                int32x4_t r10 = vdupq_n_s32(0), r11 = vdupq_n_s32(0);
                r00 = jam_i8mm_pair_acc(r00, w0lo, w1lo, a0lo, a1lo);
                r01 = jam_i8mm_pair_acc(r01, w0lo, w1lo, a2lo, a3lo);
                r10 = jam_i8mm_pair_acc(r10, w2lo, w3lo, a0lo, a1lo);
                r11 = jam_i8mm_pair_acc(r11, w2lo, w3lo, a2lo, a3lo);

                const int8x16_t a0hi = vld1q_s8(p0 + 16), a1hi = vld1q_s8(p1 + 16);
                const int8x16_t a2hi = vld1q_s8(p2 + 16), a3hi = vld1q_s8(p3 + 16);
                r00 = jam_i8mm_pair_acc(r00, w0hi, w1hi, a0hi, a1hi);
                r01 = jam_i8mm_pair_acc(r01, w0hi, w1hi, a2hi, a3hi);
                r10 = jam_i8mm_pair_acc(r10, w2hi, w3hi, a0hi, a1hi);
                r11 = jam_i8mm_pair_acc(r11, w2hi, w3hi, a2hi, a3hi);

                const float32x2_t dc01 = { d0[blk], d1[blk] };
                const float32x2_t dc23 = { d2[blk], d3[blk] };
                f00 = vfmaq_f32(f00, vcvtq_f32_s32(r00),
                                 vcombine_f32(vmul_n_f32(dc01, dw0), vmul_n_f32(dc01, dw1)));
                f01 = vfmaq_f32(f01, vcvtq_f32_s32(r01),
                                 vcombine_f32(vmul_n_f32(dc23, dw0), vmul_n_f32(dc23, dw1)));
                f10 = vfmaq_f32(f10, vcvtq_f32_s32(r10),
                                 vcombine_f32(vmul_n_f32(dc01, dw2), vmul_n_f32(dc01, dw3)));
                f11 = vfmaq_f32(f11, vcvtq_f32_s32(r11),
                                 vcombine_f32(vmul_n_f32(dc23, dw2), vmul_n_f32(dc23, dw3)));
            }

            jam_i8mm_store_2x2(C, ldc, i + 0, j + 0, f00);
            jam_i8mm_store_2x2(C, ldc, i + 0, j + 2, f01);
            jam_i8mm_store_2x2(C, ldc, i + 2, j + 0, f10);
            jam_i8mm_store_2x2(C, ldc, i + 2, j + 2, f11);
        }

        if (j + 2 <= n) {
            const float32x4_t f0 = JAM_TILE2_NAME(w0, w1, J->aq + (size_t)j * k,
                                                  J->aq + (size_t)(j + 1) * k,
                                                  J->ad + (size_t)j * nb,
                                                  J->ad + (size_t)(j + 1) * nb, nb);
            const float32x4_t f1 = JAM_TILE2_NAME(w2, w3, J->aq + (size_t)j * k,
                                                  J->aq + (size_t)(j + 1) * k,
                                                  J->ad + (size_t)j * nb,
                                                  J->ad + (size_t)(j + 1) * nb, nb);
            jam_i8mm_store_2x2(C, ldc, i + 0, j, f0);
            jam_i8mm_store_2x2(C, ldc, i + 2, j, f1);
            j += 2;
        }
        if (j < n) {
            const int8_t* a = J->aq + (size_t)j * k;
            const float* da = J->ad + (size_t)j * nb;
            C[(size_t)j * ldc + i + 0] = JAM_EDGE_NAME(w0, a, da, nb);
            C[(size_t)j * ldc + i + 1] = JAM_EDGE_NAME(w1, a, da, nb);
            C[(size_t)j * ldc + i + 2] = JAM_EDGE_NAME(w2, a, da, nb);
            C[(size_t)j * ldc + i + 3] = JAM_EDGE_NAME(w3, a, da, nb);
        }
    }

    if (i + 2 <= re) {
        const JAM_BLK* w0 = JAM_I8MM_ROW(i + 0);
        const JAM_BLK* w1 = JAM_I8MM_ROW(i + 1);
        int j = 0;
        for (; j + 2 <= n; j += 2) {
            const float32x4_t f = JAM_TILE2_NAME(w0, w1, J->aq + (size_t)j * k,
                                                 J->aq + (size_t)(j + 1) * k,
                                                 J->ad + (size_t)j * nb,
                                                 J->ad + (size_t)(j + 1) * nb, nb);
            jam_i8mm_store_2x2(C, ldc, i, j, f);
        }
        if (j < n) {
            const int8_t* a = J->aq + (size_t)j * k;
            const float* da = J->ad + (size_t)j * nb;
            C[(size_t)j * ldc + i + 0] = JAM_EDGE_NAME(w0, a, da, nb);
            C[(size_t)j * ldc + i + 1] = JAM_EDGE_NAME(w1, a, da, nb);
        }
        i += 2;
    }
    if (i < re) {
        const JAM_BLK* w = JAM_I8MM_ROW(i);
        for (int j = 0; j < n; ++j)
            C[(size_t)j * ldc + i] = JAM_EDGE_NAME(w, J->aq + (size_t)j * k,
                                                   J->ad + (size_t)j * nb, nb);
    }
    #undef JAM_I8MM_ROW
}
