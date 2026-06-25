/* Shared NEON NVFP4 body (GGUF block_nvfp4). The includer defines, before #include:
 *     JAM_NVFP4_DOT16(w, a) -> int   16-element signed int8 dot                          [per ISA]
 *     JAM_MM_NAME                    exported symbol
 * Requires <arm_neon.h>, "jam_internal.h", "jam_nvfp4.h". Decode = MXFP4's vqtbl1q LUT; the sub-block
 * nibble order (byte s*8+j: low=elem j, high=elem j+8) is reassembled per sub-block via vcombine; each
 * 16-element sub-block is dotted and scaled by its UE4M3 scale (× the per-32 activation scale ad). */
void JAM_MM_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k;
    const int nblk = k / JAM_NVFP4_QK;
    const size_t wrow = (size_t) nblk * sizeof(jam_nvfp4_blk);
    static const int8_t codes[16] = { JAM_MXFP4_CODES };
    const int8x16_t lut = vld1q_s8(codes);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int i = rb; i < re; ++i) {
        const jam_nvfp4_blk* wr = (const jam_nvfp4_blk*) (W + (size_t) i * wrow);
        for (int j = 0; j < n; ++j) {
            const int8_t* aq = AQ + (size_t) j * k;
            const float* ad = AD + (size_t) j * (k / 32);
            float acc = 0.0f;
            for (int bb = 0; bb < nblk; ++bb) {
                const jam_nvfp4_blk* w = &wr[bb];
                for (int sp = 0; sp < 2; ++sp) {              /* 2 spans of 32 = sub-blocks (2sp, 2sp+1) */
                    uint8x16_t qs = vld1q_u8(w->qs + sp * 16);
                    int8x16_t lo = vqtbl1q_s8(lut, vandq_u8(qs, m4));      /* [e0-7, e16-23] */
                    int8x16_t hi = vqtbl1q_s8(lut, vshrq_n_u8(qs, 4));     /* [e8-15, e24-31] */
                    int8x16_t wl = vcombine_s8(vget_low_s8(lo), vget_low_s8(hi));    /* elems 0-15  */
                    int8x16_t wh = vcombine_s8(vget_high_s8(lo), vget_high_s8(hi));  /* elems 16-31 */
                    int blk32 = bb * 2 + sp;
                    const int8_t* a = aq + (size_t) blk32 * 32;
                    int d0 = JAM_NVFP4_DOT16(wl, vld1q_s8(a));
                    int d1 = JAM_NVFP4_DOT16(wh, vld1q_s8(a + 16));
                    float s0 = jam_ue4m3_to_float(w->d[2*sp]), s1 = jam_ue4m3_to_float(w->d[2*sp + 1]);
                    acc += ad[blk32] * (s0 * (float) d0 + s1 * (float) d1);
                }
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}
