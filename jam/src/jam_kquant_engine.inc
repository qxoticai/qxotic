/* ARM K-quant GEMM bodies (Q4_K/Q5_K/Q6_K @ F32 -> F32), shared across ARM int8 ISAs. The decode +
 * scale/min correction live here once; the includer supplies the int8 dot (per ISA) and the names:
 *
 *   JAM_KDOT32(w0u, w1u, a, &sum)   32-wide signed dot of weight bytes w0u|w1u and int8 acts a; sets Σa
 *   JAM_KDOT16(w, a)                16-wide signed dot of int8 weights w and int8 acts a  (Q6_K)
 *   JAM_Q4K_NAME / JAM_Q5K_NAME / JAM_Q6K_NAME   exported symbols
 *
 * Consumes jam_q8_job (J->aq int8 activations + J->ad per-32 scales, from jam_q8_0_requant). K-quant
 * weights decode to non-negative small ints (Q6_K folds the -32 into a signed weight), so the signed dot
 * is exact; the dmin·min term is corrected in float via Σx ≈ ad·Σaq. Mirrors jam_kernels_generic.c.
 *
 * Each thread tiles JAM_KTN activation columns: a weight sub-block is decoded ONCE and dotted against all
 * KTN columns, so the (expensive) nibble/scale unpack and the weight read are amortized KTN× — same as the
 * x86 SSE3/AVX engines. KTN matches the SSE3 4-column tile. */
#define JAM_KTN 4

void JAM_Q4K_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t)(J->lda / JAM_QKK) * JAM_Q4K_BYTES;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j0 = 0; j0 < n; j0 += JAM_KTN) {
            float acc[JAM_KTN]; for (int c = 0; c < JAM_KTN; ++c) acc[c] = 0.0f;
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q4K_BYTES) {
                float d = (float) *(const __fp16*) w, dmin = (float) *(const __fp16*) (w + 2);
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* q = w + 16;
                for (int g = 0; g < 4; ++g) {
                    uint8x16_t q0 = vld1q_u8(q + g*32), q1 = vld1q_u8(q + g*32 + 16);
                    uint8x16_t wl0 = vandq_u8(q0, m4),  wl1 = vandq_u8(q1, m4);     /* decode ONCE */
                    uint8x16_t wh0 = vshrq_n_u8(q0, 4), wh1 = vshrq_n_u8(q1, 4);
                    int bl = B*8 + 2*g, bh = bl + 1;
                    float dl=d*sc[2*g], ml=dmin*mn[2*g], dh=d*sc[2*g+1], mh=dmin*mn[2*g+1];
                    for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j >= n) break;
                        const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
                        const float* as = J->asum + (size_t) j * nb;   /* Σaq precomputed in requant */
                        int dlo = JAM_KDOT32(wl0, wl1, aq + (size_t) bl*32);
                        int dhi = JAM_KDOT32(wh0, wh1, aq + (size_t) bh*32);
                        acc[c] += ad[bl] * (dl*dlo - ml*as[bl]) + ad[bh] * (dh*dhi - mh*as[bh]);
                    }
                }
            }
            for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j < n) C[(size_t) j*ldc + i] = acc[c]; }
        }
    }
}

/* Q5_K: Q4_K plus a 5th weight bit from qh (q5 = nibble | (bit s of qh[e] << 4), 0..31). */
void JAM_Q5K_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t)(J->lda / JAM_QKK) * JAM_Q5K_BYTES;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), one = vdupq_n_u8(1);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j0 = 0; j0 < n; j0 += JAM_KTN) {
            float acc[JAM_KTN]; for (int c = 0; c < JAM_KTN; ++c) acc[c] = 0.0f;
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q5K_BYTES) {
                float d = (float) *(const __fp16*) w, dmin = (float) *(const __fp16*) (w + 2);
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* qh = w + 16; const uint8_t* qs = w + 48;
                uint8x16_t h0 = vld1q_u8(qh), h1 = vld1q_u8(qh + 16);
                for (int g = 0; g < 4; ++g) {
                    uint8x16_t q0 = vld1q_u8(qs + g*32), q1 = vld1q_u8(qs + g*32 + 16);
                    int8x16_t shL = vdupq_n_s8(-(int8_t)(2*g)), shH = vdupq_n_s8(-(int8_t)(2*g + 1));
                    uint8x16_t wl0 = vorrq_u8(vandq_u8(q0, m4), vshlq_n_u8(vandq_u8(vshlq_u8(h0, shL), one), 4));
                    uint8x16_t wl1 = vorrq_u8(vandq_u8(q1, m4), vshlq_n_u8(vandq_u8(vshlq_u8(h1, shL), one), 4));
                    uint8x16_t wh0 = vorrq_u8(vshrq_n_u8(q0, 4), vshlq_n_u8(vandq_u8(vshlq_u8(h0, shH), one), 4));
                    uint8x16_t wh1 = vorrq_u8(vshrq_n_u8(q1, 4), vshlq_n_u8(vandq_u8(vshlq_u8(h1, shH), one), 4));
                    int bl = B*8 + 2*g, bh = bl + 1;
                    float dl=d*sc[2*g], ml=dmin*mn[2*g], dh=d*sc[2*g+1], mh=dmin*mn[2*g+1];
                    for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j >= n) break;
                        const int8_t* aq = AQ + (size_t) j * k; const float* ad = AD + (size_t) j * nb;
                        const float* as = J->asum + (size_t) j * nb;   /* Σaq precomputed in requant */
                        int dlo = JAM_KDOT32(wl0, wl1, aq + (size_t) bl*32);
                        int dhi = JAM_KDOT32(wh0, wh1, aq + (size_t) bh*32);
                        acc[c] += ad[bl] * (dl*dlo - ml*as[bl]) + ad[bh] * (dh*dhi - mh*as[bh]);
                    }
                }
            }
            for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j < n) C[(size_t) j*ldc + i] = acc[c]; }
        }
    }
}

/* Q6_K: value = d·sc·(qv-32), qv 6-bit (ql nibble | qh 2-bit << 4). -32 folds into a signed weight; int8
 * scales, one per 16 elements (two per 32-sub-block), so each sub-block is two 16-wide dots. */
void JAM_Q6K_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t)(J->lda / JAM_QKK) * JAM_Q6K_BYTES;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m2 = vdupq_n_u8(3);
    const int8x16_t bias = vdupq_n_s8(32);
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j0 = 0; j0 < n; j0 += JAM_KTN) {
            float acc[JAM_KTN]; for (int c = 0; c < JAM_KTN; ++c) acc[c] = 0.0f;
            const uint8_t* w = wrow;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q6K_BYTES) {
                const uint8_t* ql = w; const uint8_t* qh = w + 128;
                const int8_t* sc = (const int8_t*) (w + 192);
                float d = (float) *(const __fp16*) (w + 208);
                for (int h = 0; h < 2; ++h) {
                    const uint8_t* qlb = ql + h*64; const uint8_t* qhb = qh + h*32;
                    for (int g = 0; g < 4; ++g) {
                        uint8x16_t l0 = vld1q_u8(qlb + (g&1)*32), l1 = vld1q_u8(qlb + (g&1)*32 + 16);
                        uint8x16_t lo0 = (g < 2) ? vandq_u8(l0, m4) : vshrq_n_u8(l0, 4);
                        uint8x16_t lo1 = (g < 2) ? vandq_u8(l1, m4) : vshrq_n_u8(l1, 4);
                        uint8x16_t hb0 = vld1q_u8(qhb), hb1 = vld1q_u8(qhb + 16);
                        int8x16_t sh = vdupq_n_s8(-(int8_t)(2*g));
                        uint8x16_t hi0 = vshlq_n_u8(vandq_u8(vshlq_u8(hb0, sh), m2), 4);
                        uint8x16_t hi1 = vshlq_n_u8(vandq_u8(vshlq_u8(hb1, sh), m2), 4);
                        int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo0, hi0)), bias);   /* qv-32, ONCE */
                        int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo1, hi1)), bias);
                        int blk = B*8 + h*4 + g;
                        float s0 = d * (float) sc[h*8 + g*2], s1 = d * (float) sc[h*8 + g*2 + 1];
                        for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j >= n) break;
                            const int8_t* a = AQ + (size_t) j * k + (size_t) blk*32; const float* ad = AD + (size_t) j * nb;
                            int dot0 = JAM_KDOT16(w0, a), dot1 = JAM_KDOT16(w1, a + 16);
                            acc[c] += ad[blk] * (s0 * dot0 + s1 * dot1);
                        }
                    }
                }
            }
            for (int c = 0; c < JAM_KTN; ++c) { int j = j0+c; if (j < n) C[(size_t) j*ldc + i] = acc[c]; }
        }
    }
}

#undef JAM_KTN
