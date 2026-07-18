/* Shared NEON Q1_0 body (GGML block_q1_0: {fp16 d; uint8 qs[16]}, 128 elems, 1 sign bit each,
 * LSB-first; elem = bit ? +d : -d). The includer defines, before #include:
 *     JAM_Q1_0_DOT16(w, a) -> int   16-element signed int8 dot                          [per ISA]
 *     JAM_MM_NAME                   exported symbol
 * Requires <arm_neon.h>, "jam_internal.h", "jam_fp16.h". Sign expansion follows llama.cpp's arm
 * kernel: a 256-entry byte->8-nibble table (b << 4 per bit, LSB-first), shifted right by 4 to 0/1,
 * then 2*b - 1 -> -1/+1 sign bytes; dot(signs, activations) per 16, scaled per-32 by ad, per-128
 * by d. */

/* table[b] = the 8 bits of b expanded LSB-first to one byte each, value (bit << 4) - llama.cpp's
 * table_b2b_0 construction (arch/arm/quants.c), verbatim. */
#define JAM_B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#define JAM_B2(c,s,n) JAM_B1(c,s,n ## c), JAM_B1(c,s,n ## s)
#define JAM_B3(c,s,n) JAM_B2(c,s,n ## c), JAM_B2(c,s,n ## s)
#define JAM_B4(c,s,n) JAM_B3(c,s,n ## c), JAM_B3(c,s,n ## s)
#define JAM_B5(c,s,n) JAM_B4(c,s,n ## c), JAM_B4(c,s,n ## s)
#define JAM_B6(c,s,n) JAM_B5(c,s,n ## c), JAM_B5(c,s,n ## s)
#define JAM_B7(c,s,n) JAM_B6(c,s,n ## c), JAM_B6(c,s,n ## s)
#define JAM_B8(c,s  ) JAM_B7(c,s,     c), JAM_B7(c,s,     s)

static const uint64_t JAM_Q1_0_TBL[256] = { JAM_B8(00, 10) };

#include "jam_q1_0.h"   /* jam_blk_q1_0 */

void JAM_MM_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k;
    const int nblk = k / 128;
    const size_t wrow = (size_t)(J->lda / 128) * sizeof(jam_blk_q1_0);
    const int8x8_t one = vdup_n_s8(1);
    for (int i = rb; i < re; ++i) {
        const char* wr = W + (size_t) i * wrow;
        for (int j = 0; j < n; ++j) {
            const int8_t* aq = AQ + (size_t) j * k;
            const float* ad = AD + (size_t) j * (k / 32);
            float acc = 0.0f;
            for (int bb = 0; bb < nblk; ++bb) {
                const jam_blk_q1_0* w = (const jam_blk_q1_0*) (wr + (size_t) bb * sizeof(jam_blk_q1_0));
                const float d = jam_half2float(w->d);
                float bacc = 0.0f;
                for (int K = 0; K < 4; ++K) {                 /* 4 spans of 32 = 4 sign bytes each */
                    const uint8_t* bits = w->qs + K * 4;
                    int8x8_t s0 = vreinterpret_s8_u8(vshr_n_u8(vcreate_u8(JAM_Q1_0_TBL[bits[0]]), 4));
                    int8x8_t s1 = vreinterpret_s8_u8(vshr_n_u8(vcreate_u8(JAM_Q1_0_TBL[bits[1]]), 4));
                    int8x8_t s2 = vreinterpret_s8_u8(vshr_n_u8(vcreate_u8(JAM_Q1_0_TBL[bits[2]]), 4));
                    int8x8_t s3 = vreinterpret_s8_u8(vshr_n_u8(vcreate_u8(JAM_Q1_0_TBL[bits[3]]), 4));
                    s0 = vsub_s8(vadd_s8(s0, s0), one);       /* 0/1 -> -1/+1 */
                    s1 = vsub_s8(vadd_s8(s1, s1), one);
                    s2 = vsub_s8(vadd_s8(s2, s2), one);
                    s3 = vsub_s8(vadd_s8(s3, s3), one);
                    const int blk32 = bb * 4 + K;
                    const int8_t* a = aq + (size_t) blk32 * 32;
                    int p0 = JAM_Q1_0_DOT16(vcombine_s8(s0, s1), vld1q_s8(a));
                    int p1 = JAM_Q1_0_DOT16(vcombine_s8(s2, s3), vld1q_s8(a + 16));
                    bacc += ad[blk32] * (float)(p0 + p1);
                }
                acc += d * bacc;
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}
