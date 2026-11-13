/* Shared NEON body for the low-bit codes formats (jam_lowbit.h: Q2_0, PQ2_0, PTQ1_0; J->wt says
 * which). The includer defines, before #include:
 *     JAM_LOWBIT_DOT16(acc, w, a) -> int32x4   acc += a 16-element signed int8 dot     [per ISA]
 *     JAM_MM_NAME                     exported symbol
 * Requires <arm_neon.h> and "jam_internal.h". Each weight row decodes once to s8 codes and a scale per
 * 32, four rows at a time: each activation load feeds four accumulator chains (the shape of the Q4_0
 * 4x1 GEMV). A 32-block's dot stays in int32 lanes, its scales fold in through one float fma, and a
 * column reduces horizontally once. */

#include "jam_lowbit.h"

/* rows r .. r + nr (nr <= 4) against every column: nr accumulator chains share each activation load.
 * Always inlined, so the full-quad call's constant nr keeps acc[] in registers. */
static inline __attribute__((always_inline)) void lowbit_rows(const jam_q8_job* J, int r, int nr, const int8_t* codes, const float* d32) {
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb32 = k / 32;
    for (int j = 0; j < n; ++j) {
        const int8_t* aq = AQ + (size_t) j * k;
        const float* ad = AD + (size_t) j * nb32;
        float32x4_t acc[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
        for (int b = 0; b < nb32; ++b) {
            const int8x16_t a0 = vld1q_s8(aq + (size_t) b * 32), a1 = vld1q_s8(aq + (size_t) b * 32 + 16);
            for (int t = 0; t < nr; ++t) {
                const int8_t* w = codes + (size_t) t * k + (size_t) b * 32;
                int32x4_t p = JAM_LOWBIT_DOT16(vdupq_n_s32(0), vld1q_s8(w), a0);
                p = JAM_LOWBIT_DOT16(p, vld1q_s8(w + 16), a1);
                acc[t] = vfmaq_n_f32(acc[t], vcvtq_f32_s32(p), d32[(size_t) t * nb32 + b] * ad[b]);
            }
        }
        for (int t = 0; t < nr; ++t) C[(size_t) j*ldc + r + t] = vaddvq_f32(acc[t]);
    }
}

void JAM_MM_NAME(void* arg, int rb, int re, int tid) {
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int k = J->k, nb32 = k / 32, qk = jam_lowbit_qk(J->wt);
    const size_t wrow = (size_t)(J->lda / qk) * jam_lowbit_bytes(J->wt);
    int8_t* codes = (int8_t*) J->repack[tid].qs;              /* 4 rows of k codes, then their scales per 32 */
    float* d32 = (float*) (codes + 4 * (size_t) k);
    for (int r = rb; r < re; r += 4) {                         /* 4 rows decode once, then every column */
        const int nr = re - r < 4 ? re - r : 4;
        for (int t = 0; t < nr; ++t)
            jam_lowbit_row(J->wt, (const uint8_t*) J->a + (size_t) (r + t) * wrow, k / qk,
                           codes + (size_t) t * k, d32 + (size_t) t * nb32);
        if (nr == 4) lowbit_rows(J, r, 4, codes, d32);
        else         lowbit_rows(J, r, nr, codes, d32);
    }
}
