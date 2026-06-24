/* ARM DOTPROD K-quant kernels (Q4_K/Q5_K/Q6_K @ F32 -> F32) — same decode as the NEON baseline, but the
 * int8 dot uses the dot-product extension (vdotq_s32 / sdot): one instruction does a 16-wide int8 dot into
 * 4 i32 lanes, so 32-wide = two sdots. Built with -march=armv8.2-a+dotprod. i8mm cores reuse this (sdot is
 * available there too); an smmla 2x2 variant would be a further step but barely helps the decode-bound K-quants. */
#include "jam_internal.h"
#include "jam_kquant.h"
#include <arm_neon.h>

/* 32-wide signed int8 dot via two sdots, plus Σ activations (min term). */
static inline int jam_kdot32_dotprod(uint8x16_t w0u, uint8x16_t w1u, const int8_t* a, int* sum_a) {
    int8x16_t w0 = vreinterpretq_s8_u8(w0u), w1 = vreinterpretq_s8_u8(w1u);
    int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
    int32x4_t d = vdotq_s32(vdupq_n_s32(0), w0, a0);
    d = vdotq_s32(d, w1, a1);
    *sum_a = (int) vaddlvq_s8(a0) + (int) vaddlvq_s8(a1);
    return (int) vaddvq_s32(d);
}

/* 16-wide signed int8 dot (Q6_K; weight already holds qv-32). */
static inline int jam_kdot16_dotprod(int8x16_t w, const int8_t* a) {
    return (int) vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(a)));
}

#define JAM_KDOT32   jam_kdot32_dotprod
#define JAM_KDOT16   jam_kdot16_dotprod
#define JAM_Q4K_NAME jam_mm_q4k_dotprod
#define JAM_Q5K_NAME jam_mm_q5k_dotprod
#define JAM_Q6K_NAME jam_mm_q6k_dotprod
#include "jam_kquant_engine.inc"
