/* ARM NEON-baseline K-quant kernels (Q4_K/Q5_K/Q6_K @ F32 -> F32) - the int8-dot replacement for the
 * generic float floor on aarch64 cores without dotprod. The int8 dot uses widening multiply (vmull_s8) +
 * pairwise-accumulate (vpadalq_s16); the decode + correction live in jam_kquant_engine.inc. */
#include "jam_internal.h"
#include <arm_neon.h>
#include "jam_decode_neon.h"   /* jam_kquant.h, jam_neon_half_at, jam_dot16_neon */

/* 32-wide signed int8 dot of weight bytes w0u|w1u and int8 activations a (Σa is precomputed in requant). */
static inline int jam_kdot32_neon(uint8x16_t w0u, uint8x16_t w1u, const int8_t* a) {
    int8x16_t w0 = vreinterpretq_s8_u8(w0u), w1 = vreinterpretq_s8_u8(w1u);
    int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
    int32x4_t d = vdupq_n_s32(0);
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(w0),  vget_low_s8(a0)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(w0), vget_high_s8(a0)));
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(w1),  vget_low_s8(a1)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(w1), vget_high_s8(a1)));
    return (int) vaddvq_s32(d);
}

#define JAM_KDOT32         jam_kdot32_neon
#define JAM_KDOT16(w, a)   jam_dot16_neon((w), vld1q_s8(a))   /* Q6_K; weight already holds qv-32 */
#define JAM_Q4K_NAME       jam_mm_q4k_neon
#define JAM_Q5K_NAME       jam_mm_q5k_neon
#define JAM_Q6K_NAME       jam_mm_q6k_neon
#include "jam_kquant_engine.inc"
