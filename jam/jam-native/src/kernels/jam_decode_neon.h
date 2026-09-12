/* NEON weight decoders for the jam_gemm_neon simple-block engine: each turns one 32-element weight block
 * into two signed int8x16 halves + a float block scale (ARM has a native signed int8 dot, so no abs/sign
 * trick). One decoder per quant, shared by every ARM ISA - the dot is the ISA knob. */
#ifndef JAM_DECODE_NEON_H
#define JAM_DECODE_NEON_H

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include "jam_mxfp4.h"   /* jam_mxfp4_blk, jam_mxfp4_dhalf, JAM_MXFP4_CODES (pure C) */

typedef struct __attribute__((packed)) { uint16_t d; int8_t  qs[32]; } jam_q8_blk;    /* Q8_0: 34 B */
typedef struct __attribute__((packed)) { uint16_t d; uint8_t qs[16]; } jam_q4_0_blk;  /* Q4_0: 18 B */

static inline float jam_neon_h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float) x; }  /* fp16->f32 (FCVT) */

/* Q8_0: weights are already int8. */
static inline void jam_decode_q8_0_neon(const void* b, int8x16_t* wlo, int8x16_t* whi, float* dA) {
    const jam_q8_blk* w = (const jam_q8_blk*) b;
    *wlo = vld1q_s8(w->qs); *whi = vld1q_s8(w->qs + 16); *dA = jam_neon_h2f(w->d);
}

/* Q4_0: value = d·(nibble-8); low nibbles -> elems 0..15, high nibbles -> 16..31. */
static inline void jam_decode_q4_0_neon(const void* b, int8x16_t* wlo, int8x16_t* whi, float* dA) {
    const jam_q4_0_blk* w = (const jam_q4_0_blk*) b;
    uint8x16_t qs = vld1q_u8(w->qs); int8x16_t e8 = vdupq_n_s8(8);
    *wlo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qs, vdupq_n_u8(0x0F))), e8);
    *whi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qs, 4)), e8);
    *dA = jam_neon_h2f(w->d);
}

/* Q5_0: value = d·(q-16), q = nibble | 5th bit (bit j of qh -> element j, bit j+16 -> element j+16). The high
 * bits spread to bytes with a per-lane byte broadcast (vqtbl1q) + bit-mask test. */
typedef struct __attribute__((packed)) { uint16_t d; uint32_t qh; uint8_t qs[16]; } jam_q5_0_blk;  /* 22 B */
static inline int8x16_t jam_q5_0_highbits_neon(uint32_t bits) {
    static const uint8_t shuf[16] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
    static const uint8_t mask[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t b = vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(bits)), vld1q_u8(shuf));
    uint8x16_t m = vld1q_u8(mask);
    return vreinterpretq_s8_u8(vandq_u8(vceqq_u8(vandq_u8(b, m), m), vdupq_n_u8(0x10)));
}
static inline void jam_decode_q5_0_neon(const void* b, int8x16_t* wlo, int8x16_t* whi, float* dA) {
    const jam_q5_0_blk* w = (const jam_q5_0_blk*) b;
    uint32_t qh; memcpy(&qh, &w->qh, 4);
    uint8x16_t qs = vld1q_u8(w->qs); int8x16_t e16 = vdupq_n_s8(16);
    int8x16_t lo = vorrq_s8(vreinterpretq_s8_u8(vandq_u8(qs, vdupq_n_u8(0x0F))), jam_q5_0_highbits_neon(qh));
    int8x16_t hi = vorrq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qs, 4)), jam_q5_0_highbits_neon(qh >> 16));
    *wlo = vsubq_s8(lo, e16); *whi = vsubq_s8(hi, e16);
    *dA = jam_neon_h2f(w->d);
}

/* MXFP4: nibble -> int8 code (FP4 value ×2) via NEON table lookup (vqtbl1q); ×½ folds into the scale. */
static inline void jam_decode_mxfp4_neon(const void* b, int8x16_t* wlo, int8x16_t* whi, float* dA) {
    const jam_mxfp4_blk* w = (const jam_mxfp4_blk*) b;
    static const int8_t codes[16] = { JAM_MXFP4_CODES };
    int8x16_t lut = vld1q_s8(codes); uint8x16_t qs = vld1q_u8(w->qs);
    *wlo = vqtbl1q_s8(lut, vandq_u8(qs, vdupq_n_u8(0x0F)));
    *whi = vqtbl1q_s8(lut, vshrq_n_u8(qs, 4));
    *dA = jam_mxfp4_dhalf(w->e);
}

#endif /* JAM_DECODE_NEON_H */
