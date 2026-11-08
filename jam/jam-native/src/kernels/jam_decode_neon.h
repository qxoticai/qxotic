/* NEON weight decoders for the ARM simple-block kernels (jam_gemm_neon.inc, jam_gemv4_dotprod.inc,
 * jam_gemm_i8mm_4x4.inc): each turns one 32-element weight block into two signed int8x16 halves + a float
 * block scale (ARM has a native signed int8 dot, so no abs/sign trick). One decoder per quant, shared by
 * every ARM ISA - the dot is the ISA knob. Also the prefill band's row decoders and the small loads and
 * dots the ARM kernel files share. */
#ifndef JAM_DECODE_NEON_H
#define JAM_DECODE_NEON_H

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include "jam_mxfp4.h"   /* jam_mxfp4_blk, jam_mxfp4_dhalf, JAM_MXFP4_CODES (pure C) */

typedef struct __attribute__((packed)) { uint16_t d; int8_t  qs[32]; } jam_q8_blk;    /* Q8_0: 34 B */
typedef struct __attribute__((packed)) { uint16_t d; uint8_t qs[16]; } jam_q4_0_blk;  /* Q4_0: 18 B */

static inline float jam_neon_h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float) x; }  /* fp16->f32 (FCVT) */

/* Loads at any alignment: GGUF block fields and the packed layouts' f32 scales (jam.h JAM_PACK_ABI)
 * need not sit on their type's alignment. Each compiles to the plain ldr. */
static inline float jam_neon_half_at(const void* p) { __fp16 x; memcpy(&x, p, 2); return (float) x; }
static inline float jam_neon_f32_at(const void* p) { float x; memcpy(&x, p, 4); return x; }
static inline float32x4_t jam_neon_f32x4_at(const void* p) { return vreinterpretq_f32_u8(vld1q_u8((const uint8_t*) p)); }

/* 16-element signed int8 dot -> int: widening multiply + pairwise accumulate (NEON), one SDOT (dotprod). */
static inline int jam_dot16_neon(int8x16_t w, int8x16_t a) {
    int32x4_t d = vdupq_n_s32(0);
    d = vpadalq_s16(d, vmull_s8(vget_low_s8(w),  vget_low_s8(a)));
    d = vpadalq_s16(d, vmull_s8(vget_high_s8(w), vget_high_s8(a)));
    return (int) vaddvq_s32(d);
}
#ifdef __ARM_FEATURE_DOTPROD
static inline int jam_dot16_dotprod(int8x16_t w, int8x16_t a) {
    return (int) vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, a));
}

/* MXFP4 packed layout (jam.h JAM_PACK_ABI): lo[c] / hi[c] are k-chunk c of a 4-row group, 4 bytes per
 * row, lane = row, decoded to the low / high nibbles' codes (elements 4c..4c+3 and 16+4c..16+4c+3).
 * Returns the 4 rows' int32 dots against one activation block a0 (elements 0-15) | a1 (16-31). */
static inline int32x4_t jam_mxfp4_rows4(const int8x16_t lo[4], const int8x16_t hi[4], int8x16_t a0, int8x16_t a1) {
    int32x4_t s = vdupq_n_s32(0);
    s = vdotq_laneq_s32(s, lo[0], a0, 0); s = vdotq_laneq_s32(s, hi[0], a1, 0);
    s = vdotq_laneq_s32(s, lo[1], a0, 1); s = vdotq_laneq_s32(s, hi[1], a1, 1);
    s = vdotq_laneq_s32(s, lo[2], a0, 2); s = vdotq_laneq_s32(s, hi[2], a1, 2);
    s = vdotq_laneq_s32(s, lo[3], a0, 3); s = vdotq_laneq_s32(s, hi[3], a1, 3);
    return s;
}
#endif

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

/* ---- The prefill band's row decoders (jam_band4_neon.inc): one 32-block of one weight row as int8 codes,
 * the float scales of its two 16-element halves (equal where the format scales per 32 or more) and its min
 * (value = d·q - m; 0 where the format has none). Every format jam reads on ARM, so the band is the one
 * prefill path; the decode runs once per weight row per call, so clarity beats cleverness here. ---- */
#include "jam_kquant.h"   /* K-quant block layouts, jam_q4k_scales_mins */
#include "jam_nvfp4.h"    /* jam_nvfp4_blk, jam_ue4m3_lut */
#include "jam_q1_0.h"     /* JAM_Q1_0_BYTES, JAM_Q1_0_TBL */

typedef struct { int8x16_t lo, hi; float d, dh, m; } jam_b4_blk;
typedef void (*jam_b4_decode)(const uint8_t* row, int blk, jam_b4_blk* b);

static inline void jam_b4_q8_0(const uint8_t* row, int blk, jam_b4_blk* b) {
    jam_decode_q8_0_neon(row + (size_t) blk * JAM_Q8_0_BYTES, &b->lo, &b->hi, &b->d); b->dh = b->d; b->m = 0;
}
static inline void jam_b4_q4_0(const uint8_t* row, int blk, jam_b4_blk* b) {
    jam_decode_q4_0_neon(row + (size_t) blk * JAM_Q4_0_BYTES, &b->lo, &b->hi, &b->d); b->dh = b->d; b->m = 0;
}
static inline void jam_b4_q5_0(const uint8_t* row, int blk, jam_b4_blk* b) {
    jam_decode_q5_0_neon(row + (size_t) blk * JAM_Q5_0_BYTES, &b->lo, &b->hi, &b->d); b->dh = b->d; b->m = 0;
}
static inline void jam_b4_mxfp4(const uint8_t* row, int blk, jam_b4_blk* b) {
    jam_decode_mxfp4_neon(row + (size_t) blk * JAM_MXFP4_BYTES, &b->lo, &b->hi, &b->d); b->dh = b->d; b->m = 0;
}

/* Q4_K: super-block = d, dmin, 6-bit scales and mins for 8 sub-blocks, 128 nibble bytes; sub-block s
 * takes the low (s even) or high (s odd) nibbles of bytes 32·(s/2)..+32; value = d·sc·q - dmin·mn. */
static inline void jam_b4_q4k(const uint8_t* row, int blk, jam_b4_blk* b) {
    const uint8_t* w = row + (size_t) (blk >> 3) * JAM_Q4K_BYTES;
    const int s = blk & 7;
    uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t q0 = vld1q_u8(w + 16 + (s >> 1) * 32), q1 = vld1q_u8(w + 16 + (s >> 1) * 32 + 16);
    b->lo = vreinterpretq_s8_u8(s & 1 ? vshrq_n_u8(q0, 4) : vandq_u8(q0, m4));
    b->hi = vreinterpretq_s8_u8(s & 1 ? vshrq_n_u8(q1, 4) : vandq_u8(q1, m4));
    b->d = b->dh = jam_neon_half_at(w) * sc[s]; b->m = jam_neon_half_at(w + 2) * mn[s];
}
/* Q5_K: Q4_K plus a 5th bit per element, bit s of qh[e] (qh at +16, the nibbles at +48). */
static inline void jam_b4_q5k(const uint8_t* row, int blk, jam_b4_blk* b) {
    const uint8_t* w = row + (size_t) (blk >> 3) * JAM_Q5K_BYTES;
    const int s = blk & 7;
    uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
    const uint8x16_t m4 = vdupq_n_u8(0x0F), one = vdupq_n_u8(1);
    const int8x16_t sh = vdupq_n_s8((int8_t) -s);
    const uint8x16_t q0 = vld1q_u8(w + 48 + (s >> 1) * 32), q1 = vld1q_u8(w + 48 + (s >> 1) * 32 + 16);
    const uint8x16_t h0 = vshlq_n_u8(vandq_u8(vshlq_u8(vld1q_u8(w + 16), sh), one), 4);
    const uint8x16_t h1 = vshlq_n_u8(vandq_u8(vshlq_u8(vld1q_u8(w + 32), sh), one), 4);
    b->lo = vreinterpretq_s8_u8(vorrq_u8(s & 1 ? vshrq_n_u8(q0, 4) : vandq_u8(q0, m4), h0));
    b->hi = vreinterpretq_s8_u8(vorrq_u8(s & 1 ? vshrq_n_u8(q1, 4) : vandq_u8(q1, m4), h1));
    b->d = b->dh = jam_neon_half_at(w) * sc[s]; b->m = jam_neon_half_at(w + 2) * mn[s];
}
/* Q6_K: ql[128] nibbles, qh[64] 2-bit highs, int8 scales per 16 elements, d; value = d·sc·(q - 32).
 * Sub-block s = half h (0..1) x group g (0..3): nibbles of ql[64h + 32(g&1)..], low for g < 2, highs
 * bits 2g..2g+1 of qh[32h..]. */
static inline void jam_b4_q6k(const uint8_t* row, int blk, jam_b4_blk* b) {
    const uint8_t* w = row + (size_t) (blk >> 3) * JAM_Q6K_BYTES;
    const int s = blk & 7, h = s >> 2, g = s & 3;
    const uint8_t* ql = w + h * 64 + (g & 1) * 32;
    const uint8_t* qh = w + 128 + h * 32;
    const int8_t* sc = (const int8_t*) (w + 192);
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m2 = vdupq_n_u8(3);
    const int8x16_t sh = vdupq_n_s8((int8_t) -(2 * g)), bias = vdupq_n_s8(32);
    const uint8x16_t l0 = vld1q_u8(ql), l1 = vld1q_u8(ql + 16);
    const uint8x16_t lo0 = g < 2 ? vandq_u8(l0, m4) : vshrq_n_u8(l0, 4);
    const uint8x16_t lo1 = g < 2 ? vandq_u8(l1, m4) : vshrq_n_u8(l1, 4);
    const uint8x16_t hi0 = vshlq_n_u8(vandq_u8(vshlq_u8(vld1q_u8(qh), sh), m2), 4);
    const uint8x16_t hi1 = vshlq_n_u8(vandq_u8(vshlq_u8(vld1q_u8(qh + 16), sh), m2), 4);
    b->lo = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo0, hi0)), bias);
    b->hi = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo1, hi1)), bias);
    const float d = jam_neon_half_at(w + 208);
    b->d = d * sc[h * 8 + g * 2]; b->dh = d * sc[h * 8 + g * 2 + 1]; b->m = 0;
}
/* NVFP4: 64-element blocks of 4 UE4M3-scaled 16-element sub-blocks; byte 8s+j holds elements j (low
 * nibble) and j+8 (high) of sub-block s, so a 16-byte span's low and high halves interleave two
 * sub-blocks. The codes are MXFP4's (FP4 x 2), the scale here is the UE4M3 value as is. */
static inline void jam_b4_nvfp4(const uint8_t* row, int blk, jam_b4_blk* b) {
    const jam_nvfp4_blk* w = (const jam_nvfp4_blk*) (row + (size_t) (blk >> 1) * sizeof(jam_nvfp4_blk));
    const int sp = blk & 1;
    static const int8_t codes[16] = { JAM_MXFP4_CODES };
    const int8x16_t lut = vld1q_s8(codes);
    const uint8x16_t qs = vld1q_u8(w->qs + sp * 16);
    const int8x16_t lo = vqtbl1q_s8(lut, vandq_u8(qs, vdupq_n_u8(0x0F))), hi = vqtbl1q_s8(lut, vshrq_n_u8(qs, 4));
    b->lo = vcombine_s8(vget_low_s8(lo), vget_low_s8(hi));
    b->hi = vcombine_s8(vget_high_s8(lo), vget_high_s8(hi));
    b->d = jam_ue4m3_lut[w->d[2 * sp]]; b->dh = jam_ue4m3_lut[w->d[2 * sp + 1]]; b->m = 0;
}
/* Q1_0: 128 sign bits per block, LSB first; value = bit ? +d : -d. */
static inline void jam_b4_q1_0(const uint8_t* row, int blk, jam_b4_blk* b) {
    const uint8_t* w = row + (size_t) (blk >> 2) * JAM_Q1_0_BYTES;   /* fp16 d, then qs[16] */
    const uint8_t* bits = w + 2 + (blk & 3) * 4;
    const int8x8_t one = vdup_n_s8(1);
    int8x8_t s[4];
    for (int i = 0; i < 4; ++i) {
        s[i] = vreinterpret_s8_u8(vshr_n_u8(vcreate_u8(JAM_Q1_0_TBL[bits[i]]), 4));   /* 0/1 per element */
        s[i] = vsub_s8(vadd_s8(s[i], s[i]), one);                                     /* -> -1/+1 */
    }
    b->lo = vcombine_s8(s[0], s[1]); b->hi = vcombine_s8(s[2], s[3]);
    b->d = b->dh = jam_neon_half_at(w); b->m = 0;
}

#endif /* JAM_DECODE_NEON_H */
