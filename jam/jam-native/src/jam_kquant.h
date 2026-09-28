/* GGML block-format constants and the exact scalar dots shared by the int8 kernels.
 *
 * A 32-block quant (Q8_0, Q4_0, Q5_0) is one fp16 scale per 32 elements. A K-quant is a 256-element
 * super-block with a hierarchy of scales: one fp16 d (and dmin) over 8 or 16 integer sub-block scales.
 * Prefill repacks weight rows into bands, so that one dot instruction accumulates a row per lane
 * against a broadcast activation group: 32 rows on AVX-512-VNNI (jam_kernels_band32_avx512.c, integer
 * sub-block scales), 8 rows on AVX2 and AVX-VNNI (float corrections from exact per-16 sums). */
#ifndef JAM_KQUANT_H
#define JAM_KQUANT_H

#include <stdint.h>
#include "kernels/jam_fp16.h"   /* jam_half_at (for the scalar tail dots below) */

#define JAM_QK          32     /* elements per 32-block (activation quant granularity) */
#define JAM_Q8_0_BYTES  34     /* fp16 d + 32 int8 */
#define JAM_Q4_0_BYTES  18     /* fp16 d + 16 nibble bytes */
#define JAM_Q5_0_BYTES  22     /* fp16 d + 32 high bits (u32) + 16 nibble bytes */
#define JAM_QKK         256    /* elements per K-quant super-block */
#define JAM_Q4K_BYTES   144    /* d(f16) dmin(f16) scales[12] qs[128] */
#define JAM_Q5K_BYTES   176    /* d(f16) dmin(f16) scales[12] qh[32] qs[128] */
#define JAM_Q6K_BYTES   210    /* ql[128] qh[64] scales[16] d(f16) */
#define JAM_VNNI_BAND   32     /* weight rows per parallel work unit (2 groups of 16) */
#define JAM_VNNI_MIN_SEQ 8     /* columns from which a band amortizes its repack; below, the row kernels */

/* The 8 6-bit (scale, min) pairs of a Q4_K / Q5_K super-block, 12 packed bytes -> one byte each.
 * Branch- and loop-free (llama.cpp's three-mask word trick): scales 0-3 and mins 0-3 are the low 6
 * bits of bytes 0-3 and 4-7, scales / mins 4-7 the nibbles of bytes 8-11 under their top 2 bits. */
static inline void jam_q4k_scales_mins(const uint8_t* b, uint8_t* sc, uint8_t* mn) {
    const uint32_t m6 = 0x3f3f3f3f, m4 = 0x0f0f0f0f, m2 = 0x03030303;
    uint32_t u0, u1, u2, s[2], m[2];
    __builtin_memcpy(&u0, b, 4); __builtin_memcpy(&u1, b + 4, 4); __builtin_memcpy(&u2, b + 8, 4);
    s[0] = u0 & m6; s[1] = (u2 & m4)        | (((u0 >> 6) & m2) << 4);
    m[0] = u1 & m6; m[1] = ((u2 >> 4) & m4) | (((u1 >> 6) & m2) << 4);
    __builtin_memcpy(sc, s, 8); __builtin_memcpy(mn, m, 8);
}

/* Exact scalar Q8_0 / Q4_0 / Q5_0 · f32 dots over nb consecutive 32-blocks: the 8-row bands'
 * partial-row tails (dequant on the fly; the SIMD kernels use the int8 pipeline). */
static inline float jam_q8_0_dot_f32(const uint8_t* w, int nb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += JAM_Q8_0_BYTES, x += JAM_QK) {
        float d = jam_half_at(w);
        const int8_t* q = (const int8_t*) (w + 2);
        float s = 0.0f;
        for (int e = 0; e < 32; e++) s += (float) q[e] * x[e];
        acc += d * s;
    }
    return acc;
}
/* Q5_0 block {fp16 d; u32 qh; nibble qs[16]}: element j = (qs[j] & 0xF | bit j of qh << 4) - 16, element
 * j+16 = (qs[j] >> 4 | bit j+16 of qh << 4) - 16. */
static inline void jam_q5_0_unpack(const uint8_t* w, int8_t* q) {
    uint32_t qh = (uint32_t) w[2] | (uint32_t) w[3] << 8 | (uint32_t) w[4] << 16 | (uint32_t) w[5] << 24;
    const uint8_t* qs = w + 6;
    for (int j = 0; j < 16; j++) {
        q[j]      = (int8_t)(((qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16);
        q[j + 16] = (int8_t)(((qs[j] >> 4)   | ((qh >> (j + 12)) & 0x10)) - 16);
    }
}
static inline float jam_q5_0_dot_f32(const uint8_t* w, int nb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += JAM_Q5_0_BYTES, x += JAM_QK) {
        float d = jam_half_at(w);
        int8_t q[32]; jam_q5_0_unpack(w, q);
        float s = 0.0f;
        for (int e = 0; e < 32; e++) s += (float) q[e] * x[e];
        acc += d * s;
    }
    return acc;
}
static inline float jam_q4_0_dot_f32(const uint8_t* w, int nb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < nb; B++, w += JAM_Q4_0_BYTES, x += JAM_QK) {
        float d = jam_half_at(w);
        const uint8_t* q = w + 2;
        float s = 0.0f;
        for (int e = 0; e < 16; e++) {
            s += (float)((q[e] & 0xF) - 8) * x[e];
            s += (float)((q[e] >> 4)  - 8) * x[e + 16];
        }
        acc += d * s;
    }
    return acc;
}

/* Exact scalar K-quant · f32 dots over sb consecutive 256-blocks: the 8-row bands' partial-row tails. */
static inline float jam_q4k_dot_f32(const uint8_t* w, int sb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < sb; B++, w += JAM_Q4K_BYTES, x += JAM_QKK) {
        float d = jam_half_at(w), dmin = jam_half_at(w + 2);
        uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
        const uint8_t* q = w + 16;
        for (int g = 0; g < 4; g++) {
            float dl = d * sc[g * 2], ml = dmin * mn[g * 2], dh = d * sc[g * 2 + 1], mh = dmin * mn[g * 2 + 1];
            for (int i = 0; i < 32; i++) {
                acc += (dl * (q[g * 32 + i] & 0xF) - ml) * x[g * 64 + i];
                acc += (dh * (q[g * 32 + i] >> 4)  - mh) * x[g * 64 + 32 + i];
            }
        }
    }
    return acc;
}
/* Q5_K = Q4_K plus a fifth bit plane qh[32]: sub-block s takes bit s of qh[i]. */
static inline float jam_q5k_dot_f32(const uint8_t* w, int sb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < sb; B++, w += JAM_Q5K_BYTES, x += JAM_QKK) {
        float d = jam_half_at(w), dmin = jam_half_at(w + 2);
        uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
        const uint8_t* qh = w + 16; const uint8_t* q = w + 48;
        for (int g = 0; g < 4; g++) {
            float dl = d * sc[g * 2], ml = dmin * mn[g * 2], dh = d * sc[g * 2 + 1], mh = dmin * mn[g * 2 + 1];
            for (int i = 0; i < 32; i++) {
                int lo = (q[g * 32 + i] & 0xF) | (((qh[i] >> (2 * g))     & 1) << 4);
                int hi = (q[g * 32 + i] >> 4)  | (((qh[i] >> (2 * g + 1)) & 1) << 4);
                acc += (dl * lo - ml) * x[g * 64 + i];
                acc += (dh * hi - mh) * x[g * 64 + 32 + i];
            }
        }
    }
    return acc;
}
/* Q6_K: element e of a half h = e >> 7 reads the nibble plane j = (e >> 5) & 3 (ql low nibbles for
 * j < 2, high for j >= 2) plus two bits of qh; one int8 scale per 16 elements, value = d·sc·(q - 32). */
static inline float jam_q6k_dot_f32(const uint8_t* w, int sb, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < sb; B++, w += JAM_Q6K_BYTES, x += JAM_QKK) {
        const uint8_t* ql = w; const uint8_t* qh = w + 128; const int8_t* sc = (const int8_t*) (w + 192);
        float d = jam_half_at(w + 208);
        for (int e = 0; e < JAM_QKK; e++) {
            int h = e >> 7, j = (e >> 5) & 3, l = e & 31;
            int lo = ql[h * 64 + (j & 1) * 32 + l];
            lo = j < 2 ? lo & 0xF : lo >> 4;
            int q = (lo | (((qh[h * 32 + l] >> (2 * j)) & 3) << 4)) - 32;
            acc += d * sc[e >> 4] * q * x[e];
        }
    }
    return acc;
}

#endif /* JAM_KQUANT_H */
