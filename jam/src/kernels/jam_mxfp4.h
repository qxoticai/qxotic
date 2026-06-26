/* MXFP4 (OCP microscaling FP4) shared decode bits. A block is 32 elements:
 *   { uint8_t e;  uint8_t qs[16]; }   = 17 bytes
 * `e` is an E8M0 power-of-two scale; each nibble is an FP4 (E2M1) code. qs[j] low nibble = element j,
 * high nibble = element j+16. The 16 FP4 codes decode to magnitudes {0,.5,1,1.5,2,3,4,6} (signed) —
 * whose ×2 are the integers {0,1,2,3,4,6,8,12}, so we decode FP4->int8 with one shuffle and fold the
 * ×0.5 into the scale (jam_mxfp4_dhalf), then run the same int8 dot as Q8_0. */
#ifndef JAM_MXFP4_H
#define JAM_MXFP4_H

#include <stdint.h>
#include <string.h>

#define JAM_MXFP4_QK 32
typedef struct { uint8_t e; uint8_t qs[16]; } jam_mxfp4_blk;   /* 17 bytes */

/* int8 codes = FP4 value × 2 (ggml kvalues_mxfp4); the ×½ lives in jam_mxfp4_dhalf. */
#define JAM_MXFP4_CODES 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12

/* E8M0 byte -> 0.5 · 2^(e-127) = 2^(e-128). decode(value) = dhalf · (value×2) = 2^(e-127) · value. */
static inline float jam_mxfp4_dhalf(uint8_t e) {
    uint32_t bits = (e == 0) ? 0x00400000u : ((uint32_t) e << 23);   /* 2^(e-127) */
    float s; memcpy(&s, &bits, 4);
    return 0.5f * s;
}

#endif /* JAM_MXFP4_H */
