/* GGML K-quant shared bits (ported from jinferjni.c), for the AVX-512-VNNI K-quant kernels.
 *
 * K-quants are 256-element super-blocks with a hierarchy of scales — too big for the 32-block decode×dot
 * engine, so they get dedicated kernels using the jinferjni.c scheme: repack 16 weight rows into a VNNI
 * layout so ONE vpdpbusd accumulates 16 rows across the 16 i32 lanes, with the activation broadcast as
 * the signed operand. Activations are quantized to PLAIN s8 (the weight nibbles 0..15 / q6+32 are the
 * UNSIGNED vpdpbusd operand), plus exact per-16 f32 sums so the Q4_K `dmin·min` and Q6_K `-32` terms stay
 * out of the integer dot and are corrected in float. */
#ifndef JAM_KQUANT_H
#define JAM_KQUANT_H

#include <stdint.h>

#define JAM_QK          32     /* elements per 32-block (activation quant granularity) */
#define JAM_QKK         256    /* elements per K-quant super-block */
#define JAM_Q4K_BYTES   144    /* d(f16) dmin(f16) scales[12] qs[128] */
#define JAM_Q5K_BYTES   176    /* d(f16) dmin(f16) scales[12] qh[32] qs[128] */
#define JAM_Q6K_BYTES   210    /* ql[128] qh[64] scales[16] d(f16) */
#define JAM_VNNI_BAND   32     /* weight rows per parallel work unit (2 groups of 16) */
#define JAM_VNNI_MIN_SEQ 8     /* below this, activation quant + repack don't amortize -> generic floor */

/* The 8 6-bit (scale, min) pairs of a Q4_K / Q5_K super-block (ggml packing). */
static inline void jam_q4k_scales_mins(const uint8_t* b, uint8_t* sc, uint8_t* mn) {
    for (int j = 0; j < 4; j++) {
        sc[j]     = b[j] & 63;
        mn[j]     = b[j + 4] & 63;
        sc[j + 4] = (uint8_t)((b[j + 8] & 0xF) | ((b[j] >> 6) << 4));
        mn[j + 4] = (uint8_t)((b[j + 8] >> 4) | ((b[j + 4] >> 6) << 4));
    }
}

#endif /* JAM_KQUANT_H */
