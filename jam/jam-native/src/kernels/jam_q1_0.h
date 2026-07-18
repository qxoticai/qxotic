/* Q1_0 (1-bit sign) - matches GGUF / llama.cpp block_q1_0 (GGML_TYPE_Q1_0 = 41).
 *   block = { fp16 d; uint8_t qs[16]; }  = 18 bytes, 128 elements (JAM_Q1_0_QK), sign bits LSB-first
 *   (byte = elem/8); value = bit ? +d : -d. */
#ifndef JAM_Q1_0_H
#define JAM_Q1_0_H

#include <stdint.h>
#include "jam_fp16.h"

#define JAM_Q1_0_QK    128
#define JAM_Q1_0_BYTES 18
typedef struct { uint16_t d; uint8_t qs[16]; } jam_blk_q1_0;   /* 18 bytes (matches GGML block_q1_0) */

/* Exact scalar Q1_0 · f32 dot over nb128 consecutive blocks - the generic floor's per-cell dot and
 * the band kernels' <16-row tail (dequant-on-the-fly; the SIMD kernels use the int8 pipeline). */
static inline float jam_q1_0_dot_f32(const uint8_t* w, int nb128, const float* x) {
    float acc = 0.0f;
    for (int B = 0; B < nb128; B++, w += JAM_Q1_0_BYTES, x += JAM_Q1_0_QK) {
        float d = jam_half2float(*(const uint16_t*) w);
        const uint8_t* q = w + 2;
        float s = 0.0f;
        for (int j = 0; j < JAM_Q1_0_QK; j++) s += ((q[j >> 3] >> (j & 7)) & 1) ? x[j] : -x[j];
        acc += d * s;
    }
    return acc;
}

#endif /* JAM_Q1_0_H */
