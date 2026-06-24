/* NVFP4 (NVIDIA FP4) shared decode bits. PROTOTYPE layout (self-contained, deliberately simple for bring-up):
 *   per-tensor:  float32 global scale G   — a 4-byte header at the START of the weight buffer
 *   per block (16 elements): { uint8_t e;  uint8_t qs[8]; } = 9 bytes
 *       e  = E4M3 (OCP FP8) per-block scale
 *       qs = 16 E2M1 FP4 nibbles; qs[t] low nibble = element t, high nibble = element t+8
 *   decode(element) = G · e4m3(e) · fp4_value
 *
 * Like MXFP4 (identical E2M1 element codes) but: a 16-element block (vs 32), an E4M3 block scale (vs E8M0),
 * and a second-level per-tensor FP32 scale. The codes are stored ×2 (JAM_MXFP4_CODES) so the int8 dot stays
 * exact; the ×½ folds into jam_nvfp4_dhalf, and G is applied once per output element.
 *
 * NOTE (layout is NOT final): real NVFP4 checkpoints (TensorRT-LLM / compressed-tensors) store the FP4 data,
 * the FP8 block scales, and the FP32 global scale as THREE SEPARATE (planar) tensors, and may pack the two
 * nibbles of a byte consecutively (2t, 2t+1) rather than (t, t+8). This interleaved+header form is for
 * decode/kernel bring-up only — the decode MATH is identical; only the addressing differs. */
#ifndef JAM_NVFP4_H
#define JAM_NVFP4_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "jam_mxfp4.h"   /* JAM_MXFP4_CODES — NVFP4 shares the E2M1 element codes */

#define JAM_NVFP4_BLK 16
typedef struct __attribute__((packed)) { uint8_t e; uint8_t qs[8]; } jam_nvfp4_blk;   /* 9 bytes / 16 elems */

/* OCP FP8 E4M3 -> float. 4-bit exponent (bias 7), 3-bit mantissa, no infinities; 0x7F/0xFF are NaN
 * (never produced for a scale). (1 + m/8)·2^(e-7) = (8+m)·2^(e-10); subnormal (e==0): (m/8)·2^-6 = m·2^-9. */
static inline float jam_e4m3_to_float(uint8_t x) {
    int s = (x >> 7) & 1, e = (x >> 3) & 0xF, m = x & 0x7;
    float v = e ? ldexpf((float) (8 + m), e - 10) : ldexpf((float) m, -9);
    return s ? -v : v;
}

/* per-block float scale for the int8 path: codes are stored ×2 (JAM_MXFP4_CODES), so fold the ×½ here. */
static inline float jam_nvfp4_dhalf(uint8_t e) { return 0.5f * jam_e4m3_to_float(e); }

#endif /* JAM_NVFP4_H */
