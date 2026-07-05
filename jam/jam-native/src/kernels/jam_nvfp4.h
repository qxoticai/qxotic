/* NVFP4 (NVIDIA FP4) — matches GGUF / llama.cpp block_nvfp4 (GGML_TYPE_NVFP4 = 40). Self-contained block,
 * NO per-tensor global scale.
 *   block = { uint8_t d[4];  uint8_t qs[32]; }  = 36 bytes, 64 elements (QK_NVFP4 = 64), 4 sub-blocks of 16.
 *     d[s] = UE4M3 (unsigned FP8) scale for sub-block s
 *     qs   = 64 packed E2M1 nibbles; within sub-block s, byte (s*8 + j): LOW nibble = element s*16 + j,
 *            HIGH nibble = element s*16 + 8 + j.
 *   value(sub s, element) = kvalues_mxfp4[nibble] · ue4m3(d[s])
 * Reuses JAM_MXFP4_CODES (== ggml kvalues_mxfp4). Like MXFP4, the ×½ IS required: kvalues are 2× the E2M1
 * value, and ggml bakes the compensating ×0.5 INTO ggml_ue4m3_to_fp32 itself ("Returns value * 0.5 to match
 * kvalues_mxfp4 convention") — jam_ue4m3_to_float mirrors that, so value = code · scale exactly. */
#ifndef JAM_NVFP4_H
#define JAM_NVFP4_H

#include <stdint.h>
#include <math.h>
#include "jam_mxfp4.h"   /* JAM_MXFP4_CODES == ggml kvalues_mxfp4 */

#define JAM_NVFP4_QK     64
#define JAM_NVFP4_SUB    16
typedef struct __attribute__((packed)) { uint8_t d[4]; uint8_t qs[32]; } jam_nvfp4_blk;   /* 36 bytes */

/* UE4M3 (unsigned OCP FP8 E4M3) -> float, matching ggml_ue4m3_to_fp32 EXACTLY (incl. its ×0.5, which
 * compensates kvalues_mxfp4 being 2× the E2M1 values): bit 7 ignored; 0 and 0x7F -> 0;
 * normal (1 + m/8)·2^(e-7)·½; subnormal (e == 0) m·2^-9·½. */
static inline float jam_ue4m3_to_float(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF, m = x & 0x7;
    return 0.5f * (e ? ldexpf(1.0f + (float) m / 8.0f, e - 7) : ldexpf((float) m, -9));
}

#endif /* JAM_NVFP4_H */
