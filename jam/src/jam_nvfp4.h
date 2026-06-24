/* NVFP4 (NVIDIA FP4) shared decode bits. LOCKED layout: PLANAR scales, global scale applied by the CALLER.
 *
 *   weight buffer = [ FP4 data plane | E4M3 scale plane ]   (two contiguous planes, one jam_mm pointer)
 *     FP4 data plane : row-major, k/2 bytes/row. Packed in 32-element spans MXFP4-style — for span s, byte
 *                      (s*16+t) holds element (s*32+t) in its LOW nibble and element (s*32+16+t) in its HIGH
 *                      nibble. So each 16-byte span decodes to two contiguous 16-element halves.
 *     E4M3 scale plane: row-major, k/16 bytes/row — ONE E4M3 (FP8) scale per 16-element block, i.e. two per
 *                      32-span (scale[2s] = low half, scale[2s+1] = high half).
 *   per-tensor FP32 global scale G: NOT handled here — jam returns the un-G result; the caller multiplies
 *                      its output by G (G factors out exactly, so this is a single post-scale per element).
 *
 * decode(element) = e4m3(block_scale) · fp4_value   (× G at the caller). Same E2M1 element codes as MXFP4,
 * stored ×2 (JAM_MXFP4_CODES) so the int8 dot stays exact; the ×½ folds into jam_nvfp4_dhalf.
 *
 * NOTE: real NVFP4 checkpoints may pack the two nibbles of a byte consecutively (2t, 2t+1) rather than the
 * MXFP4 (t, t+16) order used here; that is a decode-time relabel, the scale/global handling is unchanged. */
#ifndef JAM_NVFP4_H
#define JAM_NVFP4_H

#include <stdint.h>
#include <math.h>
#include "jam_mxfp4.h"   /* JAM_MXFP4_CODES — NVFP4 shares the E2M1 element codes */

#define JAM_NVFP4_BLK 16   /* elements per E4M3 scale (two per 32-element span) */

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
