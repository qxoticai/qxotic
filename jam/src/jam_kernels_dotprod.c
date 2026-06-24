/* DOTPROD Q8_0 @ F32 -> F32 (this TU built with -march=armv8.2-a+dotprod). The int8 block-dot is two
 * vdotq_s32 (signed 8-bit dot-product, 4 elems/lane) — the int8 workhorse on modern ARM (all Apple
 * M-series, Graviton2+, recent phones). Same deferred-float accumulation as the NEON-baseline. */
#include "jam_internal.h"
#include <string.h>
#include <arm_neon.h>

/* vdotq_s32(acc,a,b): acc[l] += Σ_{4} a[4l+..]·b[4l+..]. Two of them cover the 32-elem block -> int32x4. */
#define JAM_Q8_BLKDOT(wlo,whi,blo,bhi) vdotq_s32(vdotq_s32(vdupq_n_s32(0), wlo, blo), whi, bhi)
#define JAM_Q8_MM_NAME jam_mm_q8_0_dotprod
#include "jam_q8_neon.inc"
