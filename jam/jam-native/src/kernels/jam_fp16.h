/* Portable software fp16 -> fp32 (IEEE half to float). The single source of truth for hosts/kernels
 * without F16C — the generic floor and the SSE3 decoders both use it, so the conversion can't drift. */
#ifndef JAM_FP16_H
#define JAM_FP16_H

#include <stdint.h>

static inline float jam_half2float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }                            /* zero */
        else {                                                  /* subnormal */
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3FFu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {                                  /* inf / nan */
        f = sign | 0x7F800000u | (mant << 13);
    } else {                                                    /* normal */
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float r; __builtin_memcpy(&r, &f, sizeof r); return r;
}

#endif /* JAM_FP16_H */
