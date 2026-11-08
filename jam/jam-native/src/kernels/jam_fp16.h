/* fp16 -> fp32 (IEEE half to float). jam_half2float is the portable software conversion, the single
 * source of truth for code without F16C (the generic floor, the SSE3 decoders), so it can't drift.
 * jam_h2f is the one to call: in a TU built with F16C it is vcvtph2ps, the same value for every half
 * (a signaling NaN comes out quiet, still a NaN), else jam_half2float. */
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

#if defined(__F16C__)
#include <immintrin.h>
static inline float jam_h2f(uint16_t h) { return _cvtsh_ss(h); }
#else
static inline float jam_h2f(uint16_t h) { return jam_half2float(h); }
#endif

/* The half at p, at any alignment: a GGUF block's scale need not sit on an even address. */
static inline float jam_half_at(const void* p) {
    uint16_t h; __builtin_memcpy(&h, p, sizeof h); return jam_h2f(h);
}

#endif /* JAM_FP16_H */
