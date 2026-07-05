/* Shared test/bench helpers: IEEE half<->float, Q8_0 quantization, deterministic fill. */
#ifndef JAM_REF_H
#define JAM_REF_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "jam.h"

/* ISA ladder benched/tested, in order — shared by jam_bench.c and jam_test.c so adding a tier is one edit.
 * Levels the hardware lacks are auto-skipped (jam_active_isa != the requested cap). */
__attribute__((unused)) static const jam_isa jam_isa_levels[] = {
    JAM_ISA_GENERIC, JAM_ISA_SSE3, JAM_ISA_SSSE3, JAM_ISA_AVX2, JAM_ISA_AVX_VNNI, JAM_ISA_AVX512, JAM_ISA_AVX512_VNNI,
    JAM_ISA_NEON, JAM_ISA_DOTPROD, JAM_ISA_I8MM, JAM_ISA_SVE, JAM_ISA_METAL };
#define JAM_ISA_LEVELS_N (sizeof jam_isa_levels / sizeof *jam_isa_levels)

typedef struct { uint16_t d; int8_t qs[32]; } jam_ref_blk;   /* == GGML block_q8_0 (34 bytes) */

static inline uint16_t jam_ref_f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e    = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m    = x & 0x7FFFFFu;
    if (e <= 0)  return (uint16_t) sign;                       /* flush tiny to zero (scale only) */
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t) e << 10) | (m >> 13));
}

static inline float jam_ref_h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, f;
    if (e == 0) { if (!m) f = sign; else { e = 113; while (!(m & 0x400u)) { m <<= 1; --e; } m &= 0x3FFu; f = sign | (e << 23) | (m << 13); } }
    else if (e == 0x1Fu) f = sign | 0x7F800000u | (m << 13);
    else f = sign | ((e - 15 + 127) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}

/* Quantize a [rows×k] f32 matrix to contiguous Q8_0 blocks (malloc'd; caller frees). */
static inline jam_ref_blk* jam_ref_quant_q8_0(const float* X, int rows, int k) {
    int nb = k / 32;
    jam_ref_blk* o = (jam_ref_blk*) malloc((size_t) rows * nb * sizeof(jam_ref_blk));
    for (int i = 0; i < rows; ++i) for (int b = 0; b < nb; ++b) {
        const float* x = X + (size_t) i * k + b * 32;
        float amax = 0; for (int e = 0; e < 32; ++e) { float a = fabsf(x[e]); if (a > amax) amax = a; }
        float d = amax / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        jam_ref_blk* p = &o[(size_t) i * nb + b]; p->d = jam_ref_f2h(d);
        for (int e = 0; e < 32; ++e) { int v = (int) lrintf(x[e] * id); if (v > 127) v = 127; else if (v < -128) v = -128; p->qs[e] = (int8_t) v; }
    }
    return o;
}

/* Deterministic, reproducible fill in roughly [-1, 1]. */
static inline void jam_ref_fill(float* X, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        X[i] = (float)((double)(((unsigned)(i * 2654435761u) + seed) % 2003u) / 1000.0) - 1.0f;
}

/* ---- MXFP4 (OCP microscaling FP4): block = { e8m0 scale; 32 FP4 nibbles } = 17 bytes ---- */
typedef struct { uint8_t e; uint8_t qs[16]; } jam_ref_mxfp4_blk;
static const float jam_ref_mxfp4_mag[8] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };

static inline float jam_ref_mxfp4_dhalf(uint8_t e) {   /* matches jam_mxfp4_dhalf: 0.5·2^(e-127) */
    uint32_t bits = (e == 0) ? 0x00400000u : ((uint32_t) e << 23);
    float s; memcpy(&s, &bits, 4); return 0.5f * s;
}
static inline float jam_ref_mxfp4_decode(uint8_t nib, float dhalf) {   /* code -> value×2 -> ×dhalf */
    static const int8_t kv[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };
    return dhalf * (float) kv[nib & 0x0F];
}
static inline uint8_t jam_ref_fp4_code(float q) {      /* nearest E2M1 code */
    int sgn = q < 0; float a = fabsf(q); int best = 0; float bd = 1e30f;
    for (int i = 0; i < 8; ++i) { float d = fabsf(a - jam_ref_mxfp4_mag[i]); if (d < bd) { bd = d; best = i; } }
    return (uint8_t)(sgn ? (best | 8) : best);
}
static inline jam_ref_mxfp4_blk* jam_ref_quant_mxfp4(const float* X, int rows, int k) {
    int nb = k / 32;
    jam_ref_mxfp4_blk* o = (jam_ref_mxfp4_blk*) malloc((size_t) rows * nb * sizeof(jam_ref_mxfp4_blk));
    for (int i = 0; i < rows; ++i) for (int b = 0; b < nb; ++b) {
        const float* x = X + (size_t) i * k + b * 32;
        float amax = 0; for (int e = 0; e < 32; ++e) { float a = fabsf(x[e]); if (a > amax) amax = a; }
        jam_ref_mxfp4_blk* p = &o[(size_t) i * nb + b];
        if (amax == 0) { p->e = 127; for (int t = 0; t < 16; ++t) p->qs[t] = 0; continue; }
        int sexp = (int) floorf(log2f(amax)) - 2;   /* E2M1 emax = 2 (OCP) */
        int e = sexp + 127; if (e < 1) e = 1; if (e > 254) e = 254; p->e = (uint8_t) e;
        float scale = exp2f((float)(e - 127));
        for (int t = 0; t < 16; ++t)
            p->qs[t] = (uint8_t)(jam_ref_fp4_code(x[t]/scale) | (jam_ref_fp4_code(x[t+16]/scale) << 4));
    }
    return o;
}

/* ---- NVFP4 (NVIDIA FP4) — GGUF block_nvfp4 { uint8_t d[4] (UE4M3 per-16); uint8_t qs[32] } = 36 bytes,
 * 64 elements. value = kvalues_mxfp4[nibble] · ue4m3(d[s]); no global scale. Mirrors ggml's quant/dequant. */
typedef struct { uint8_t d[4]; uint8_t qs[32]; } jam_ref_nvfp4_blk;
static inline float jam_ref_ue4m3_to_float(uint8_t x) {   /* == ggml_ue4m3_to_fp32 incl. its ×0.5 (kvalues are 2× E2M1) */
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF, m = x & 0x7;
    return 0.5f * (e ? ldexpf(1.0f + (float) m / 8.0f, e - 7) : ldexpf((float) m, -9));
}
static inline uint8_t jam_ref_fp32_to_ue4m3(float v) {    /* nearest UE4M3 code (scales are >= 0) */
    uint8_t best = 0; float bd = 1e30f;
    for (int c = 0; c < 127; ++c) {                       /* skip 0x7F (== 0) */
        float d = fabsf(v - jam_ref_ue4m3_to_float((uint8_t) c));
        if (d < bd) { bd = d; best = (uint8_t) c; }
    }
    return best;
}
static inline uint8_t jam_ref_best_mxfp4(float x, float d) {   /* argmin |kvalues[i]·d - x| (== ggml) */
    static const int8_t kv[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };
    int best = 0; float bd = fabsf(kv[0] * d - x);
    for (int i = 1; i < 16; ++i) { float e = fabsf(kv[i] * d - x); if (e < bd) { bd = e; best = i; } }
    return (uint8_t) best;
}
/* Quantize [rows×k] to GGUF NVFP4 blocks (mirrors quantize_row_nvfp4_ref). Caller frees the returned buffer. */
static inline void* jam_ref_quant_nvfp4(const float* X, int rows, int k) {
    int nblk = k / 64;
    jam_ref_nvfp4_blk* o = (jam_ref_nvfp4_blk*) malloc((size_t) rows * nblk * sizeof(jam_ref_nvfp4_blk));
    for (int i = 0; i < rows; ++i) for (int bb = 0; bb < nblk; ++bb) {
        jam_ref_nvfp4_blk* p = &o[(size_t) i * nblk + bb];
        for (int s = 0; s < 4; ++s) {                     /* 4 sub-blocks of 16 */
            const float* xb = X + (size_t) i * k + bb * 64 + s * 16;
            float amax = 0; for (int e = 0; e < 16; ++e) { float a = fabsf(xb[e]); if (a > amax) amax = a; }
            uint8_t ue = jam_ref_fp32_to_ue4m3(amax / 6.0f);
            p->d[s] = ue;
            float d = jam_ref_ue4m3_to_float(ue); if (d <= 0) d = 1e-30f;
            for (int j = 0; j < 8; ++j) {
                uint8_t x0 = jam_ref_best_mxfp4(xb[j],     d);    /* low  -> elem j     */
                uint8_t x1 = jam_ref_best_mxfp4(xb[8 + j], d);    /* high -> elem j + 8 */
                p->qs[s * 8 + j] = (uint8_t) (x0 | (x1 << 4));
            }
        }
    }
    return o;
}

/* ---- Q4_K: build random VALID super-blocks [rows×k] + the matching dequantized f32 weight (reference).
 * Block = { d(f16) dmin(f16) scales[12] qs[128] } = 144 bytes; value = d·sc·nibble - dmin·min. ---- */
/* wdq = full dequant (scale·nibble - min); wmin = just the `min` term per element (applied with EXACT
 * activation in the VNNI kernel, so the test must split it out). wscale = wdq + wmin. */
static inline uint8_t* jam_ref_make_q4k(int rows, int k, unsigned seed, float* wdq, float* wmin) {
    int sblocks = k / 256;
    uint8_t* W = (uint8_t*) malloc((size_t) rows * sblocks * 144);
    uint32_t st = seed * 2654435761u + 1u;
    #define JAM_RND() (st = st * 1664525u + 1013904223u, (st >> 16))
    for (int i = 0; i < rows; i++) {
        uint8_t* w = W + (size_t) i * sblocks * 144;
        float* dq = wdq + (size_t) i * k;
        float* mq = wmin + (size_t) i * k;
        for (int B = 0; B < sblocks; B++, w += 144, dq += 256, mq += 256) {
            float d = 0.01f + (JAM_RND() % 100) / 2000.0f, dmin = (JAM_RND() % 100) / 4000.0f;
            *(uint16_t*) w = jam_ref_f2h(d); *(uint16_t*)(w + 2) = jam_ref_f2h(dmin);
            d = jam_ref_h2f(jam_ref_f2h(d)); dmin = jam_ref_h2f(jam_ref_f2h(dmin));   /* fp16 round-trip */
            uint8_t sc[8], mn[8];
            for (int t = 0; t < 8; t++) { sc[t] = JAM_RND() % 64; mn[t] = JAM_RND() % 64; }
            uint8_t* b = w + 4;
            for (int j = 0; j < 4; j++) {
                b[j]     = (uint8_t)((sc[j] & 63)    | ((sc[j + 4] >> 4) << 6));
                b[j + 4] = (uint8_t)((mn[j] & 63)    | ((mn[j + 4] >> 4) << 6));
                b[j + 8] = (uint8_t)((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4));
            }
            uint8_t* q = w + 16;
            for (int t = 0; t < 128; t++) q[t] = (uint8_t) JAM_RND();
            for (int g = 0; g < 4; g++) {
                float dl = d*sc[g*2], ml = dmin*mn[g*2], dh = d*sc[g*2+1], mh = dmin*mn[g*2+1];
                for (int e = 0; e < 32; e++) {
                    dq[g*64+e]    = dl * (q[g*32+e] & 0xF) - ml;  mq[g*64+e]    = ml;
                    dq[g*64+32+e] = dh * (q[g*32+e] >> 4) - mh;   mq[g*64+32+e] = mh;
                }
            }
        }
    }
    #undef JAM_RND
    return W;
}

/* Q6_K: ql[128] qh[64] scales[16](s8) d(f16) = 210B; value = d·scale·(qv-32). wmin = 32·d·scale. */
static inline uint8_t* jam_ref_make_q6k(int rows, int k, unsigned seed, float* wdq, float* wmin) {
    int sblocks = k / 256;
    uint8_t* W = (uint8_t*) malloc((size_t) rows * sblocks * 210);
    uint32_t st = seed * 2654435761u + 1u;
    #define JAM_RND() (st = st * 1664525u + 1013904223u, (st >> 16))
    for (int i = 0; i < rows; i++) {
        uint8_t* w = W + (size_t) i * sblocks * 210;
        float* dq = wdq + (size_t) i * k; float* mq = wmin + (size_t) i * k;
        for (int B = 0; B < sblocks; B++, w += 210, dq += 256, mq += 256) {
            uint8_t* ql = w; uint8_t* qh = w + 128; int8_t* sc = (int8_t*)(w + 192);
            for (int t = 0; t < 128; t++) ql[t] = (uint8_t) JAM_RND();
            for (int t = 0; t < 64;  t++) qh[t] = (uint8_t) JAM_RND();
            for (int t = 0; t < 16;  t++) sc[t] = (int8_t)(JAM_RND() % 64);   /* 0..63 */
            float d = 0.01f + (JAM_RND() % 100) / 2000.0f;
            *(uint16_t*)(w + 208) = jam_ref_f2h(d); d = jam_ref_h2f(jam_ref_f2h(d));
            for (int h = 0; h < 2; h++) {
                uint8_t* qlb = ql + h*64; uint8_t* qhb = qh + h*32;
                for (int j = 0; j < 4; j++) for (int l = 0; l < 32; l++) {
                    int qv;
                    switch (j) { case 0: qv=qlb[l]&0xF; break; case 1: qv=qlb[32+l]&0xF; break;
                                 case 2: qv=qlb[l]>>4; break; default: qv=qlb[32+l]>>4; break; }
                    qv |= ((qhb[l] >> (2*j)) & 3) << 4;
                    float ws = d * sc[h*8 + j*2 + l/16];
                    int idx = h*128 + j*32 + l;
                    dq[idx] = ws * (qv - 32); mq[idx] = ws * 32.0f;
                }
            }
        }
    }
    #undef JAM_RND
    return W;
}

/* Q5_K: d dmin scales[12] qh[32] qs[128] = 176B; 5-bit q = qs nibble | (qh bit<<4); value = d·sc·q5 - dmin·min. */
static inline uint8_t* jam_ref_make_q5k(int rows, int k, unsigned seed, float* wdq, float* wmin) {
    int sblocks = k / 256;
    uint8_t* W = (uint8_t*) malloc((size_t) rows * sblocks * 176);
    uint32_t st = seed * 2654435761u + 1u;
    #define JAM_RND() (st = st * 1664525u + 1013904223u, (st >> 16))
    for (int i = 0; i < rows; i++) {
        uint8_t* w = W + (size_t) i * sblocks * 176;
        float* dq = wdq + (size_t) i * k; float* mq = wmin + (size_t) i * k;
        for (int B = 0; B < sblocks; B++, w += 176, dq += 256, mq += 256) {
            float d = 0.01f + (JAM_RND() % 100) / 2000.0f, dmin = (JAM_RND() % 100) / 4000.0f;
            *(uint16_t*) w = jam_ref_f2h(d); *(uint16_t*)(w + 2) = jam_ref_f2h(dmin);
            d = jam_ref_h2f(jam_ref_f2h(d)); dmin = jam_ref_h2f(jam_ref_f2h(dmin));
            uint8_t sc[8], mn[8];
            for (int t = 0; t < 8; t++) { sc[t] = JAM_RND() % 64; mn[t] = JAM_RND() % 64; }
            uint8_t* b = w + 4;
            for (int j = 0; j < 4; j++) {
                b[j]     = (uint8_t)((sc[j] & 63)    | ((sc[j + 4] >> 4) << 6));
                b[j + 4] = (uint8_t)((mn[j] & 63)    | ((mn[j + 4] >> 4) << 6));
                b[j + 8] = (uint8_t)((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4));
            }
            uint8_t* qh = w + 16; uint8_t* qs = w + 48;
            for (int t = 0; t < 32;  t++) qh[t] = (uint8_t) JAM_RND();
            for (int t = 0; t < 128; t++) qs[t] = (uint8_t) JAM_RND();
            for (int g = 0; g < 4; g++) {
                float dl=d*sc[g*2], ml=dmin*mn[g*2], dh=d*sc[g*2+1], mh=dmin*mn[g*2+1];
                for (int e = 0; e < 32; e++) {
                    int qlo = (qs[g*32+e] & 0xF) | (((qh[e] >> (2*g))   & 1) << 4);
                    int qhi = (qs[g*32+e] >> 4)  | (((qh[e] >> (2*g+1)) & 1) << 4);
                    dq[g*64+e]    = dl*qlo - ml; mq[g*64+e]    = ml;
                    dq[g*64+32+e] = dh*qhi - mh; mq[g*64+32+e] = mh;
                }
            }
        }
    }
    #undef JAM_RND
    return W;
}

/* Q4_0: { fp16 d; nibble qs[16] } = 18B; value = d·(nibble-8), no min (wmin=0). Builder signature
 * matches the K-quant suite (wdq + wmin) so it reuses suite_kquant. */
static inline uint8_t* jam_ref_make_q4_0(int rows, int k, unsigned seed, float* wdq, float* wmin) {
    int nb = k / 32;
    uint8_t* W = (uint8_t*) malloc((size_t) rows * nb * 18);
    uint32_t st = seed * 2654435761u + 1u;
    #define JAM_RND() (st = st * 1664525u + 1013904223u, (st >> 16))
    for (int i = 0; i < rows; i++)
        for (int b = 0; b < nb; b++) {
            float d = 0.02f + (JAM_RND() % 100) / 1000.0f;
            uint8_t* w = W + (size_t)(i * nb + b) * 18;
            *(uint16_t*) w = jam_ref_f2h(d); float dd = jam_ref_h2f(jam_ref_f2h(d));
            uint8_t* qs = w + 2;
            float* dq = wdq + (size_t) i * k + b * 32; float* mq = wmin + (size_t) i * k + b * 32;
            for (int e = 0; e < 16; e++) {
                int lo = JAM_RND() & 0xF, hi = JAM_RND() & 0xF;
                qs[e] = (uint8_t)(lo | (hi << 4));
                /* wmin = 8·d: the -8 offset (repack corrects it exactly; wscale = wmin+wdq = d·nibble) */
                dq[e]    = dd * (lo - 8); mq[e]    = 8.f * dd;
                dq[e+16] = dd * (hi - 8); mq[e+16] = 8.f * dd;
            }
        }
    #undef JAM_RND
    return W;
}

#endif /* JAM_REF_H */
