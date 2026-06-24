/* Portable-C kernels — the always-available floor (no -m flags, compiles anywhere).
 * Per-ISA kernels live in sibling TUs (jam_kernels_avx2.c, _avx512.c, _neon.c, ...) compiled with
 * their own flags and bound at create; this scalar one is the fallback and the reference. */
#include "jam_internal.h"
#include <stddef.h>   /* size_t */
#include <stdint.h>
#include <math.h>     /* fabsf, lrintf */
#include "jam_mxfp4.h"
#include "jam_kquant.h"

/* C[i, :] = A[i, :] @ Bᵀ  for i in [row_begin, row_end).  C[i,j] = dot(A row i, B row j). */
void jam_mm_f32_generic(void* arg, int row_begin, int row_end, int tid) {
    (void) tid;
    const jam_mm_job* j = (const jam_mm_job*) arg;
    const float* A = (const float*) j->a;
    const float* B = (const float*) j->b;
    float*       C = (float*)       j->c;
    const int n = j->n, k = j->k;

    const int ldc = j->ldc;
    for (int i = row_begin; i < row_end; ++i) {
        const float* ar = A + (size_t) i * j->lda;
        for (int t = 0; t < n; ++t) {
            const float* br = B + (size_t) t * j->ldb;
            float s = 0.0f;
            for (int kk = 0; kk < k; ++kk) s += ar[kk] * br[kk];   /* compiler auto-vectorizes */
            C[(size_t) t * ldc + i] = s;                           /* token-major: feature i contiguous */
        }
    }
}

/* ---- Q8_0 (weight) @ F32 (activation) -> F32, portable floor ----
 * Dequantize the weight block (d * qs) and float-dot the F32 activation directly — no requant, no
 * scratch. The inner 32-wide loop is plain C the compiler vectorizes (int8->float convert + fmadd). */

typedef struct { uint16_t d; int8_t qs[32]; } jam_blk_q8_0;   /* 34 bytes (matches GGML block_q8_0) */

/* portable IEEE half -> float (no F16C dependency) */
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

void jam_mm_q8_0_f32_generic(void* arg, int row_begin, int row_end, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* A = (const char*) J->a;
    const float* B = (const float*) J->b;
    float*       C = (float*)       J->c;
    const int n = J->n, nb = J->nb, ldb = J->ldb, ldc = J->ldc;
    const size_t wrow = (size_t)(J->lda / 32);                 /* weight-row stride in blocks */

    for (int i = row_begin; i < row_end; ++i) {
        const jam_blk_q8_0* arow = (const jam_blk_q8_0*) (A + (size_t) i * wrow * sizeof(jam_blk_q8_0));
        for (int t = 0; t < n; ++t) {
            const float* brow = B + (size_t) t * ldb;
            float acc = 0.0f;
            for (int blk = 0; blk < nb; ++blk) {
                const jam_blk_q8_0* wb = &arow[blk];
                const float* b = brow + (size_t) blk * 32;
                float d = jam_half2float(wb->d);
                float s = 0.0f;
                for (int e = 0; e < 32; ++e) s += (float) wb->qs[e] * b[e];   /* vectorizes */
                acc += d * s;
            }
            C[(size_t) t*ldc+i] = acc;
        }
    }
}

/* Shared phase-1 requant for the SIMD Q8_0 paths (AVX-512-VNNI and AVX2): F32 activation rows
 * [rb,re) -> int8 (per-32-block scale dB in ad). Scalar/portable; the compiler vectorizes it. */
void jam_q8_0_requant(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const float* B = (const float*) J->b;
    const int ldb = J->ldb, k = J->k, nb = J->nb;
    for (int j = rb; j < re; ++j) {
        const float* brow = B + (size_t) j * ldb;
        int8_t* q = J->aq + (size_t) j * k;
        for (int blk = 0; blk < nb; ++blk) {
            const float* b = brow + (size_t) blk * 32;
            float amax = 0.0f;
            for (int e = 0; e < 32; ++e) { float a = fabsf(b[e]); if (a > amax) amax = a; }
            float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
            for (int e = 0; e < 32; ++e) {
                int v = (int) lrintf(b[e] * id);
                if (v > 127) v = 127; else if (v < -128) v = -128;
                q[(size_t) blk * 32 + e] = (int8_t) v;
            }
            J->ad[(size_t) j * nb + blk] = d;
        }
    }
}

/* MXFP4 weight @ F32 activation -> F32, portable reference. Decodes each FP4 nibble to value×2 and folds
 * the ×½ into the block scale (jam_mxfp4_dhalf); dots the EXACT float activation (no requant). */
void jam_mm_mxfp4_f32_generic(void* arg, int rb, int re, int tid) {
    (void) tid;
    static const int8_t kv[16] = { JAM_MXFP4_CODES };
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char*  W = (const char*)  J->a;     /* MXFP4 weights */
    const float* A = (const float*) J->b;     /* F32 activations */
    float* C = (float*) J->c;
    const int ldc = J->ldc, ldb = J->ldb, n = J->n, k = J->k, nb = J->nb;
    const size_t wrow = (size_t)(J->lda / 32);
    for (int i = rb; i < re; ++i) {
        const jam_mxfp4_blk* wr = (const jam_mxfp4_blk*) (W + (size_t) i * wrow * sizeof(jam_mxfp4_blk));
        for (int j = 0; j < n; ++j) {
            const float* arow = A + (size_t) j * ldb;
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const jam_mxfp4_blk* w = &wr[b];
                float dW = jam_mxfp4_dhalf(w->e);
                const float* aa = arow + (size_t) b * 32;
                float s = 0.0f;
                for (int t = 0; t < 16; ++t) {
                    s += (float) kv[w->qs[t] & 0x0F] * aa[t];
                    s += (float) kv[w->qs[t] >> 4]   * aa[t + 16];
                }
                acc += dW * s;
            }
            C[(size_t) j*ldc+i] = acc;
        }
    }
}

/* Q4_K weight @ F32 activation -> F32, portable floor (and the test reference). Dequant per super-block:
 * value = d·sc·nibble - dmin·mn; exact float dot. Reuses jam_q8_job (a=weight, b=activation). */
void jam_mm_q4k_f32_generic(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char*  W = (const char*)  J->a;     /* Q4_K weights */
    const float* A = (const float*) J->b;     /* F32 activations */
    float* C = (float*) J->c;
    const int ldc = J->ldc, ldb = J->ldb, n = J->n, k = J->k;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q4K_BYTES;
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j = 0; j < n; ++j) {
            const float* xrow = A + (size_t) j * ldb;
            const uint8_t* w = wrow; const float* x = xrow;
            float acc = 0.0f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q4K_BYTES, x += JAM_QKK) {
                float d = jam_half2float(*(const uint16_t*) w), dmin = jam_half2float(*(const uint16_t*)(w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* q = w + 16;
                for (int g = 0; g < 4; ++g) {
                    float dl = d*sc[g*2], ml = dmin*mn[g*2], dh = d*sc[g*2+1], mh = dmin*mn[g*2+1];
                    for (int e = 0; e < 32; ++e) {
                        acc += (dl * (q[g*32+e] & 0xF) - ml) * x[g*64+e];
                        acc += (dh * (q[g*32+e] >> 4) - mh) * x[g*64+32+e];
                    }
                }
            }
            C[(size_t) j*ldc+i] = acc;
        }
    }
}

/* Q6_K weight @ F32 activation -> F32, portable floor + reference. value = d·scale·(qv-32). */
void jam_mm_q6k_f32_generic(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char*  W = (const char*)  J->a;
    const float* A = (const float*) J->b;
    float* C = (float*) J->c;
    const int ldc = J->ldc, ldb = J->ldb, n = J->n, k = J->k;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q6K_BYTES;
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j = 0; j < n; ++j) {
            const float* xrow = A + (size_t) j * ldb;
            const uint8_t* w = wrow; const float* x = xrow;
            float acc = 0.0f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q6K_BYTES, x += JAM_QKK) {
                const uint8_t* ql = w; const uint8_t* qh = w + 128;
                const int8_t* sc = (const int8_t*) (w + 192);
                float d = jam_half2float(*(const uint16_t*)(w + 208));
                for (int h = 0; h < 2; ++h) {
                    const uint8_t* qlb = ql + h*64; const uint8_t* qhb = qh + h*32;
                    for (int g = 0; g < 4; ++g)
                        for (int l = 0; l < 32; ++l) {
                            int qv;
                            switch (g) { case 0: qv=qlb[l]&0xF; break; case 1: qv=qlb[32+l]&0xF; break;
                                         case 2: qv=qlb[l]>>4; break; default: qv=qlb[32+l]>>4; break; }
                            qv |= ((qhb[l] >> (2*g)) & 3) << 4;
                            acc += d * sc[h*8 + g*2 + l/16] * (qv - 32) * x[h*128 + g*32 + l];
                        }
                }
            }
            C[(size_t) j*ldc+i] = acc;
        }
    }
}

/* Q5_K weight @ F32 -> F32, portable floor (jinferjni.c has no VNNI path for Q5_K). 5-bit quant =
 * qs nibble | (qh bit << 4); value = d·sc·q5 - dmin·min. Block = d dmin scales[12] qh[32] qs[128] = 176B. */
void jam_mm_q5k_f32_generic(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char*  W = (const char*)  J->a;
    const float* A = (const float*) J->b;
    float* C = (float*) J->c;
    const int ldc = J->ldc, ldb = J->ldb, n = J->n, k = J->k;
    const int sblocks = k / JAM_QKK;
    const size_t w_stride = (size_t) sblocks * JAM_Q5K_BYTES;
    for (int i = rb; i < re; ++i) {
        const uint8_t* wrow = (const uint8_t*) (W + (size_t) i * w_stride);
        for (int j = 0; j < n; ++j) {
            const float* xrow = A + (size_t) j * ldb;
            const uint8_t* w = wrow; const float* x = xrow;
            float acc = 0.0f;
            for (int B = 0; B < sblocks; ++B, w += JAM_Q5K_BYTES, x += JAM_QKK) {
                float d = jam_half2float(*(const uint16_t*) w), dmin = jam_half2float(*(const uint16_t*)(w + 2));
                uint8_t sc[8], mn[8]; jam_q4k_scales_mins(w + 4, sc, mn);
                const uint8_t* qh = w + 16; const uint8_t* qs = w + 48;
                for (int g = 0; g < 4; ++g) {
                    float dl=d*sc[g*2], ml=dmin*mn[g*2], dh=d*sc[g*2+1], mh=dmin*mn[g*2+1];
                    const uint8_t* q = qs + g*32; const float* xlo = x + g*64; const float* xhi = xlo + 32;
                    for (int e = 0; e < 32; ++e) {
                        int qlo = (q[e] & 0xF) | (((qh[e] >> (2*g))   & 1) << 4);
                        int qhi = (q[e] >> 4)  | (((qh[e] >> (2*g+1)) & 1) << 4);
                        acc += (dl * qlo - ml) * xlo[e];
                        acc += (dh * qhi - mh) * xhi[e];
                    }
                }
            }
            C[(size_t) j*ldc+i] = acc;
        }
    }
}

/* Q4_0 weight @ F32 -> F32, portable floor + reference. block = { fp16 d; nibble qs[16] } = 18B;
 * value = d·(nibble-8); qs[j] low nibble = element j, high = element j+16. */
typedef struct __attribute__((packed)) { uint16_t d; uint8_t qs[16]; } jam_blk_q4_0;
void jam_mm_q4_0_f32_generic(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const float* A = (const float*) J->b;
    float* C = (float*) J->c;
    const int ldc = J->ldc, ldb = J->ldb, n = J->n, nb = J->nb;
    const size_t wrow = (size_t)(J->lda / 32);
    for (int i = rb; i < re; ++i) {
        const jam_blk_q4_0* arow = (const jam_blk_q4_0*) (W + (size_t) i * wrow * sizeof(jam_blk_q4_0));
        for (int j = 0; j < n; ++j) {
            const float* brow = A + (size_t) j * ldb;
            float acc = 0.0f;
            for (int blk = 0; blk < nb; ++blk) {
                const jam_blk_q4_0* wb = &arow[blk];
                float d = jam_half2float(wb->d);
                const float* b = brow + (size_t) blk * 32;
                float s = 0.0f;
                for (int e = 0; e < 16; ++e) {
                    s += (float)((wb->qs[e] & 0xF) - 8) * b[e];
                    s += (float)((wb->qs[e] >> 4)  - 8) * b[e + 16];
                }
                acc += d * s;
            }
            C[(size_t) j*ldc+i] = acc;
        }
    }
}

/* ---- F16 / BF16 DENSE weight @ F32 -> F32, portable floor (any k; the AVX-512 path needs k%16==0).
 * Weight converted per element; output token-major C[s*ldc + r]. ---- */
static inline float jam_bf16_to_float(uint16_t h) {
    union { uint32_t u; float f; } v; v.u = (uint32_t) h << 16; return v.f;
}
#define JAM_DENSE_GENERIC(NAME, CVT)                                                  \
void NAME(void* arg, int rb, int re, int tid) {                                      \
    (void) tid;                                                                      \
    const jam_mm_job* J = (const jam_mm_job*) arg;                                   \
    const uint16_t* W = (const uint16_t*) J->a;                                      \
    const float* X = (const float*) J->b;                                            \
    float* C = (float*) J->c;                                                        \
    const int ldw = J->lda, ldx = J->ldb, ldc = J->ldc, n = J->n, k = J->k;          \
    for (int r = rb; r < re; ++r) {                                                  \
        const uint16_t* w = W + (size_t) r * ldw;                                    \
        for (int s = 0; s < n; ++s) {                                                \
            const float* x = X + (size_t) s * ldx;                                   \
            float acc = 0.0f;                                                        \
            for (int t = 0; t < k; ++t) acc += CVT(w[t]) * x[t];                     \
            C[(size_t) s * ldc + r] = acc;                                           \
        }                                                                            \
    }                                                                                \
}
JAM_DENSE_GENERIC(jam_mm_f16_f32_generic,  jam_half2float)
JAM_DENSE_GENERIC(jam_mm_bf16_f32_generic, jam_bf16_to_float)
