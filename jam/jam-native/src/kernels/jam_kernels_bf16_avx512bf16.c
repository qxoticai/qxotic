/* AVX512-BF16 dense kernel — this TU only, built with -mavx512bf16 (Zen 4+/Cooper Lake+); bound at
 * create only when the CPU reports avx512bf16 (orthogonal to the ISA ladder, like AVX-VNNI).
 *
 * BF16 weight @ F32 activation -> F32, in two phases:
 *   phase 1 (jam_bf16_cvt_avx512bf16): convert the activation rows to BF16 ONCE into context
 *     scratch (contiguous k per row) — vcvtne2ps2bf16 packs 32 floats -> 32 bf16 per op.
 *   phase 2 (jam_mm_bf16_avx512bf16): 4×4 register tile over weight rows; each k-step of 32 does
 *     16 vdpbf16ps, and every vdpbf16ps is 32 bf16 MACs into f32 lanes — twice the MAC rate of the
 *     FMA convert-tile it replaces, at half the activation-load bytes.
 * The activation-side bf16 rounding matches llama.cpp's BF16 path (tinyBLAS converts A the same
 * way); the weights are bf16 already, so the product picks up ~one extra ulp of bf16 noise.
 * k must be a multiple of 32 (else dispatch falls back to the convert-tile kernel). */
#include "jam_internal.h"
#include <immintrin.h>

/* phase 1: rows [rb,re) of the n×k F32 activation X -> bf16 scratch XB (same row order, stride k) */
void jam_bf16_cvt_avx512bf16(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_bf16_job* J = (const jam_bf16_job*) arg;
    const float* X = J->x;
    uint16_t* XB = J->xb;
    const long k = J->k, ldx = J->ldx;
    for (int s = rb; s < re; s++) {
        const float* xs = X + (size_t) s * ldx;
        uint16_t* os = XB + (size_t) s * k;
        long t = 0;
        for (; t + 32 <= k; t += 32) {
            __m512 lo = _mm512_loadu_ps(xs + t), hi = _mm512_loadu_ps(xs + t + 16);
            _mm512_storeu_si512((void*) (os + t), (__m512i) _mm512_cvtne2ps_pbh(hi, lo));
        }
        for (; t < k; t++) {   /* tail (dispatch gates k%32==0, but stay correct) */
            union { float f; uint32_t u; } v; v.f = xs[t];
            os[t] = (uint16_t) (v.u >> 16);
        }
    }
}

/* one bf16 weight row · one bf16 activation row */
static inline float bf16_dot(const uint16_t* w, const uint16_t* x, long k) {
    __m512 acc = _mm512_setzero_ps();
    for (long t = 0; t + 32 <= k; t += 32)
        acc = _mm512_dpbf16_ps(acc,
                               (__m512bh) _mm512_loadu_si512((const void*) (w + t)),
                               (__m512bh) _mm512_loadu_si512((const void*) (x + t)));
    return _mm512_reduce_add_ps(acc);
}

/* phase 2: weight rows [rb,re) × all n activation columns, 5×5 vdpbf16ps tile (25 acc + 5 w +
 * 1 live x = 31 zmm). Per 32-elem k-step: 10 loads feed 25 dpbf16 ops (32 MACs each) - the same
 * flops-per-L2-byte reasoning as the F32 5x5 tile, doubled by the bf16 MAC rate. Weights stream
 * from DRAM once; the converted activations (n·k bf16) stay cache-resident, shared by workers. */
void jam_mm_bf16_avx512bf16(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_bf16_job* J = (const jam_bf16_job*) arg;
    const uint16_t* W = J->w;
    const uint16_t* X = J->xb;
    float* C = (float*) J->c;
    const long ldw = J->ldw, ldc = J->ldc, k = J->k;
    const int n = J->n;
    int r = rb;
    for (; r + 5 <= re; r += 5) {
        const uint16_t* w[5];
        for (int a = 0; a < 5; ++a) w[a] = W + (size_t) (r + a) * ldw;
        int s = 0;
        for (; s + 5 <= n; s += 5) {
            const uint16_t* x[5];
            for (int b = 0; b < 5; ++b) x[b] = X + (size_t) (s + b) * k;
            __m512 acc[5][5];
            for (int b = 0; b < 5; ++b) for (int a = 0; a < 5; ++a) acc[b][a] = _mm512_setzero_ps();
            for (long t = 0; t + 32 <= k; t += 32) {
                __m512bh wv[5];
                for (int a = 0; a < 5; ++a)
                    wv[a] = (__m512bh) _mm512_loadu_si512((const void*) (w[a] + t));
                for (int b = 0; b < 5; ++b) {
                    __m512bh xv = (__m512bh) _mm512_loadu_si512((const void*) (x[b] + t));
                    for (int a = 0; a < 5; ++a) acc[b][a] = _mm512_dpbf16_ps(acc[b][a], wv[a], xv);
                }
            }
            for (int b = 0; b < 5; ++b) {
                float* o = C + (size_t) (s + b) * ldc + r;
                for (int a = 0; a < 5; ++a) o[a] = _mm512_reduce_add_ps(acc[b][a]);
            }
        }
        for (; s < n; s++) {
            const uint16_t* xs = X + (size_t) s * k;
            float* o = C + (size_t) s * ldc + r;
            for (int a = 0; a < 5; ++a) o[a] = bf16_dot(w[a], xs, k);
        }
    }
    for (; r < re; r++) {
        const uint16_t* w = W + (size_t) r * ldw;
        for (int s = 0; s < n; s++)
            C[(size_t) s * ldc + r] = bf16_dot(w, X + (size_t) s * k, k);
    }
}
