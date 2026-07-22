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
        const uint16_t* w0 = W + (size_t) (r + 0) * ldw;
        const uint16_t* w1 = W + (size_t) (r + 1) * ldw;
        const uint16_t* w2 = W + (size_t) (r + 2) * ldw;
        const uint16_t* w3 = W + (size_t) (r + 3) * ldw;
        const uint16_t* w4 = W + (size_t) (r + 4) * ldw;
        int s = 0;
        for (; s + 5 <= n; s += 5) {
            const uint16_t* x0 = X + (size_t) (s + 0) * k;
            const uint16_t* x1 = X + (size_t) (s + 1) * k;
            const uint16_t* x2 = X + (size_t) (s + 2) * k;
            const uint16_t* x3 = X + (size_t) (s + 3) * k;
            const uint16_t* x4 = X + (size_t) (s + 4) * k;
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = c00;
            __m512 c02 = c00;
            __m512 c03 = c00;
            __m512 c04 = c00;
            __m512 c10 = c00;
            __m512 c11 = c00;
            __m512 c12 = c00;
            __m512 c13 = c00;
            __m512 c14 = c00;
            __m512 c20 = c00;
            __m512 c21 = c00;
            __m512 c22 = c00;
            __m512 c23 = c00;
            __m512 c24 = c00;
            __m512 c30 = c00;
            __m512 c31 = c00;
            __m512 c32 = c00;
            __m512 c33 = c00;
            __m512 c34 = c00;
            __m512 c40 = c00;
            __m512 c41 = c00;
            __m512 c42 = c00;
            __m512 c43 = c00;
            __m512 c44 = c00;
            for (long t = 0; t + 32 <= k; t += 32) {
                __m512bh v0 = (__m512bh) _mm512_loadu_si512((const void*) (w0 + t));
                __m512bh v1 = (__m512bh) _mm512_loadu_si512((const void*) (w1 + t));
                __m512bh v2 = (__m512bh) _mm512_loadu_si512((const void*) (w2 + t));
                __m512bh v3 = (__m512bh) _mm512_loadu_si512((const void*) (w3 + t));
                __m512bh v4 = (__m512bh) _mm512_loadu_si512((const void*) (w4 + t));
                __m512bh u0 = (__m512bh) _mm512_loadu_si512((const void*) (x0 + t));
                c00 = _mm512_dpbf16_ps(c00, v0, u0);
                c01 = _mm512_dpbf16_ps(c01, v1, u0);
                c02 = _mm512_dpbf16_ps(c02, v2, u0);
                c03 = _mm512_dpbf16_ps(c03, v3, u0);
                c04 = _mm512_dpbf16_ps(c04, v4, u0);
                __m512bh u1 = (__m512bh) _mm512_loadu_si512((const void*) (x1 + t));
                c10 = _mm512_dpbf16_ps(c10, v0, u1);
                c11 = _mm512_dpbf16_ps(c11, v1, u1);
                c12 = _mm512_dpbf16_ps(c12, v2, u1);
                c13 = _mm512_dpbf16_ps(c13, v3, u1);
                c14 = _mm512_dpbf16_ps(c14, v4, u1);
                __m512bh u2 = (__m512bh) _mm512_loadu_si512((const void*) (x2 + t));
                c20 = _mm512_dpbf16_ps(c20, v0, u2);
                c21 = _mm512_dpbf16_ps(c21, v1, u2);
                c22 = _mm512_dpbf16_ps(c22, v2, u2);
                c23 = _mm512_dpbf16_ps(c23, v3, u2);
                c24 = _mm512_dpbf16_ps(c24, v4, u2);
                __m512bh u3 = (__m512bh) _mm512_loadu_si512((const void*) (x3 + t));
                c30 = _mm512_dpbf16_ps(c30, v0, u3);
                c31 = _mm512_dpbf16_ps(c31, v1, u3);
                c32 = _mm512_dpbf16_ps(c32, v2, u3);
                c33 = _mm512_dpbf16_ps(c33, v3, u3);
                c34 = _mm512_dpbf16_ps(c34, v4, u3);
                __m512bh u4 = (__m512bh) _mm512_loadu_si512((const void*) (x4 + t));
                c40 = _mm512_dpbf16_ps(c40, v0, u4);
                c41 = _mm512_dpbf16_ps(c41, v1, u4);
                c42 = _mm512_dpbf16_ps(c42, v2, u4);
                c43 = _mm512_dpbf16_ps(c43, v3, u4);
                c44 = _mm512_dpbf16_ps(c44, v4, u4);
            }
            { float* o = C + (size_t) (s + 0) * ldc + r;
              o[0] = _mm512_reduce_add_ps(c00);
              o[1] = _mm512_reduce_add_ps(c01);
              o[2] = _mm512_reduce_add_ps(c02);
              o[3] = _mm512_reduce_add_ps(c03);
              o[4] = _mm512_reduce_add_ps(c04);
            }
            { float* o = C + (size_t) (s + 1) * ldc + r;
              o[0] = _mm512_reduce_add_ps(c10);
              o[1] = _mm512_reduce_add_ps(c11);
              o[2] = _mm512_reduce_add_ps(c12);
              o[3] = _mm512_reduce_add_ps(c13);
              o[4] = _mm512_reduce_add_ps(c14);
            }
            { float* o = C + (size_t) (s + 2) * ldc + r;
              o[0] = _mm512_reduce_add_ps(c20);
              o[1] = _mm512_reduce_add_ps(c21);
              o[2] = _mm512_reduce_add_ps(c22);
              o[3] = _mm512_reduce_add_ps(c23);
              o[4] = _mm512_reduce_add_ps(c24);
            }
            { float* o = C + (size_t) (s + 3) * ldc + r;
              o[0] = _mm512_reduce_add_ps(c30);
              o[1] = _mm512_reduce_add_ps(c31);
              o[2] = _mm512_reduce_add_ps(c32);
              o[3] = _mm512_reduce_add_ps(c33);
              o[4] = _mm512_reduce_add_ps(c34);
            }
            { float* o = C + (size_t) (s + 4) * ldc + r;
              o[0] = _mm512_reduce_add_ps(c40);
              o[1] = _mm512_reduce_add_ps(c41);
              o[2] = _mm512_reduce_add_ps(c42);
              o[3] = _mm512_reduce_add_ps(c43);
              o[4] = _mm512_reduce_add_ps(c44);
            }
        }
        for (; s < n; s++) {
            const uint16_t* xs = X + (size_t) s * k;
            float* o = C + (size_t) s * ldc + r;
            o[0] = bf16_dot(w0, xs, k);
            o[1] = bf16_dot(w1, xs, k);
            o[2] = bf16_dot(w2, xs, k);
            o[3] = bf16_dot(w3, xs, k);
            o[4] = bf16_dot(w4, xs, k);
        }
    }
    for (; r < re; r++) {
        const uint16_t* w = W + (size_t) r * ldw;
        for (int s = 0; s < n; s++)
            C[(size_t) s * ldc + r] = bf16_dot(w, X + (size_t) s * k, k);
    }
}

/* ======================= packed-panel BF16 (vdpbf16ps microkernel) =======================
 * Same structure as the packed F32 path, doubled by the instruction: activations convert AND
 * transpose once into k-pair-major token panels (xp[p][t/2][32 tokens x pair]), and the 8x32
 * microkernel broadcasts a 32-bit WEIGHT PAIR (two adjacent bf16) against 2 panel vectors -
 * each vdpbf16ps covers two k-steps for 16 tokens. 16 dp per ~10 load-uops. */

static inline uint16_t jam_f2bf_rne(float v) {   /* round-to-nearest-even, matches vcvtneps2bf16 */
    union { float f; uint32_t u; } x; x.f = v;
    uint32_t r = x.u + 0x7FFF + ((x.u >> 16) & 1);
    return (uint16_t) (r >> 16);
}

/* phase 1: convert+transpose token-panels of 32 into xp (uint16 pairs: [t/2][j*2 + (t&1)]). */
void jam_bf16_pack_avx512bf16(void* arg, int pb, int pe, int tid) {
    (void) tid;
    const jam_bf16_job* J = (const jam_bf16_job*) arg;
    const float* X = J->x;
    const long k = J->k, ldx = J->ldx;
    const int n = J->n;
    for (int p = pb; p < pe; p++) {
        uint16_t* out = J->xb + (size_t) p * 32 * k;
        int j0 = p * 32;
        int cols = n - j0 < 32 ? n - j0 : 32;
        for (int j = 0; j < cols; j++) {
            const float* xs = X + (size_t)(j0 + j) * ldx;
            for (long t = 0; t < k; t++)
                out[(size_t)(t / 2) * 64 + (size_t) j * 2 + (t & 1)] = jam_f2bf_rne(xs[t]);
        }
        for (int j = cols; j < 32; j++)
            for (long t = 0; t < k; t++)
                out[(size_t)(t / 2) * 64 + (size_t) j * 2 + (t & 1)] = 0;
    }
}

/* phase 2: 8 weight rows x one 32-token panel. Broadcast the row's bf16 pair, dp both halves. */
void jam_mm_bf16p_avx512bf16(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_bf16_job* J = (const jam_bf16_job*) arg;
    const uint16_t* W = J->w;
    float* C = (float*) J->c;
    const long ldw = J->ldw, ldc = J->ldc, k = J->k;
    const int n = J->n, npanels = (n + 31) / 32;
    for (int p = 0; p < npanels; p++) {
        const uint16_t* xp = J->xb + (size_t) p * 32 * k;
        int j0 = p * 32;
        int cols = n - j0 < 32 ? n - j0 : 32;
        int r = rb;
        for (; r + 8 <= re; r += 8) {
            const uint16_t* w[8];
            for (int a = 0; a < 8; ++a) w[a] = W + (size_t)(r + a) * ldw;
            __m512 c0a = _mm512_setzero_ps(), c0b = c0a, c1a = c0a, c1b = c0a,
                   c2a = c0a, c2b = c0a, c3a = c0a, c3b = c0a,
                   c4a = c0a, c4b = c0a, c5a = c0a, c5b = c0a,
                   c6a = c0a, c6b = c0a, c7a = c0a, c7b = c0a;
            for (long t = 0; t + 2 <= k; t += 2) {
                const uint16_t* xr = xp + (size_t)(t / 2) * 64;
                __m512bh xa = (__m512bh) _mm512_loadu_si512((const void*) xr);
                __m512bh xb = (__m512bh) _mm512_loadu_si512((const void*) (xr + 32));
                __m512bh b0 = (__m512bh) _mm512_set1_epi32(*(const int*) (w[0] + t));
                __m512bh b1 = (__m512bh) _mm512_set1_epi32(*(const int*) (w[1] + t));
                c0a = _mm512_dpbf16_ps(c0a, b0, xa); c0b = _mm512_dpbf16_ps(c0b, b0, xb);
                c1a = _mm512_dpbf16_ps(c1a, b1, xa); c1b = _mm512_dpbf16_ps(c1b, b1, xb);
                __m512bh b2 = (__m512bh) _mm512_set1_epi32(*(const int*) (w[2] + t));
                __m512bh b3 = (__m512bh) _mm512_set1_epi32(*(const int*) (w[3] + t));
                c2a = _mm512_dpbf16_ps(c2a, b2, xa); c2b = _mm512_dpbf16_ps(c2b, b2, xb);
                c3a = _mm512_dpbf16_ps(c3a, b3, xa); c3b = _mm512_dpbf16_ps(c3b, b3, xb);
                __m512bh b4 = (__m512bh) _mm512_set1_epi32(*(const int*) (w[4] + t));
                __m512bh b5 = (__m512bh) _mm512_set1_epi32(*(const int*) (w[5] + t));
                c4a = _mm512_dpbf16_ps(c4a, b4, xa); c4b = _mm512_dpbf16_ps(c4b, b4, xb);
                c5a = _mm512_dpbf16_ps(c5a, b5, xa); c5b = _mm512_dpbf16_ps(c5b, b5, xb);
                __m512bh b6 = (__m512bh) _mm512_set1_epi32(*(const int*) (w[6] + t));
                __m512bh b7 = (__m512bh) _mm512_set1_epi32(*(const int*) (w[7] + t));
                c6a = _mm512_dpbf16_ps(c6a, b6, xa); c6b = _mm512_dpbf16_ps(c6b, b6, xb);
                c7a = _mm512_dpbf16_ps(c7a, b7, xa); c7b = _mm512_dpbf16_ps(c7b, b7, xb);
            }
            /* lane j of the "a" accumulators = token j0+j (pairs interleave j*2, j*2+1 across the
             * two halves: lanes 0..15 of xa are tokens 0..15's pairs; xb tokens 16..31). */
            float tile[16][16] __attribute__((aligned(64)));
            _mm512_store_ps(tile[0],  c0a); _mm512_store_ps(tile[1],  c0b);
            _mm512_store_ps(tile[2],  c1a); _mm512_store_ps(tile[3],  c1b);
            _mm512_store_ps(tile[4],  c2a); _mm512_store_ps(tile[5],  c2b);
            _mm512_store_ps(tile[6],  c3a); _mm512_store_ps(tile[7],  c3b);
            _mm512_store_ps(tile[8],  c4a); _mm512_store_ps(tile[9],  c4b);
            _mm512_store_ps(tile[10], c5a); _mm512_store_ps(tile[11], c5b);
            _mm512_store_ps(tile[12], c6a); _mm512_store_ps(tile[13], c6b);
            _mm512_store_ps(tile[14], c7a); _mm512_store_ps(tile[15], c7b);
            for (int j = 0; j < cols; j++) {
                float* o = C + (size_t)(j0 + j) * ldc + r;
                for (int a = 0; a < 8; ++a) o[a] = tile[a * 2 + (j >> 4)][j & 15];
            }
        }
        for (; r < re; r++) {                        /* row tail: 1 x 32 */
            const uint16_t* w = W + (size_t) r * ldw;
            __m512 ca = _mm512_setzero_ps(), cb = ca;
            for (long t = 0; t + 2 <= k; t += 2) {
                const uint16_t* xr = xp + (size_t)(t / 2) * 64;
                __m512bh b = (__m512bh) _mm512_set1_epi32(*(const int*) (w + t));
                ca = _mm512_dpbf16_ps(ca, b, (__m512bh) _mm512_loadu_si512((const void*) xr));
                cb = _mm512_dpbf16_ps(cb, b, (__m512bh) _mm512_loadu_si512((const void*) (xr + 32)));
            }
            float tile[32] __attribute__((aligned(64)));
            _mm512_store_ps(tile, ca); _mm512_store_ps(tile + 16, cb);
            for (int j = 0; j < cols; j++) C[(size_t)(j0 + j) * ldc + r] = tile[j >> 4 ? 16 + (j & 15) : j];
        }
    }
}
