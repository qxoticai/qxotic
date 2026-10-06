/* SSE3 (true SSE3: no SSSE3 maddubs, no F16C) 128-bit int8 GEMM - the pre-AVX2 x86 floor, replacing the
 * generic dequant-to-float path on machines without AVX2. Built with -msse3, dispatched when the CPU has
 * SSE3 but not AVX2. The q128 engine + the 128-bit decoders cover Q8_0, Q4_0, Q5_0 and MXFP4; the dense
 * float weights (F32, F16, BF16) get a 4x4 tile at the bottom. */
#include <pmmintrin.h>
#include <stdint.h>
#include "jam_internal.h"
#include "jam_decode_x86_128.h"

#define JAM_BLK     jam_q8_blk
#define JAM_DECODE  jam_decode_q8_0_128
#define JAM_MM_NAME jam_mm_q8_0_sse3
#include "jam_gemm_q128.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_q4_0_blk
#define JAM_DECODE  jam_decode_q4_0_128
#define JAM_MM_NAME jam_mm_q4_0_sse3
#include "jam_gemm_q128.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_q5_0_blk
#define JAM_DECODE  jam_decode_q5_0_128
#define JAM_MM_NAME jam_mm_q5_0_sse3
#include "jam_gemm_q128.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK     jam_mxfp4_blk
#define JAM_DECODE  jam_decode_mxfp4_128
#define JAM_MM_NAME jam_mm_mxfp4_sse3
#include "jam_gemm_q128.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

/* ---- dense F32 / F16 / BF16 weight @ F32 -> F32 (the pre-AVX2 float floor) ----
 * Every output is one serial sum over k in element order (as the portable loop), so speed comes from many
 * independent sums: a 4-row x 4-column tile with the ROWS in the lanes. Each 4-k step loads 4 elements of
 * the 4 rows and transposes them, so vector q holds element t+q of every row, then adds q = 0..3 in order
 * into each column's accumulator: lane r performs exactly row r's scalar adds. Written with intrinsics
 * because the portable tiled loop, left to the vectorizers, came out 40% apart between gcc and clang. */

/* 4 consecutive weights -> f32, the portable floor's conversions bit for bit */
static inline __m128 jam_ld4_f32(const float* w) { return _mm_loadu_ps(w); }
static inline __m128 jam_ld4_bf16(const uint16_t* w) {
    const __m128i h = _mm_loadl_epi64((const __m128i*) w);
    return _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), h));   /* bf16 << 16 */
}
/* fp16 -> f32 in integer ops, jam_half2float's result for every input (NaN payloads included): the
 * exponent rebiases by 112, inf/NaN by 224, and a subnormal m·2^-24 is (2^-14 + m·2^-24) - 2^-14 in
 * float, which is exact. */
static inline __m128 jam_ld4_f16(const uint16_t* w) {
    const __m128i h  = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) w), _mm_setzero_si128());
    const __m128i em = _mm_and_si128(h, _mm_set1_epi32(0x7FFF));
    const __m128i e  = _mm_and_si128(em, _mm_set1_epi32(0x7C00));
    const __m128i o  = _mm_slli_epi32(em, 13);
    const __m128i normal = _mm_add_epi32(o, _mm_set1_epi32(112 << 23));
    const __m128i infnan = _mm_add_epi32(o, _mm_set1_epi32(224 << 23));
    const __m128i sub = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_add_epi32(o, _mm_set1_epi32(113 << 23))),
                                                    _mm_castsi128_ps(_mm_set1_epi32(113 << 23))));
    const __m128i is_inf = _mm_cmpeq_epi32(e, _mm_set1_epi32(0x7C00));
    const __m128i is_sub = _mm_cmpeq_epi32(e, _mm_setzero_si128());
    __m128i r = _mm_or_si128(_mm_and_si128(is_inf, infnan), _mm_andnot_si128(is_inf, normal));
    r = _mm_or_si128(_mm_and_si128(is_sub, sub), _mm_andnot_si128(is_sub, r));
    return _mm_castsi128_ps(_mm_or_si128(r, _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x8000)), 16)));
}
static inline float jam_cvt_f32(float v) { return v; }
static inline float jam_cvt_bf16(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; __builtin_memcpy(&f, &u, 4); return f; }

/* acc += v_q · x[q] for q = 0..3, in order */
#define JAM_DENSE_STEP(acc, xc, v0, v1, v2, v3) do { const __m128 xv = _mm_loadu_ps(xc);  \
        acc = _mm_add_ps(acc, _mm_mul_ps(v0, _mm_shuffle_ps(xv, xv, 0x00)));              \
        acc = _mm_add_ps(acc, _mm_mul_ps(v1, _mm_shuffle_ps(xv, xv, 0x55)));              \
        acc = _mm_add_ps(acc, _mm_mul_ps(v2, _mm_shuffle_ps(xv, xv, 0xAA)));              \
        acc = _mm_add_ps(acc, _mm_mul_ps(v3, _mm_shuffle_ps(xv, xv, 0xFF))); } while (0)

/* NAME: kernel; T: weight element; LD4: 4 weights -> f32 vector; CVT: 1 weight -> f32 */
#define JAM_DENSE_SSE3(NAME, T, LD4, CVT)                                                                   \
void NAME(void* arg, int rb, int re, int tid) {                                                            \
    (void) tid;                                                                                            \
    const jam_mm_job* J = (const jam_mm_job*) arg;                                                         \
    const T* W = (const T*) J->a; const float* X = (const float*) J->b; float* C = (float*) J->c;          \
    const int ldw = J->lda, ldx = J->ldb, ldc = J->ldc, n = J->n, k = J->k, k4 = k & ~3;                    \
    int r = rb;                                                                                            \
    for (; r + 4 <= re; r += 4) {                                                                          \
        const T* w0 = W + (size_t) r * ldw;  const T* w1 = w0 + ldw;                                       \
        const T* w2 = w1 + ldw;              const T* w3 = w2 + ldw;                                       \
        for (int s = 0; s < n; s += 4) {                                                                   \
            const int nc = n - s < 4 ? n - s : 4;              /* missing columns re-read column s */      \
            const float* x0 = X + (size_t) s * ldx;                                                        \
            const float* x1 = nc > 1 ? x0 + ldx : x0;                                                      \
            const float* x2 = nc > 2 ? x0 + 2 * (size_t) ldx : x0;                                         \
            const float* x3 = nc > 3 ? x0 + 3 * (size_t) ldx : x0;                                         \
            __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps(), a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps(); \
            int t = 0;                                                                                     \
            for (; t < k4; t += 4) {                                                                       \
                __m128 v0 = LD4(w0 + t), v1 = LD4(w1 + t), v2 = LD4(w2 + t), v3 = LD4(w3 + t);             \
                _MM_TRANSPOSE4_PS(v0, v1, v2, v3);              /* vq = element t+q of rows 0..3 */        \
                JAM_DENSE_STEP(a0, x0 + t, v0, v1, v2, v3); JAM_DENSE_STEP(a1, x1 + t, v0, v1, v2, v3);    \
                JAM_DENSE_STEP(a2, x2 + t, v0, v1, v2, v3); JAM_DENSE_STEP(a3, x3 + t, v0, v1, v2, v3);    \
            }                                                                                              \
            for (; t < k; ++t) {                                                                           \
                const __m128 v = _mm_setr_ps(CVT(w0[t]), CVT(w1[t]), CVT(w2[t]), CVT(w3[t]));              \
                a0 = _mm_add_ps(a0, _mm_mul_ps(v, _mm_set1_ps(x0[t])));                                    \
                a1 = _mm_add_ps(a1, _mm_mul_ps(v, _mm_set1_ps(x1[t])));                                    \
                a2 = _mm_add_ps(a2, _mm_mul_ps(v, _mm_set1_ps(x2[t])));                                    \
                a3 = _mm_add_ps(a3, _mm_mul_ps(v, _mm_set1_ps(x3[t])));                                    \
            }                                                                                              \
            _mm_storeu_ps(C + (size_t) s * ldc + r, a0);       /* token-major: rows r..r+3 contiguous */  \
            if (nc > 1) _mm_storeu_ps(C + (size_t) (s + 1) * ldc + r, a1);                                 \
            if (nc > 2) _mm_storeu_ps(C + (size_t) (s + 2) * ldc + r, a2);                                 \
            if (nc > 3) _mm_storeu_ps(C + (size_t) (s + 3) * ldc + r, a3);                                 \
        }                                                                                                  \
    }                                                                                                      \
    for (; r < re; ++r) {                      /* row tail: the portable floor's loop */                   \
        const T* w = W + (size_t) r * ldw;                                                                 \
        for (int s = 0; s < n; ++s) {                                                                      \
            const float* x = X + (size_t) s * ldx;                                                         \
            float acc = 0.0f;                                                                              \
            for (int t = 0; t < k; ++t) acc += CVT(w[t]) * x[t];                                           \
            C[(size_t) s * ldc + r] = acc;                                                                 \
        }                                                                                                  \
    }                                                                                                      \
}
JAM_DENSE_SSE3(jam_mm_f32_sse3,  float,    jam_ld4_f32,  jam_cvt_f32)
JAM_DENSE_SSE3(jam_mm_f16_sse3,  uint16_t, jam_ld4_f16,  jam_half2float)
JAM_DENSE_SSE3(jam_mm_bf16_sse3, uint16_t, jam_ld4_bf16, jam_cvt_bf16)
#undef JAM_DENSE_SSE3
#undef JAM_DENSE_STEP
