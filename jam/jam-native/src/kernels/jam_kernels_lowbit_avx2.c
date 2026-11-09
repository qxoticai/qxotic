/* AVX2 kernel for the low-bit codes formats (jam_lowbit.h: Q2_0, PQ2_0, PTQ1_0; J->wt says which).
 * Each weight row decodes once to s8 codes (-1..2) and a scale per 32; every 32-element span is then
 * llama.cpp's signed int8 dot maddubs(|w|, sign(a, w)) + madd(1), scaled by the per-32 activation
 * scale and the weight scale. Activations are int8-requantized (J->aq/ad). 4 columns share one load of
 * the weight codes. Built -mavx2 -mfma. */
#include <immintrin.h>
#include "jam_internal.h"
#include "jam_lowbit.h"
#include "jam_decode_x86_256.h"   /* jam_hsum8_256 */

/* sum(w * a) over 32 lanes as 8 float partial sums */
#define LB_DOT(aw, w, a) _mm256_cvtepi32_ps(_mm256_madd_epi16( \
    _mm256_maddubs_epi16((aw), _mm256_sign_epi8((a), (w))), _mm256_set1_epi16(1)))

void jam_mm_lowbit_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb32 = k / 32, qk = jam_lowbit_qk(J->wt);
    const size_t wrow = (size_t)(J->lda / qk) * jam_lowbit_bytes(J->wt);   /* row stride honors lda (ldw) */
    int8_t* codes = jam_lowbit_scratch((size_t) k + sizeof(float) * (size_t) nb32);
    float* d32 = (float*) (codes + k);

    for (int i = rb; i < re; ++i) {
        jam_lowbit_row(J->wt, (const uint8_t*) J->a + (size_t) i * wrow, k / qk, codes, d32);
        int j = 0;
        for (; j + 4 <= n; j += 4) {                          /* 4 columns per load of the codes */
            const int8_t *a0=AQ+(size_t)(j+0)*k, *a1=AQ+(size_t)(j+1)*k, *a2=AQ+(size_t)(j+2)*k, *a3=AQ+(size_t)(j+3)*k;
            const float  *e0=AD+(size_t)(j+0)*nb32, *e1=AD+(size_t)(j+1)*nb32, *e2=AD+(size_t)(j+2)*nb32, *e3=AD+(size_t)(j+3)*nb32;
            __m256 f0=_mm256_setzero_ps(), f1=_mm256_setzero_ps(), f2=_mm256_setzero_ps(), f3=_mm256_setzero_ps();
            for (int b = 0; b < nb32; ++b) {
                const size_t off = (size_t) b * 32;
                const __m256i wc = _mm256_loadu_si256((const __m256i*) (codes + off)), aw = _mm256_abs_epi8(wc);
                f0 = _mm256_fmadd_ps(_mm256_set1_ps(d32[b] * e0[b]), LB_DOT(aw, wc, _mm256_loadu_si256((const __m256i*)(a0+off))), f0);
                f1 = _mm256_fmadd_ps(_mm256_set1_ps(d32[b] * e1[b]), LB_DOT(aw, wc, _mm256_loadu_si256((const __m256i*)(a1+off))), f1);
                f2 = _mm256_fmadd_ps(_mm256_set1_ps(d32[b] * e2[b]), LB_DOT(aw, wc, _mm256_loadu_si256((const __m256i*)(a2+off))), f2);
                f3 = _mm256_fmadd_ps(_mm256_set1_ps(d32[b] * e3[b]), LB_DOT(aw, wc, _mm256_loadu_si256((const __m256i*)(a3+off))), f3);
            }
            C[(size_t)(j+0)*ldc+i]=jam_hsum8_256(f0); C[(size_t)(j+1)*ldc+i]=jam_hsum8_256(f1);
            C[(size_t)(j+2)*ldc+i]=jam_hsum8_256(f2); C[(size_t)(j+3)*ldc+i]=jam_hsum8_256(f3);
        }
        for (; j < n; ++j) {                                  /* column tail */
            const int8_t* aj = AQ + (size_t) j * k;
            const float* ej = AD + (size_t) j * nb32;
            __m256 f = _mm256_setzero_ps();
            for (int b = 0; b < nb32; ++b) {
                const size_t off = (size_t) b * 32;
                const __m256i wc = _mm256_loadu_si256((const __m256i*) (codes + off));
                f = _mm256_fmadd_ps(_mm256_set1_ps(d32[b] * ej[b]),
                                    LB_DOT(_mm256_abs_epi8(wc), wc, _mm256_loadu_si256((const __m256i*)(aj+off))), f);
            }
            C[(size_t) j*ldc + i] = jam_hsum8_256(f);
        }
    }
}
