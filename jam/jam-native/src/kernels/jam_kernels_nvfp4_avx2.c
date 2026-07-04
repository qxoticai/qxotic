/* AVX2 NVFP4 kernel — GGUF block_nvfp4 ({d[4] UE4M3; qs[32]}, 64 elems = 4 sub-blocks of 16, no global
 * scale). Decode = MXFP4's pshufb LUT; the sub-block nibble order (byte s*8+j: low=elem j, high=elem j+8)
 * means a 32-element span (2 sub-blocks, 16 bytes) decodes interleaved, so an unpacklo/hi_epi64 reorders it
 * to two contiguous 16-element halves. Per-16 UE4M3 scale via a 256-entry LUT (ldexpf is a libm call — it
 * profiled as scalbnf in the hot loop). Activations are int8-requantized (J->aq/ad).
 *
 * Structure mirrors the q256 engine: 4 activation columns share one decoded weight span (decode and |w|
 * amortized 4x), and the per-16 scales are applied to a VECTOR accumulator (one fmadd per span-column) so
 * the horizontal sum happens once per (row, column) instead of once per 32 elements. Built -mavx2 -mfma. */
#include <immintrin.h>
#include "jam_internal.h"
#include "jam_nvfp4.h"
#include "jam_decode_x86_256.h"   /* jam_hsum8_256; JAM_MXFP4_CODES via jam_mxfp4.h */

#define KDOT(u, s) _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)))

static float jam_ue4m3_lut[256];
__attribute__((constructor)) static void jam_ue4m3_lut_init(void) {
    for (int i = 0; i < 256; ++i) jam_ue4m3_lut[i] = jam_ue4m3_to_float((uint8_t) i);
}

/* Decode one 32-element span (sub-blocks 2sp, 2sp+1) of block w to int8 codes + its per-16 scale vector. */
#define NVFP4_SPAN(w, sp, lut, m4, wq, scv) do {                                                  \
        __m128i qs = _mm_loadu_si128((const __m128i*) ((w)->qs + (sp) * 16));                     \
        __m128i lo = _mm_shuffle_epi8((lut), _mm_and_si128(qs, (m4)));                            \
        __m128i hi = _mm_shuffle_epi8((lut), _mm_and_si128(_mm_srli_epi16(qs, 4), (m4)));         \
        (wq) = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));          \
        float s0 = jam_ue4m3_lut[(w)->d[2*(sp)]], s1 = jam_ue4m3_lut[(w)->d[2*(sp) + 1]];         \
        (scv) = _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1);                                   \
    } while (0)

void jam_mm_nvfp4_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k;
    const int nblk = k / JAM_NVFP4_QK;                       /* 64-element blocks */
    const int nb32 = k / 32;                                 /* per-32 activation scale stride */
    const size_t wrow = (size_t)(J->lda / JAM_NVFP4_QK) * sizeof(jam_nvfp4_blk);   /* row stride honors lda (ldw) */
    const __m128i lut = _mm_setr_epi8(JAM_MXFP4_CODES);
    const __m128i m4  = _mm_set1_epi8(0x0F);
    for (int i = rb; i < re; ++i) {
        const jam_nvfp4_blk* wr = (const jam_nvfp4_blk*) (W + (size_t) i * wrow);
        int j = 0;
        for (; j + 4 <= n; j += 4) {                          /* tile 4 columns: decode once, use 4x */
            const int8_t *a0=AQ+(size_t)(j+0)*k, *a1=AQ+(size_t)(j+1)*k, *a2=AQ+(size_t)(j+2)*k, *a3=AQ+(size_t)(j+3)*k;
            const float  *d0=AD+(size_t)(j+0)*nb32, *d1=AD+(size_t)(j+1)*nb32, *d2=AD+(size_t)(j+2)*nb32, *d3=AD+(size_t)(j+3)*nb32;
            __m256 f0=_mm256_setzero_ps(), f1=_mm256_setzero_ps(), f2=_mm256_setzero_ps(), f3=_mm256_setzero_ps();
            for (int bb = 0; bb < nblk; ++bb) {
                const jam_nvfp4_blk* w = &wr[bb];
                for (int sp = 0; sp < 2; ++sp) {
                    __m256i wq; __m256 scv;
                    NVFP4_SPAN(w, sp, lut, m4, wq, scv);
                    __m256i uq = _mm256_abs_epi8(wq);
                    const int blk32 = bb * 2 + sp; const size_t off = (size_t) blk32 * 32;
                    __m256i v0 = _mm256_loadu_si256((const __m256i*)(a0+off));
                    __m256i v1 = _mm256_loadu_si256((const __m256i*)(a1+off));
                    __m256i v2 = _mm256_loadu_si256((const __m256i*)(a2+off));
                    __m256i v3 = _mm256_loadu_si256((const __m256i*)(a3+off));
                    f0 = _mm256_fmadd_ps(_mm256_mul_ps(KDOT(uq, _mm256_sign_epi8(v0, wq)), scv), _mm256_set1_ps(d0[blk32]), f0);
                    f1 = _mm256_fmadd_ps(_mm256_mul_ps(KDOT(uq, _mm256_sign_epi8(v1, wq)), scv), _mm256_set1_ps(d1[blk32]), f1);
                    f2 = _mm256_fmadd_ps(_mm256_mul_ps(KDOT(uq, _mm256_sign_epi8(v2, wq)), scv), _mm256_set1_ps(d2[blk32]), f2);
                    f3 = _mm256_fmadd_ps(_mm256_mul_ps(KDOT(uq, _mm256_sign_epi8(v3, wq)), scv), _mm256_set1_ps(d3[blk32]), f3);
                }
            }
            C[(size_t)(j+0)*ldc+i]=jam_hsum8_256(f0); C[(size_t)(j+1)*ldc+i]=jam_hsum8_256(f1);
            C[(size_t)(j+2)*ldc+i]=jam_hsum8_256(f2); C[(size_t)(j+3)*ldc+i]=jam_hsum8_256(f3);
        }
        for (; j < n; ++j) {                                  /* column tail */
            const int8_t* aj = AQ + (size_t) j * k;
            const float* dj = AD + (size_t) j * nb32;
            __m256 f = _mm256_setzero_ps();
            for (int bb = 0; bb < nblk; ++bb) {
                const jam_nvfp4_blk* w = &wr[bb];
                for (int sp = 0; sp < 2; ++sp) {
                    __m256i wq; __m256 scv;
                    NVFP4_SPAN(w, sp, lut, m4, wq, scv);
                    const int blk32 = bb * 2 + sp;
                    __m256i av = _mm256_loadu_si256((const __m256i*)(aj + (size_t) blk32 * 32));
                    __m256 prod = KDOT(_mm256_abs_epi8(wq), _mm256_sign_epi8(av, wq));
                    f = _mm256_fmadd_ps(_mm256_mul_ps(prod, scv), _mm256_set1_ps(dj[blk32]), f);
                }
            }
            C[(size_t) j*ldc + i] = jam_hsum8_256(f);
        }
    }
}
