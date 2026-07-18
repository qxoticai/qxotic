/* AVX2 Q1_0 kernel - GGML block_q1_0 ({fp16 d; uint8 qs[16]}, 128 elems, 1 sign bit each, LSB-first;
 * elem = bit ? +d : -d). Faithful port of llama.cpp's ggml_vec_dot_q1_0_q8_0 AVX2 body: per 32-element
 * sub-block, broadcast the 4 sign bytes across lanes (pshufb byte_shuf), AND with per-lane bit masks,
 * compare-eq-zero -> sm (0xFF where the bit is CLEAR = negative); conditionally negate the int8
 * activations sy = (qy ^ sm) - sm; horizontal-sum via maddubs(1, sy) + madd(1); scale by the per-32
 * activation scale, then the whole 128-block by d. Activations are int8-requantized (J->aq/ad).
 *
 * Structure mirrors the NVFP4 kernel: 4 activation columns share one sign-mask expansion (amortized 4x).
 * Built -mavx2 -mfma. */
#include <immintrin.h>
#include "jam_internal.h"
#include "jam_fp16.h"
#include "jam_q1_0.h"             /* jam_blk_q1_0 */
#include "jam_decode_x86_256.h"   /* jam_hsum8_256 */

#define Q1_HSUM(sy) _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), (sy)), _mm256_set1_epi16(1)))

/* Expand sub-block K's 4 sign bytes to a 32-lane negate mask (0xFF where bit clear). */
#define Q1_SM(qs32, K, shuf, masks, zero) \
    _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) (qs32)[K]), (shuf)), (masks)), (zero))

#define Q1_SIGNED(qy, sm) _mm256_sub_epi8(_mm256_xor_si256((qy), (sm)), (sm))

void jam_mm_q1_0_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k;
    const int nblk = k / 128;
    const int nb32 = k / 32;
    const size_t wrow = (size_t)(J->lda / 128) * sizeof(jam_blk_q1_0);   /* row stride honors lda (ldw) */
    const __m256i byte_shuf = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit_masks = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128,
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128);
    const __m256i zero = _mm256_setzero_si256();

    for (int i = rb; i < re; ++i) {
        const char* wr = W + (size_t) i * wrow;
        int j = 0;
        for (; j + 4 <= n; j += 4) {                          /* tile 4 columns: expand signs once, use 4x */
            const int8_t *a0=AQ+(size_t)(j+0)*k, *a1=AQ+(size_t)(j+1)*k, *a2=AQ+(size_t)(j+2)*k, *a3=AQ+(size_t)(j+3)*k;
            const float  *d0=AD+(size_t)(j+0)*nb32, *d1=AD+(size_t)(j+1)*nb32, *d2=AD+(size_t)(j+2)*nb32, *d3=AD+(size_t)(j+3)*nb32;
            __m256 f0=_mm256_setzero_ps(), f1=_mm256_setzero_ps(), f2=_mm256_setzero_ps(), f3=_mm256_setzero_ps();
            for (int bb = 0; bb < nblk; ++bb) {
                const jam_blk_q1_0* w = (const jam_blk_q1_0*) (wr + (size_t) bb * sizeof(jam_blk_q1_0));
                const uint32_t* qs32 = (const uint32_t*) w->qs;
                const __m256 vd = _mm256_set1_ps(jam_half2float(w->d));
                __m256 b0=_mm256_setzero_ps(), b1=_mm256_setzero_ps(), b2=_mm256_setzero_ps(), b3=_mm256_setzero_ps();
                for (int K = 0; K < 4; ++K) {
                    const __m256i sm = Q1_SM(qs32, K, byte_shuf, bit_masks, zero);
                    const int blk32 = bb * 4 + K; const size_t off = (size_t) blk32 * 32;
                    b0 = _mm256_fmadd_ps(_mm256_set1_ps(d0[blk32]), Q1_HSUM(Q1_SIGNED(_mm256_loadu_si256((const __m256i*)(a0+off)), sm)), b0);
                    b1 = _mm256_fmadd_ps(_mm256_set1_ps(d1[blk32]), Q1_HSUM(Q1_SIGNED(_mm256_loadu_si256((const __m256i*)(a1+off)), sm)), b1);
                    b2 = _mm256_fmadd_ps(_mm256_set1_ps(d2[blk32]), Q1_HSUM(Q1_SIGNED(_mm256_loadu_si256((const __m256i*)(a2+off)), sm)), b2);
                    b3 = _mm256_fmadd_ps(_mm256_set1_ps(d3[blk32]), Q1_HSUM(Q1_SIGNED(_mm256_loadu_si256((const __m256i*)(a3+off)), sm)), b3);
                }
                f0 = _mm256_fmadd_ps(vd, b0, f0); f1 = _mm256_fmadd_ps(vd, b1, f1);
                f2 = _mm256_fmadd_ps(vd, b2, f2); f3 = _mm256_fmadd_ps(vd, b3, f3);
            }
            C[(size_t)(j+0)*ldc+i]=jam_hsum8_256(f0); C[(size_t)(j+1)*ldc+i]=jam_hsum8_256(f1);
            C[(size_t)(j+2)*ldc+i]=jam_hsum8_256(f2); C[(size_t)(j+3)*ldc+i]=jam_hsum8_256(f3);
        }
        for (; j < n; ++j) {                                  /* column tail */
            const int8_t* aj = AQ + (size_t) j * k;
            const float* dj = AD + (size_t) j * nb32;
            __m256 f = _mm256_setzero_ps();
            for (int bb = 0; bb < nblk; ++bb) {
                const jam_blk_q1_0* w = (const jam_blk_q1_0*) (wr + (size_t) bb * sizeof(jam_blk_q1_0));
                const uint32_t* qs32 = (const uint32_t*) w->qs;
                const __m256 vd = _mm256_set1_ps(jam_half2float(w->d));
                __m256 fb = _mm256_setzero_ps();
                for (int K = 0; K < 4; ++K) {
                    const __m256i sm = Q1_SM(qs32, K, byte_shuf, bit_masks, zero);
                    const int blk32 = bb * 4 + K;
                    const __m256i qy = _mm256_loadu_si256((const __m256i*)(aj + (size_t) blk32 * 32));
                    fb = _mm256_fmadd_ps(_mm256_set1_ps(dj[blk32]), Q1_HSUM(Q1_SIGNED(qy, sm)), fb);
                }
                f = _mm256_fmadd_ps(vd, fb, f);
            }
            C[(size_t) j*ldc + i] = jam_hsum8_256(f);
        }
    }
}
