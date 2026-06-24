/* AVX2 NVFP4 kernel (E2M1 nibbles + a per-16 E4M3 scale, PLANAR layout). Decode = MXFP4's pshufb LUT into
 * two contiguous 16-element halves (low/high nibbles); the per-half E4M3 scale is the Q6_K trick — scale the
 * two halves of the 8-lane int8 dot by [s0,s1] before the hsum. Activations are int8-requantized (J->aq/ad);
 * the per-tensor FP32 global scale G is the caller's (a post-scale). Built -mavx2 -mfma -mf16c. */
#include <immintrin.h>
#include "jam_internal.h"
#include "jam_nvfp4.h"
#include "jam_decode_x86_256.h"   /* jam_hsum8_256; JAM_MXFP4_CODES via jam_mxfp4.h */

#define KDOT(u, s) _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)))

void jam_mm_nvfp4_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const uint8_t* DATA  = (const uint8_t*) J->a;
    const uint8_t* SCALE = (const uint8_t*) J->wscale;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;   /* nb = k/32 */
    const size_t drow = (size_t) nb * 16, srow = (size_t) nb * 2;
    const __m128i lut = _mm_setr_epi8(JAM_MXFP4_CODES);
    const __m128i m4  = _mm_set1_epi8(0x0F);
    for (int i = rb; i < re; ++i) {
        const uint8_t* dr = DATA + (size_t) i * drow;
        const uint8_t* sr = SCALE + (size_t) i * srow;
        for (int j = 0; j < n; ++j) {
            const int8_t* aq = AQ + (size_t) j * k;
            const float* ad = AD + (size_t) j * nb;
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                __m128i qs = _mm_loadu_si128((const __m128i*) (dr + (size_t) b * 16));
                __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(qs, m4));
                __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(qs, 4), m4));
                __m256i wq = _mm256_set_m128i(hi, lo);     /* 32 int8: lo = elems 0-15, hi = 16-31 */
                __m256i av = _mm256_loadu_si256((const __m256i*) (aq + (size_t) b * 32));
                __m256 prod = KDOT(_mm256_abs_epi8(wq), _mm256_sign_epi8(av, wq));   /* signed dot */
                float s0 = jam_nvfp4_dhalf(sr[2*b]), s1 = jam_nvfp4_dhalf(sr[2*b + 1]);
                __m256 scv = _mm256_setr_ps(s0,s0,s0,s0, s1,s1,s1,s1);   /* lanes 0-3 -> s0, 4-7 -> s1 */
                acc += ad[b] * jam_hsum8_256(_mm256_mul_ps(prod, scv));
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}
