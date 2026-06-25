/* AVX2 NVFP4 kernel — GGUF block_nvfp4 ({d[4] UE4M3; qs[32]}, 64 elems = 4 sub-blocks of 16, no global
 * scale). Decode = MXFP4's pshufb LUT; the sub-block nibble order (byte s*8+j: low=elem j, high=elem j+8)
 * means a 32-element span (2 sub-blocks, 16 bytes) decodes interleaved, so an unpacklo/hi_epi64 reorders it
 * to two contiguous 16-element halves. Per-16 UE4M3 scale via the Q6_K trick ([s0,s1] on the 8-lane dot).
 * Activations are int8-requantized (J->aq/ad). Built -mavx2 -mfma -mf16c. */
#include <immintrin.h>
#include "jam_internal.h"
#include "jam_nvfp4.h"
#include "jam_decode_x86_256.h"   /* jam_hsum8_256; JAM_MXFP4_CODES via jam_mxfp4.h */

#define KDOT(u, s) _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)))

void jam_mm_nvfp4_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    const int8_t* AQ = J->aq; const float* AD = J->ad;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k;
    const int nblk = k / JAM_NVFP4_QK;                       /* 64-element blocks */
    const size_t wrow = (size_t) nblk * sizeof(jam_nvfp4_blk);
    const __m128i lut = _mm_setr_epi8(JAM_MXFP4_CODES);
    const __m128i m4  = _mm_set1_epi8(0x0F);
    for (int i = rb; i < re; ++i) {
        const jam_nvfp4_blk* wr = (const jam_nvfp4_blk*) (W + (size_t) i * wrow);
        for (int j = 0; j < n; ++j) {
            const int8_t* aq = AQ + (size_t) j * k;
            const float* ad = AD + (size_t) j * (k / 32);     /* per-32 activation scales */
            float acc = 0.0f;
            for (int bb = 0; bb < nblk; ++bb) {
                const jam_nvfp4_blk* w = &wr[bb];
                for (int sp = 0; sp < 2; ++sp) {               /* 2 spans of 32 = sub-blocks (2sp, 2sp+1) */
                    __m128i qs = _mm_loadu_si128((const __m128i*) (w->qs + sp * 16));
                    __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(qs, m4));            /* [e0-7, e16-23] */
                    __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(qs, 4), m4)); /* [e8-15,e24-31] */
                    __m128i wl = _mm_unpacklo_epi64(lo, hi);   /* elems 0-15  (sub-block 2sp)   */
                    __m128i wh = _mm_unpackhi_epi64(lo, hi);   /* elems 16-31 (sub-block 2sp+1) */
                    __m256i wq = _mm256_set_m128i(wh, wl);
                    int blk32 = bb * 2 + sp;
                    __m256i av = _mm256_loadu_si256((const __m256i*) (aq + (size_t) blk32 * 32));
                    __m256 prod = KDOT(_mm256_abs_epi8(wq), _mm256_sign_epi8(av, wq));
                    float s0 = jam_ue4m3_to_float(w->d[2*sp]), s1 = jam_ue4m3_to_float(w->d[2*sp + 1]);
                    __m256 scv = _mm256_setr_ps(s0,s0,s0,s0, s1,s1,s1,s1);
                    acc += ad[blk32] * jam_hsum8_256(_mm256_mul_ps(prod, scv));
                }
            }
            C[(size_t) j*ldc + i] = acc;
        }
    }
}
