/* AVX-512BW (no VNNI): 512-bit maddubs int8 dot for Q8_0. Bound at the AVX512 level on AVX-512 CPUs
 * that LACK VNNI (Skylake-SP/X) — ~2× the 256-bit AVX2 maddubs path. Built without -mavx512vnni so it
 * provably contains no VNNI instructions.
 *
 * Two wrinkles vs the 256-bit kernels:
 *   1. A 512-bit register holds 64 int8 = TWO Q8_0 blocks, so we process a block-PAIR per iteration.
 *      maddubs+madd yields 16 int32: lanes 0..7 are the first block's partial sums, 8..15 the second,
 *      so each block-pair is scaled by [dA0·dB0 ×8, dA1·dB1 ×8] and accumulated; reduce_add at the end.
 *   2. AVX-512 has no vpsignb. The abs/sign identity (dpbusd(|qa|, sign(qb,qa)) = Σ qa·qb) is kept by
 *      negating qb where qa<0 via movepi8_mask + mask_sub; qa==0 is nulled by |qa|=0. */
#include "jam_internal.h"
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

typedef struct __attribute__((packed)) { uint16_t d; int8_t qs[32]; } block_q8_0;

/* dot of a block-PAIR (64 elems) -> 16 float lanes (lower 8 = first block, upper 8 = second). */
static inline __m512 dotpair(__m512i aqa, __m512i sqb) {
    __m512i p16 = _mm512_maddubs_epi16(aqa, sqb);                 /* u8·s8, adjacent pairs summed (s16) */
    __m512i d32 = _mm512_madd_epi16(p16, _mm512_set1_epi16(1));   /* widen + sum pairs -> 16 s32 */
    return _mm512_cvtepi32_ps(d32);
}

/* sign(qb,qa): negate qb where qa<0 (qa==0 is handled by |qa|=0 nulling the product). No vpsignb @512. */
static inline __m512i sign_fold(__m512i qb, __m512i qa) {
    return _mm512_mask_sub_epi8(qb, _mm512_movepi8_mask(qa), _mm512_setzero_si512(), qb);
}

/* scale vector: lower 8 lanes = s0 (first block), upper 8 = s1 (second block). */
static inline __m512 scale2(float s0, float s1) {
    return _mm512_insertf32x8(_mm512_set1_ps(s0), _mm256_set1_ps(s1), 1);
}

/* the two weight blocks' int8 (interleaved by their fp16 scales in memory) -> one 512-bit register. */
static inline __m512i wpair(const block_q8_0* w0, const block_q8_0* w1) {
    __m256i lo = _mm256_loadu_si256((const __m256i*) w0->qs);
    __m256i hi = _mm256_loadu_si256((const __m256i*) w1->qs);
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

void jam_mm_q8_0_avx512bw(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* A = (const char*) J->a;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, nb = J->nb, k = J->k;
    const size_t wrow = (size_t)(J->lda / 32);
    for (int i = rb; i < re; ++i) {
        const block_q8_0* arow = (const block_q8_0*) (A + (size_t) i * wrow * sizeof(block_q8_0));
        int j = 0;
        for (; j + 4 <= n; j += 4) {                          /* tile 4 activation cols, reuse the weight pair */
            const int8_t *q0=J->aq+(size_t)(j+0)*k,*q1=J->aq+(size_t)(j+1)*k,*q2=J->aq+(size_t)(j+2)*k,*q3=J->aq+(size_t)(j+3)*k;
            const float  *d0=J->ad+(size_t)(j+0)*nb,*d1=J->ad+(size_t)(j+1)*nb,*d2=J->ad+(size_t)(j+2)*nb,*d3=J->ad+(size_t)(j+3)*nb;
            __m512 f0=_mm512_setzero_ps(),f1=_mm512_setzero_ps(),f2=_mm512_setzero_ps(),f3=_mm512_setzero_ps();
            int blk = 0;
            for (; blk + 2 <= nb; blk += 2) {                 /* a block-PAIR per iteration */
                __m512i qa  = wpair(&arow[blk], &arow[blk+1]);
                __m512i aqa = _mm512_abs_epi8(qa);
                float a0 = _cvtsh_ss(arow[blk].d), a1 = _cvtsh_ss(arow[blk+1].d);
                f0=_mm512_fmadd_ps(scale2(a0*d0[blk],a1*d0[blk+1]), dotpair(aqa, sign_fold(_mm512_loadu_si512((const void*)(q0+(size_t)blk*32)), qa)), f0);
                f1=_mm512_fmadd_ps(scale2(a0*d1[blk],a1*d1[blk+1]), dotpair(aqa, sign_fold(_mm512_loadu_si512((const void*)(q1+(size_t)blk*32)), qa)), f1);
                f2=_mm512_fmadd_ps(scale2(a0*d2[blk],a1*d2[blk+1]), dotpair(aqa, sign_fold(_mm512_loadu_si512((const void*)(q2+(size_t)blk*32)), qa)), f2);
                f3=_mm512_fmadd_ps(scale2(a0*d3[blk],a1*d3[blk+1]), dotpair(aqa, sign_fold(_mm512_loadu_si512((const void*)(q3+(size_t)blk*32)), qa)), f3);
            }
            if (blk < nb) {                                   /* odd leftover block: zero-extend into the lower half */
                __m512i qa  = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*) arow[blk].qs));
                __m512i aqa = _mm512_abs_epi8(qa);
                float a0 = _cvtsh_ss(arow[blk].d);
                f0=_mm512_fmadd_ps(scale2(a0*d0[blk],0.f), dotpair(aqa, sign_fold(_mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)(q0+(size_t)blk*32))), qa)), f0);
                f1=_mm512_fmadd_ps(scale2(a0*d1[blk],0.f), dotpair(aqa, sign_fold(_mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)(q1+(size_t)blk*32))), qa)), f1);
                f2=_mm512_fmadd_ps(scale2(a0*d2[blk],0.f), dotpair(aqa, sign_fold(_mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)(q2+(size_t)blk*32))), qa)), f2);
                f3=_mm512_fmadd_ps(scale2(a0*d3[blk],0.f), dotpair(aqa, sign_fold(_mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)(q3+(size_t)blk*32))), qa)), f3);
            }
            C[(size_t)(j+0)*ldc+i]=_mm512_reduce_add_ps(f0); C[(size_t)(j+1)*ldc+i]=_mm512_reduce_add_ps(f1);
            C[(size_t)(j+2)*ldc+i]=_mm512_reduce_add_ps(f2); C[(size_t)(j+3)*ldc+i]=_mm512_reduce_add_ps(f3);
        }
        for (; j < n; ++j) {                                  /* column tail */
            const int8_t* qj = J->aq+(size_t)j*k; const float* dj = J->ad+(size_t)j*nb;
            __m512 f = _mm512_setzero_ps();
            int blk = 0;
            for (; blk + 2 <= nb; blk += 2) {
                __m512i qa = wpair(&arow[blk], &arow[blk+1]);
                __m512i aqa = _mm512_abs_epi8(qa);
                f=_mm512_fmadd_ps(scale2(_cvtsh_ss(arow[blk].d)*dj[blk], _cvtsh_ss(arow[blk+1].d)*dj[blk+1]),
                        dotpair(aqa, sign_fold(_mm512_loadu_si512((const void*)(qj+(size_t)blk*32)), qa)), f);
            }
            if (blk < nb) {
                __m512i qa = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*) arow[blk].qs));
                __m512i aqa = _mm512_abs_epi8(qa);
                f=_mm512_fmadd_ps(scale2(_cvtsh_ss(arow[blk].d)*dj[blk], 0.f),
                        dotpair(aqa, sign_fold(_mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)(qj+(size_t)blk*32))), qa)), f);
            }
            C[(size_t)j*ldc+i] = _mm512_reduce_add_ps(f);
        }
    }
}
