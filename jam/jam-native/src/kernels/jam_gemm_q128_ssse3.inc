/* 128-bit SSSE3 quantized GEMM engine: C = W @ Aᵀ, W decoded to int8 (two __m128i halves), A int8. Same
 * shape as jam_gemm_q128.inc (the true-SSE3 floor) but the int8 dot uses the SSSE3 maddubs sign-trick —
 * maddubs(|a|, sign(w,a)) -> int16 pairs, madd_epi16(.,1) -> int32 — instead of sign-extend + madd, halving
 * the dot's op count. Weights are assumed in [-127,127] (real Q8_0/Q4_0 quantization; the sign-trick's only
 * pitfall is w=-128, which symmetric quant never produces). The includer defines, before #include:
 *
 *   JAM_BLK                            weight block struct (for the row stride)           [per QUANT]
 *   JAM_DECODE(blk, &wlo, &whi, &dW)   decode -> two __m128i int8 halves + float scale    [per QUANT]
 *   JAM_MM_NAME                        exported symbol
 *
 * Requires jam_decode_x86_128.h (jam_hsum4_sse3) and "jam_internal.h" (jam_q8_job) already included; the
 * file-scope helpers JAM_SSSE3_DOT16 / JAM_SSSE3_DOTBLK below (defined once, guarded). */

#ifndef JAM_SSSE3_DOT_DEFINED
#define JAM_SSSE3_DOT_DEFINED
/* 16-int8 dot via maddubs sign-trick (|a| u8 operand, sign(w,a) carries the product sign) -> 4 int32. */
#define JAM_SSSE3_DOT16(w, a, ones) _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(a), _mm_sign_epi8((w), (a))), (ones))
/* full 32-int8 block dot (two halves) for activation pointer p -> 4 int32. */
#define JAM_SSSE3_DOTBLK(wlo, whi, p, ones) _mm_add_epi32( \
        JAM_SSSE3_DOT16((wlo), _mm_loadu_si128((const __m128i*) (p)), (ones)), \
        JAM_SSSE3_DOT16((whi), _mm_loadu_si128((const __m128i*) ((p) + 16)), (ones)))
#endif

void JAM_MM_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, nb = J->nb, k = J->k;
    const size_t wrow = (size_t)(J->lda / 32);
    const __m128i ones = _mm_set1_epi16(1);
    for (int i = rb; i < re; ++i) {
        const JAM_BLK* wr = (const JAM_BLK*) (W + (size_t) i * wrow * sizeof(JAM_BLK));
        int j = 0;
        for (; j + 4 <= n; j += 4) {          /* tile 4 activation cols, reuse the decoded weight block */
            const int8_t *a0=J->aq+(size_t)(j+0)*k,*a1=J->aq+(size_t)(j+1)*k,*a2=J->aq+(size_t)(j+2)*k,*a3=J->aq+(size_t)(j+3)*k;
            const float  *s0=J->ad+(size_t)(j+0)*nb,*s1=J->ad+(size_t)(j+1)*nb,*s2=J->ad+(size_t)(j+2)*nb,*s3=J->ad+(size_t)(j+3)*nb;
            __m128 f0=_mm_setzero_ps(),f1=_mm_setzero_ps(),f2=_mm_setzero_ps(),f3=_mm_setzero_ps();
            for (int b = 0; b < nb; ++b) {
                __m128i wlo, whi; float dW; JAM_DECODE(&wr[b], &wlo, &whi, &dW);
                f0=_mm_add_ps(f0,_mm_mul_ps(_mm_set1_ps(dW*s0[b]),_mm_cvtepi32_ps(JAM_SSSE3_DOTBLK(wlo,whi,a0+(size_t)b*32,ones))));
                f1=_mm_add_ps(f1,_mm_mul_ps(_mm_set1_ps(dW*s1[b]),_mm_cvtepi32_ps(JAM_SSSE3_DOTBLK(wlo,whi,a1+(size_t)b*32,ones))));
                f2=_mm_add_ps(f2,_mm_mul_ps(_mm_set1_ps(dW*s2[b]),_mm_cvtepi32_ps(JAM_SSSE3_DOTBLK(wlo,whi,a2+(size_t)b*32,ones))));
                f3=_mm_add_ps(f3,_mm_mul_ps(_mm_set1_ps(dW*s3[b]),_mm_cvtepi32_ps(JAM_SSSE3_DOTBLK(wlo,whi,a3+(size_t)b*32,ones))));
            }
            C[(size_t)(j+0)*ldc+i]=jam_hsum4_sse3(f0); C[(size_t)(j+1)*ldc+i]=jam_hsum4_sse3(f1);
            C[(size_t)(j+2)*ldc+i]=jam_hsum4_sse3(f2); C[(size_t)(j+3)*ldc+i]=jam_hsum4_sse3(f3);
        }
        for (; j < n; ++j) {                  /* column tail */
            const int8_t* aj=J->aq+(size_t)j*k; const float* sj=J->ad+(size_t)j*nb;
            __m128 f = _mm_setzero_ps();
            for (int b = 0; b < nb; ++b) {
                __m128i wlo, whi; float dW; JAM_DECODE(&wr[b], &wlo, &whi, &dW);
                f=_mm_add_ps(f,_mm_mul_ps(_mm_set1_ps(dW*sj[b]),_mm_cvtepi32_ps(JAM_SSSE3_DOTBLK(wlo,whi,aj+(size_t)b*32,ones))));
            }
            C[(size_t)j*ldc+i] = jam_hsum4_sse3(f);
        }
    }
}
