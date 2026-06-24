/* 256-bit quantized GEMM engine:  C = W @ Aᵀ , W quantized (decoded to int8), A int8-requantized.
 * The tiling (4 activation columns), abs/sign trick, and deferred-float accumulation live HERE, once.
 * The includer composes one kernel from a (quant) decoder and an (ISA) dot by defining, before #include:
 *
 *   JAM_BLK                       weight block struct (for the row stride)              [per QUANT]
 *   JAM_DECODE(blk, &wq, &dW)     decode block -> __m256i int8 weights + float scale    [per QUANT]
 *   JAM_DOT(aqa, sqb) -> __m256   int8 dot of |w| and sign(a,w), as float               [per ISA]
 *   JAM_MM_NAME                   exported symbol
 *
 * Requires jam_decode_x86_256.h (jam_hsum8_256) and "jam_internal.h" (jam_q8_job) already included. */

void JAM_MM_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;       /* weights  (J->a is the weight operand) */
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, nb = J->nb, k = J->k;
    const size_t wrow = (size_t)(J->lda / 32);
    for (int i = rb; i < re; ++i) {
        const JAM_BLK* wr = (const JAM_BLK*) (W + (size_t) i * wrow * sizeof(JAM_BLK));
        int j = 0;
        for (; j + 4 <= n; j += 4) {          /* tile 4 activation cols, reuse the decoded weight block */
            const int8_t *a0=J->aq+(size_t)(j+0)*k,*a1=J->aq+(size_t)(j+1)*k,*a2=J->aq+(size_t)(j+2)*k,*a3=J->aq+(size_t)(j+3)*k;
            const float  *s0=J->ad+(size_t)(j+0)*nb,*s1=J->ad+(size_t)(j+1)*nb,*s2=J->ad+(size_t)(j+2)*nb,*s3=J->ad+(size_t)(j+3)*nb;
            __m256 f0=_mm256_setzero_ps(),f1=_mm256_setzero_ps(),f2=_mm256_setzero_ps(),f3=_mm256_setzero_ps();
            for (int b = 0; b < nb; ++b) {
                __m256i wq; float dW; JAM_DECODE(&wr[b], &wq, &dW);
                __m256i aqa = _mm256_abs_epi8(wq);
                f0=_mm256_fmadd_ps(_mm256_set1_ps(dW*s0[b]), JAM_DOT(aqa,_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(a0+(size_t)b*32)),wq)), f0);
                f1=_mm256_fmadd_ps(_mm256_set1_ps(dW*s1[b]), JAM_DOT(aqa,_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(a1+(size_t)b*32)),wq)), f1);
                f2=_mm256_fmadd_ps(_mm256_set1_ps(dW*s2[b]), JAM_DOT(aqa,_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(a2+(size_t)b*32)),wq)), f2);
                f3=_mm256_fmadd_ps(_mm256_set1_ps(dW*s3[b]), JAM_DOT(aqa,_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(a3+(size_t)b*32)),wq)), f3);
            }
            C[(size_t)(j+0)*ldc+i]=jam_hsum8_256(f0); C[(size_t)(j+1)*ldc+i]=jam_hsum8_256(f1);
            C[(size_t)(j+2)*ldc+i]=jam_hsum8_256(f2); C[(size_t)(j+3)*ldc+i]=jam_hsum8_256(f3);
        }
        for (; j < n; ++j) {                  /* column tail */
            const int8_t* aj=J->aq+(size_t)j*k; const float* sj=J->ad+(size_t)j*nb;
            __m256 f = _mm256_setzero_ps();
            for (int b = 0; b < nb; ++b) {
                __m256i wq; float dW; JAM_DECODE(&wr[b], &wq, &dW);
                f=_mm256_fmadd_ps(_mm256_set1_ps(dW*sj[b]), JAM_DOT(_mm256_abs_epi8(wq),_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(aj+(size_t)b*32)),wq)), f);
            }
            C[(size_t)j*ldc+i] = jam_hsum8_256(f);
        }
    }
}
