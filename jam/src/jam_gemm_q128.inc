/* 128-bit (SSE3) quantized GEMM engine: C = W @ Aᵀ, W decoded to int8 (two __m128i halves), A int8-requantized.
 * A *true* SSE3 floor: the int8 dot is sign-extend (unpacklo/hi + arithmetic-shift) + madd_epi16 — NO SSSE3
 * maddubs, so no abs/sign trick (signed int16 madd carries the signs). The block reduce is SSE3 haddps and the
 * scale·dot accumulate is mul+add (SSE has no FMA). The includer defines, before #include:
 *
 *   JAM_BLK                            weight block struct (for the row stride)           [per QUANT]
 *   JAM_DECODE(blk, &wlo, &whi, &dW)   decode -> two __m128i int8 halves + float scale    [per QUANT]
 *   JAM_MM_NAME                        exported symbol
 *
 * Requires jam_decode_x86_128.h (jam_hsum4_sse3) and "jam_internal.h" (jam_q8_job) already included. */

/* signed int8 -> int16 (low / high 8 lanes): duplicate each byte then arithmetic-shift right 8 to sign-extend. */
#define JAM_SEXT_LO(x) _mm_srai_epi16(_mm_unpacklo_epi8((x), (x)), 8)
#define JAM_SEXT_HI(x) _mm_srai_epi16(_mm_unpackhi_epi8((x), (x)), 8)
/* dot of 16 signed int8 w and 16 signed int8 a -> 4 int32 partial sums (4 products per lane). */
#define JAM_DOT16(w, a) _mm_add_epi32(_mm_madd_epi16(JAM_SEXT_LO(w), JAM_SEXT_LO(a)), \
                                      _mm_madd_epi16(JAM_SEXT_HI(w), JAM_SEXT_HI(a)))
/* full 32-int8 block dot (two halves) for activation pointer p -> 4 int32 (8 products per lane). */
#define JAM_DOTBLK(wlo, whi, p) _mm_add_epi32( \
        JAM_DOT16((wlo), _mm_loadu_si128((const __m128i*) (p))), \
        JAM_DOT16((whi), _mm_loadu_si128((const __m128i*) ((p) + 16))))

void JAM_MM_NAME(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* W = (const char*) J->a;       /* weights (J->a is the weight operand) */
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, nb = J->nb, k = J->k;
    const size_t wrow = (size_t)(J->lda / 32);
    for (int i = rb; i < re; ++i) {
        const JAM_BLK* wr = (const JAM_BLK*) (W + (size_t) i * wrow * sizeof(JAM_BLK));
        int j = 0;
        for (; j + 4 <= n; j += 4) {          /* tile 4 activation cols, reuse the decoded weight block */
            const int8_t *a0=J->aq+(size_t)(j+0)*k,*a1=J->aq+(size_t)(j+1)*k,*a2=J->aq+(size_t)(j+2)*k,*a3=J->aq+(size_t)(j+3)*k;
            const float  *s0=J->ad+(size_t)(j+0)*nb,*s1=J->ad+(size_t)(j+1)*nb,*s2=J->ad+(size_t)(j+2)*nb,*s3=J->ad+(size_t)(j+3)*nb;
            __m128 f0=_mm_setzero_ps(),f1=_mm_setzero_ps(),f2=_mm_setzero_ps(),f3=_mm_setzero_ps();
            for (int b = 0; b < nb; ++b) {
                __m128i wlo, whi; float dW; JAM_DECODE(&wr[b], &wlo, &whi, &dW);
                f0=_mm_add_ps(f0,_mm_mul_ps(_mm_set1_ps(dW*s0[b]),_mm_cvtepi32_ps(JAM_DOTBLK(wlo,whi,a0+(size_t)b*32))));
                f1=_mm_add_ps(f1,_mm_mul_ps(_mm_set1_ps(dW*s1[b]),_mm_cvtepi32_ps(JAM_DOTBLK(wlo,whi,a1+(size_t)b*32))));
                f2=_mm_add_ps(f2,_mm_mul_ps(_mm_set1_ps(dW*s2[b]),_mm_cvtepi32_ps(JAM_DOTBLK(wlo,whi,a2+(size_t)b*32))));
                f3=_mm_add_ps(f3,_mm_mul_ps(_mm_set1_ps(dW*s3[b]),_mm_cvtepi32_ps(JAM_DOTBLK(wlo,whi,a3+(size_t)b*32))));
            }
            C[(size_t)(j+0)*ldc+i]=jam_hsum4_sse3(f0); C[(size_t)(j+1)*ldc+i]=jam_hsum4_sse3(f1);
            C[(size_t)(j+2)*ldc+i]=jam_hsum4_sse3(f2); C[(size_t)(j+3)*ldc+i]=jam_hsum4_sse3(f3);
        }
        for (; j < n; ++j) {                  /* column tail */
            const int8_t* aj=J->aq+(size_t)j*k; const float* sj=J->ad+(size_t)j*nb;
            __m128 f = _mm_setzero_ps();
            for (int b = 0; b < nb; ++b) {
                __m128i wlo, whi; float dW; JAM_DECODE(&wr[b], &wlo, &whi, &dW);
                f=_mm_add_ps(f,_mm_mul_ps(_mm_set1_ps(dW*sj[b]),_mm_cvtepi32_ps(JAM_DOTBLK(wlo,whi,aj+(size_t)b*32))));
            }
            C[(size_t)j*ldc+i] = jam_hsum4_sse3(f);
        }
    }
}

#undef JAM_SEXT_LO
#undef JAM_SEXT_HI
#undef JAM_DOT16
#undef JAM_DOTBLK
