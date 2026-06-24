/* AVX2 kernels — this TU only, built with -mavx2 -mfma -mf16c. Bound at create when
 * JAM_ISA_AVX2 <= ctx->active < JAM_ISA_AVX512 (consumer / older CPUs, Haswell 2013+).
 *
 * Same structure as the AVX-512 TU but 8-wide (__m256) with only 16 ymm registers, so tiles cap at
 * 4×2. Q8_0 has NO VNNI here: the int8 dot uses maddubs(|qa|, sign(qb,qa)) -> int16 pairs, then
 * madd_epi16 -> int32 — the standard ggml AVX2 path, same deferred-float accumulation as VNNI. */
#include "jam_internal.h"
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

static inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
    return _mm_cvtss_f32(s);
}

/* mask for the last partial 8-lane chunk (lanes 0..r-1 set) — AVX2 has no mask registers */
static const int32_t JAM_MASK_TAB[16] = { -1,-1,-1,-1,-1,-1,-1,-1, 0,0,0,0,0,0,0,0 };
static inline __m256i tailmask(int r) { return _mm256_loadu_si256((const __m256i*)(JAM_MASK_TAB + (8 - r))); }

/* ---- F32: C = A @ Bᵀ, register-tiled + mnpack (tiles cap at 4×2: 8 acc + 4 + 2 = 14/16 ymm) ---- */
#define JAM_GEMM_TILE_AVX2(RM, RN)                                                                  \
static void gemm2_##RM##x##RN(const float* A, long lda, const float* B, long ldb,                   \
                              float* C, long ldc, long i0, long iend, long j0, long jend, long k) {  \
    for (long i = i0; i + RM <= iend; i += RM) {                                                     \
        for (long j = j0; j + RN <= jend; j += RN) {                                                 \
            __m256 acc[RN][RM];                                                                      \
            for (int b=0;b<RN;++b) for (int a=0;a<RM;++a) acc[b][a]=_mm256_setzero_ps();             \
            long t = 0;                                                                              \
            for (; t + 8 <= k; t += 8) {                                                             \
                __m256 av[RM], bv[RN];                                                               \
                for (int a=0;a<RM;++a) av[a]=_mm256_loadu_ps(A+(i+a)*lda+t);                         \
                for (int b=0;b<RN;++b) bv[b]=_mm256_loadu_ps(B+(j+b)*ldb+t);                         \
                for (int b=0;b<RN;++b) for (int a=0;a<RM;++a) acc[b][a]=_mm256_fmadd_ps(av[a],bv[b],acc[b][a]); \
            }                                                                                        \
            if (t < k) {                                                                             \
                __m256i mk = tailmask((int)(k - t));                                                 \
                __m256 av[RM], bv[RN];                                                               \
                for (int a=0;a<RM;++a) av[a]=_mm256_maskload_ps(A+(i+a)*lda+t, mk);                  \
                for (int b=0;b<RN;++b) bv[b]=_mm256_maskload_ps(B+(j+b)*ldb+t, mk);                  \
                for (int b=0;b<RN;++b) for (int a=0;a<RM;++a) acc[b][a]=_mm256_fmadd_ps(av[a],bv[b],acc[b][a]); \
            }                                                                                        \
            for (int b=0;b<RN;++b) for (int a=0;a<RM;++a) C[(size_t)(j+b)*ldc+(i+a)]=hsum8(acc[b][a]); \
        }                                                                                            \
    }                                                                                                \
}
JAM_GEMM_TILE_AVX2(4,2) JAM_GEMM_TILE_AVX2(4,1)
JAM_GEMM_TILE_AVX2(2,2) JAM_GEMM_TILE_AVX2(2,1)
JAM_GEMM_TILE_AVX2(1,2) JAM_GEMM_TILE_AVX2(1,1)

static void mnpack2(const float* A, long lda, const float* B, long ldb, float* C, long ldc,
                    long i0, long iend, long j0, long jend, long k) {
    long mrem = iend - i0, nrem = jend - j0;
    if (mrem <= 0 || nrem <= 0) return;
    long mc = mrem >= 4 ? 4 : mrem >= 2 ? 2 : 1;
    long nc = nrem >= 2 ? 2 : 1;
    switch (mc * 10 + nc) {
        case 42: gemm2_4x2(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 41: gemm2_4x1(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 22: gemm2_2x2(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 21: gemm2_2x1(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 12: gemm2_1x2(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        default: gemm2_1x1(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
    }
    long mp = i0 + mrem / mc * mc, np = j0 + nrem / nc * nc;
    mnpack2(A,lda,B,ldb,C,ldc, mp, iend, j0, np,  k);
    mnpack2(A,lda,B,ldb,C,ldc, i0, iend, np, jend, k);
}

void jam_mm_f32_avx2(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_mm_job* J = (const jam_mm_job*) arg;
    mnpack2((const float*) J->a, J->lda, (const float*) J->b, J->ldb, (float*) J->c, J->ldc,
            rb, re, 0, J->n, J->k);
}

/* ---- quantized weight @ F32 -> F32, AVX2: int8 dot via maddubs+madd (no VNNI). One engine, per-quant
 * decoder. maddubs(|w|,sign) -> int16 pairs, madd_epi16 -> 8 int32 (same lanes vpdpbusd would give). */
#include "jam_decode_x86_256.h"
#define JAM_DOT(aqa, sqb) \
    _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aqa, sqb), _mm256_set1_epi16(1)))

#define JAM_BLK    jam_q8_blk
#define JAM_DECODE jam_decode_q8_0_256
#define JAM_MM_NAME jam_mm_q8_0_avx2
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK    jam_mxfp4_blk
#define JAM_DECODE jam_decode_mxfp4_256
#define JAM_MM_NAME jam_mm_mxfp4_avx2
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME

#define JAM_BLK    jam_q4_0_blk
#define JAM_DECODE jam_decode_q4_0_256
#define JAM_MM_NAME jam_mm_q4_0_avx2
#include "jam_gemm_q256.inc"
#undef JAM_BLK
#undef JAM_DECODE
#undef JAM_MM_NAME
#undef JAM_DOT
