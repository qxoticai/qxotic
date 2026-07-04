/* AVX2 kernels — this TU only, built with -mavx2 -mfma -mf16c. Bound at create when
 * JAM_ISA_AVX2 <= ctx->active < JAM_ISA_AVX512 (consumer / older CPUs, Haswell 2013+).
 *
 * Same structure as the AVX-512 TU but 8-wide (__m256) with only 16 ymm registers, so tiles cap at
 * 4×2 (F32 mnpack; the F16/BF16 dense gemm at the bottom uses a 2×4 dot tile). Q8_0 has NO VNNI here:
 * the int8 dot uses maddubs(|qa|, sign(qb,qa)) -> int16 pairs, then madd_epi16 -> int32 — the standard
 * ggml AVX2 path, same deferred-float accumulation as VNNI. (The 8-feature-wide cached-repack K-quant +
 * Q8_0 rp kernels live in jam_kernels_kquant_avx2.c.) */
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

/* F16/BF16 dense weight @ F32 -> F32, avx2: a 2-weight-row × 4-activation-col dot tile, so each row's
 * vcvtph2ps converts amortize across 4 columns (matches the avx512 4x4's 1:4 convert:FMA ratio). 8 ymm
 * accumulators (+ 2 wv + 4 xv) fit the 16 registers. k stepped by 8 (dispatch gates on k%16==0). */
static inline __m256 loadw_f16_256(const uint16_t* p)  { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) p)); }
static inline __m256 loadw_bf16_256(const uint16_t* p) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) p)), 16)); }

#define JAM_DENSE_AVX2(NAME, LOADW, WTYPE)                                                                \
void NAME(void* arg, int rb, int re, int tid) {                                                           \
    (void) tid;                                                                                           \
    const jam_mm_job* J = (const jam_mm_job*) arg;                                                        \
    const WTYPE* W = (const WTYPE*) J->a;                                                           \
    const float* X = (const float*) J->b;                                                                 \
    float* C = (float*) J->c;                                                                             \
    const int ldw = J->lda, ldx = J->ldb, ldc = J->ldc, n = J->n, k = J->k;                              \
    /* Row-blocked, activation-tile-outer nest: the 4-col x tile (4*k*4B, L1/L2) is loaded once and      \
     * swept across the whole row block (weights ~RB*k*2B, L2-resident), so the activation matrix is     \
     * read from L3 once per BLOCK instead of once per row-pair: L3 activation traffic drops RB/2 x.     \
     * The old nest collapsed at 4096^3 (413 -> 210 GMAC/s) on exactly that traffic. */                  \
    int RB = (int)(512*1024 / ((size_t) k * sizeof(WTYPE))); RB &= ~1; if (RB < 8) RB = 8; if (RB > 128) RB = 128;   \
    for (int r0 = rb; r0 < re; r0 += RB) {                                                                \
        const int rbe = r0 + RB < re ? r0 + RB : re;                                                      \
        int s = 0;                                                                                        \
        for (; s + 4 <= n; s += 4) {                                                                      \
            const float *x0=X+(int64_t)s*ldx,*x1=x0+ldx,*x2=x1+ldx,*x3=x2+ldx;                            \
            int r = r0;                                                                                   \
            for (; r + 3 <= rbe; r += 3) {                                                                \
                const WTYPE *w0=W+(int64_t)r*ldw,*w1=w0+ldw,*w2=w1+ldw;                                \
                __m256 a00=_mm256_setzero_ps(),a01=a00,a02=a00,a03=a00;                                   \
                __m256 a10=a00,a11=a00,a12=a00,a13=a00, a20=a00,a21=a00,a22=a00,a23=a00;                  \
                for (int t = 0; t < k; t += 8) {                                                          \
                    __m256 wv0=LOADW(w0+t),wv1=LOADW(w1+t),wv2=LOADW(w2+t);                               \
                    __m256 xv;                                                                            \
                    xv=_mm256_loadu_ps(x0+t); a00=_mm256_fmadd_ps(wv0,xv,a00);a10=_mm256_fmadd_ps(wv1,xv,a10);a20=_mm256_fmadd_ps(wv2,xv,a20); \
                    xv=_mm256_loadu_ps(x1+t); a01=_mm256_fmadd_ps(wv0,xv,a01);a11=_mm256_fmadd_ps(wv1,xv,a11);a21=_mm256_fmadd_ps(wv2,xv,a21); \
                    xv=_mm256_loadu_ps(x2+t); a02=_mm256_fmadd_ps(wv0,xv,a02);a12=_mm256_fmadd_ps(wv1,xv,a12);a22=_mm256_fmadd_ps(wv2,xv,a22); \
                    xv=_mm256_loadu_ps(x3+t); a03=_mm256_fmadd_ps(wv0,xv,a03);a13=_mm256_fmadd_ps(wv1,xv,a13);a23=_mm256_fmadd_ps(wv2,xv,a23); \
                }                                                                                         \
                float *o0=C+(int64_t)s*ldc+r,*o1=o0+ldc,*o2=o1+ldc,*o3=o2+ldc;                            \
                o0[0]=hsum8(a00);o0[1]=hsum8(a10);o0[2]=hsum8(a20); o1[0]=hsum8(a01);o1[1]=hsum8(a11);o1[2]=hsum8(a21); \
                o2[0]=hsum8(a02);o2[1]=hsum8(a12);o2[2]=hsum8(a22); o3[0]=hsum8(a03);o3[1]=hsum8(a13);o3[2]=hsum8(a23); \
            }                                                                                             \
            for (; r + 2 <= rbe; r += 2) {                                                                \
                const WTYPE *w0=W+(int64_t)r*ldw,*w1=w0+ldw;                                           \
                __m256 a00=_mm256_setzero_ps(),a01=a00,a02=a00,a03=a00, a10=a00,a11=a00,a12=a00,a13=a00;  \
                for (int t = 0; t < k; t += 8) {                                                          \
                    __m256 wv0=LOADW(w0+t),wv1=LOADW(w1+t);                                               \
                    __m256 xv0=_mm256_loadu_ps(x0+t),xv1=_mm256_loadu_ps(x1+t),xv2=_mm256_loadu_ps(x2+t),xv3=_mm256_loadu_ps(x3+t); \
                    a00=_mm256_fmadd_ps(wv0,xv0,a00);a01=_mm256_fmadd_ps(wv0,xv1,a01);a02=_mm256_fmadd_ps(wv0,xv2,a02);a03=_mm256_fmadd_ps(wv0,xv3,a03); \
                    a10=_mm256_fmadd_ps(wv1,xv0,a10);a11=_mm256_fmadd_ps(wv1,xv1,a11);a12=_mm256_fmadd_ps(wv1,xv2,a12);a13=_mm256_fmadd_ps(wv1,xv3,a13); \
                }                                                                                         \
                float *o0=C+(int64_t)s*ldc+r,*o1=o0+ldc,*o2=o1+ldc,*o3=o2+ldc;                            \
                o0[0]=hsum8(a00);o0[1]=hsum8(a10); o1[0]=hsum8(a01);o1[1]=hsum8(a11);                      \
                o2[0]=hsum8(a02);o2[1]=hsum8(a12); o3[0]=hsum8(a03);o3[1]=hsum8(a13);                      \
            }                                                                                             \
            if (r < rbe) {                                                                                \
                const WTYPE* w = W+(int64_t)r*ldw;                                                     \
                __m256 b0=_mm256_setzero_ps(),b1=b0,b2=b0,b3=b0;                                          \
                for (int t = 0; t < k; t += 8) {                                                          \
                    __m256 wv=LOADW(w+t);                                                                 \
                    b0=_mm256_fmadd_ps(wv,_mm256_loadu_ps(x0+t),b0);b1=_mm256_fmadd_ps(wv,_mm256_loadu_ps(x1+t),b1); \
                    b2=_mm256_fmadd_ps(wv,_mm256_loadu_ps(x2+t),b2);b3=_mm256_fmadd_ps(wv,_mm256_loadu_ps(x3+t),b3); \
                }                                                                                         \
                C[(int64_t)s*ldc+r]=hsum8(b0);C[(int64_t)(s+1)*ldc+r]=hsum8(b1);                          \
                C[(int64_t)(s+2)*ldc+r]=hsum8(b2);C[(int64_t)(s+3)*ldc+r]=hsum8(b3);                      \
            }                                                                                             \
        }                                                                                                 \
        for (; s < n; s++) {                                                                              \
            const float* xs = X+(int64_t)s*ldx;                                                           \
            int r = r0;                                                                                   \
            for (; r + 2 <= rbe; r += 2) {                                                                \
                const WTYPE *w0=W+(int64_t)r*ldw,*w1=w0+ldw;                                           \
                __m256 b0=_mm256_setzero_ps(),b1=b0;                                                      \
                for (int t=0;t<k;t+=8){ __m256 xv=_mm256_loadu_ps(xs+t); b0=_mm256_fmadd_ps(LOADW(w0+t),xv,b0);b1=_mm256_fmadd_ps(LOADW(w1+t),xv,b1); } \
                float* o=C+(int64_t)s*ldc+r; o[0]=hsum8(b0);o[1]=hsum8(b1);                                \
            }                                                                                             \
            if (r < rbe) {                                                                                \
                const WTYPE* w = W+(int64_t)r*ldw;                                                     \
                __m256 acc=_mm256_setzero_ps();                                                           \
                for (int t=0;t<k;t+=8) acc=_mm256_fmadd_ps(LOADW(w+t), _mm256_loadu_ps(xs+t), acc);       \
                C[(int64_t)s*ldc+r]=hsum8(acc);                                                           \
            }                                                                                             \
        }                                                                                                 \
    }                                                                                                     \
}
static inline __m256 loadw_f32_256(const float* p) { return _mm256_loadu_ps(p); }
JAM_DENSE_AVX2(jam_mm_f16_avx2,  loadw_f16_256,  uint16_t)
JAM_DENSE_AVX2(jam_mm_bf16_avx2, loadw_bf16_256, uint16_t)
JAM_DENSE_AVX2(jam_mm_f32d_avx2, loadw_f32_256,  float)
