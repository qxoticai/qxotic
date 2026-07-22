/* AVX-512 kernels — this TU only, built with -mavx512f -mfma; bound at create when
 * ctx->active >= JAM_ISA_AVX512. C = A @ Bᵀ, F32 (== tinyBLAS's C = Aᵀ·B: k is contiguous on both,
 * each output is a dot of two contiguous k-vectors).
 *
 * Design follows llamafile's tinyBLAS: register-tiled kernels + `mnpack` recursive tile selection so
 * the whole region is covered by register tiles down to 1×1 — no scalar edges. Tiles pre-load the RM
 * A-rows and RN B-rows once per k-step (optimal RM+RN loads), and a 4×4 tile holds 16 acc + 4 + 4 =
 * 24/32 zmm, spill-free. Threading is handled by the engine (it hands each thread a row range); each
 * thread mnpacks [row_begin,row_end) × [0,n). */
#include "jam_internal.h"
#include <stddef.h>
#include <stdlib.h>
#include <immintrin.h>

/* Generate gemm_<RM>x<RN>: process every full RM×RN tile in [i0,iend) × [j0,jend). */
#define JAM_GEMM_TILE(RM, RN)                                                                       \
static void gemm_##RM##x##RN(const float* A, long lda, const float* B, long ldb,                   \
                             float* C, long ldc, long i0, long iend, long j0, long jend, long k) {  \
    for (long i = i0; i + RM <= iend; i += RM) {                                                    \
        for (long j = j0; j + RN <= jend; j += RN) {                                                \
            __m512 acc[RN][RM];                                                                     \
            for (int b = 0; b < RN; ++b) for (int a = 0; a < RM; ++a) acc[b][a] = _mm512_setzero_ps(); \
            long t = 0;                                                                             \
            for (; t + 16 <= k; t += 16) {                                                          \
                __m512 av[RM], bv[RN];                                                              \
                for (int a = 0; a < RM; ++a) av[a] = _mm512_loadu_ps(A + (i + a) * lda + t);        \
                for (int b = 0; b < RN; ++b) bv[b] = _mm512_loadu_ps(B + (j + b) * ldb + t);        \
                for (int b = 0; b < RN; ++b) for (int a = 0; a < RM; ++a)                           \
                    acc[b][a] = _mm512_fmadd_ps(av[a], bv[b], acc[b][a]);                           \
            }                                                                                       \
            if (t < k) {                                                                            \
                __mmask16 mk = (__mmask16) ((1u << (k - t)) - 1);                                   \
                __m512 av[RM], bv[RN];                                                              \
                for (int a = 0; a < RM; ++a) av[a] = _mm512_maskz_loadu_ps(mk, A + (i + a) * lda + t); \
                for (int b = 0; b < RN; ++b) bv[b] = _mm512_maskz_loadu_ps(mk, B + (j + b) * ldb + t); \
                for (int b = 0; b < RN; ++b) for (int a = 0; a < RM; ++a)                           \
                    acc[b][a] = _mm512_fmadd_ps(av[a], bv[b], acc[b][a]);                           \
            }                                                                                       \
            for (int b = 0; b < RN; ++b) for (int a = 0; a < RM; ++a)                               \
                C[(size_t)(j + b) * ldc + (i + a)] = _mm512_reduce_add_ps(acc[b][a]);               \
        }                                                                                           \
    }                                                                                               \
}

/* 5x5: 25 acc + 5 A-rows + 1 live B-row = 31 zmm. Per k-step: 10 loads for 25 FMAs - higher
 * flops-per-L2-byte than 4x4 (16 FMAs per 8 loads), which is the difference on Zen 5 where the
 * 4x4 shape sits exactly at the L2 load-bandwidth limit. Mirrors tinyBLAS's AVX-512 tile. */
static void gemm_5x5(const float* A, long lda, const float* B, long ldb,
                     float* C, long ldc, long i0, long iend, long j0, long jend, long k) {
    for (long i = i0; i + 5 <= iend; i += 5) {
        for (long j = j0; j + 5 <= jend; j += 5) {
            __m512 acc[5][5];
            for (int b = 0; b < 5; ++b) for (int a = 0; a < 5; ++a) acc[b][a] = _mm512_setzero_ps();
            long t = 0;
            for (; t + 16 <= k; t += 16) {
                __m512 av[5];
                for (int a = 0; a < 5; ++a) av[a] = _mm512_loadu_ps(A + (i + a) * lda + t);
                for (int b = 0; b < 5; ++b) {
                    __m512 bv = _mm512_loadu_ps(B + (j + b) * ldb + t);
                    for (int a = 0; a < 5; ++a) acc[b][a] = _mm512_fmadd_ps(av[a], bv, acc[b][a]);
                }
            }
            if (t < k) {
                __mmask16 mk = (__mmask16) ((1u << (k - t)) - 1);
                __m512 av[5];
                for (int a = 0; a < 5; ++a) av[a] = _mm512_maskz_loadu_ps(mk, A + (i + a) * lda + t);
                for (int b = 0; b < 5; ++b) {
                    __m512 bv = _mm512_maskz_loadu_ps(mk, B + (j + b) * ldb + t);
                    for (int a = 0; a < 5; ++a) acc[b][a] = _mm512_fmadd_ps(av[a], bv, acc[b][a]);
                }
            }
            for (int b = 0; b < 5; ++b) for (int a = 0; a < 5; ++a)
                C[(size_t)(j + b) * ldc + (i + a)] = _mm512_reduce_add_ps(acc[b][a]);
        }
    }
}

JAM_GEMM_TILE(4,4) JAM_GEMM_TILE(4,2) JAM_GEMM_TILE(4,1)
JAM_GEMM_TILE(2,4) JAM_GEMM_TILE(2,2) JAM_GEMM_TILE(2,1)
JAM_GEMM_TILE(1,4) JAM_GEMM_TILE(1,2) JAM_GEMM_TILE(1,1)

/* tinyBLAS-style mnpack: pick the largest tile (from {4,2,1}²) that fits the remaining M,N; process
 * its full-tile block; recurse on the row remainder then the column remainder. Bottoms out at 1×1. */
static void mnpack(const float* A, long lda, const float* B, long ldb, float* C, long ldc,
                   long i0, long iend, long j0, long jend, long k) {
    long mrem = iend - i0, nrem = jend - j0;
    if (mrem <= 0 || nrem <= 0) return;
    long mc = mrem >= 5 ? 5 : mrem >= 4 ? 4 : mrem >= 2 ? 2 : 1;
    long nc = nrem >= 5 ? 5 : nrem >= 4 ? 4 : nrem >= 2 ? 2 : 1;
    if (mc == 5 && nc != 5) mc = 4;   /* only the 5x5 pairing exists; degrade to the 4-ladder */
    if (nc == 5 && mc != 5) nc = 4;

    switch (mc * 10 + nc) {
        case 55: gemm_5x5(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 44: gemm_4x4(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 42: gemm_4x2(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 41: gemm_4x1(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 24: gemm_2x4(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 22: gemm_2x2(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 21: gemm_2x1(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 14: gemm_1x4(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        case 12: gemm_1x2(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
        default: gemm_1x1(A,lda,B,ldb,C,ldc,i0,iend,j0,jend,k); break;
    }
    long mp = i0 + mrem / mc * mc;   /* last full-tile row boundary */
    long np = j0 + nrem / nc * nc;   /* last full-tile col boundary */
    mnpack(A,lda,B,ldb,C,ldc, mp, iend, j0, np,  k);   /* remainder rows × done cols */
    mnpack(A,lda,B,ldb,C,ldc, i0, iend, np, jend, k);  /* remainder cols × all rows */
}

/* Row-range kernel (jam_task_fn): the engine pool hands each thread output rows [row_begin,row_end).
 * We block the N (activation-column) loop into panels whose k-vectors fit most of L3: B (activations)
 * stays L3-resident and is re-read per M-tile from there, while A (the WEIGHTS - the far larger
 * operand in LLM shapes) streams from DRAM exactly once per panel. The old ~half-L2 panel had that
 * backwards: with ~10 panels the whole multi-GB weight matrix was re-streamed 10x, which is why F32
 * prefill sat at ~half of llama.cpp. Panel width stays a multiple of the 4-wide register tile. */
void jam_mm_f32_avx512(void* arg, int row_begin, int row_end, int tid) {
    (void) tid;
    const jam_mm_job* J = (const jam_mm_job*) arg;
    long k = J->k;
    static long panel_bytes = 0;                           /* one-time read; benign race (idempotent) */
    if (!panel_bytes) {
        const char* pk = getenv("JAM_F32_PANEL_KB");       /* tunable per-uarch; ~3/4 of L3 by default */
        panel_bytes = (pk ? atoi(pk) : 24576) * 1024L;
    }
    long nc = panel_bytes / (k * (long) sizeof(float));    /* cols whose k-vectors fill the panel */
    nc &= ~3L;                                             /* align to the register-tile width */
    if (nc < 4) nc = 4;
    for (long jc = 0; jc < J->n; jc += nc) {
        long jend = jc + nc < J->n ? jc + nc : J->n;
        mnpack((const float*) J->a, J->lda, (const float*) J->b, J->ldb, (float*) J->c, J->ldc,
               row_begin, row_end, jc, jend, k);
    }
}

/* ======================= Q8_0 (weight) × F32 (activation) via AVX-512-VNNI =======================
 * Q8_0 block = { fp16 d; int8 qs[32] }. Dequant(x) = d * qs. C[i,j] = Σ_blk dA·dB · Σ_e qa·qb.
 *
 * VNNI vpdpbusd does UNSIGNED×SIGNED (u8·s8). The weight qa is s8; we make it u8 with uw = qa XOR 0x80
 * (= qa+128) and correct: Σ uw·qb = Σ(qa+128)·qb = Σ qa·qb + 128·Σqb, so Σ qa·qb = vpdpbusd − 128·qbsum.
 * The activation B is requantized once to int8 (per-32-block scale dB + block sum qbsum) into context
 * scratch, then reused across every weight row. Both phases are row-range kernels (auto-multithreaded). */

#include <stdint.h>

typedef struct __attribute__((packed)) { uint16_t d; int8_t qs[32]; } block_q8_0;  /* 34 bytes */

static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

static inline float hsum8f(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
    return _mm_cvtss_f32(s);
}

/* ---- 512-bit VNNI Q8_0 helpers: process a block-PAIR (64 int8 = two Q8_0 blocks) per vpdpbusd ----
 * dpbusd yields 16 i32 lanes: 0..7 = first block's partials, 8..15 = second's, so each pair is scaled
 * by scale2(dA0·dB0, dA1·dB1) and float-accumulated; reduce_add at the end. abs/sign trick keeps the
 * weight unsigned (no +128 correction): dpbusd(|w|, sign(a,w)) = Σ w·a. */
static inline __m512i q8_wpair(const block_q8_0* a, const block_q8_0* b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) a->qs)),
                              _mm256_loadu_si256((const __m256i*) b->qs), 1);
}
static inline __m512i q8_sgnfold(__m512i a, __m512i w) {   /* negate a where w<0 (w==0 nulled by |w|=0) */
    return _mm512_mask_sub_epi8(a, _mm512_movepi8_mask(w), _mm512_setzero_si512(), a);
}
static inline __m512 q8_scl2(float lo, float hi) {         /* lanes 0..7 = lo (block A), 8..15 = hi (block B) */
    return _mm512_insertf32x8(_mm512_set1_ps(lo), _mm256_set1_ps(hi), 1);
}
static inline __m512 q8_dot(__m512i aw, __m512i sa) {
    return _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sa));
}

/* one weight row · one activation col (block-pairs) -> float; used for the m/n tile remainders. */
static inline float q8_bp_row(const block_q8_0* w, const int8_t* a, const float* ad, int nb) {
    __m512 f = _mm512_setzero_ps();
    int blk = 0;
    for (; blk + 2 <= nb; blk += 2) {
        __m512i wp = q8_wpair(&w[blk], &w[blk+1]);
        __m512i ap = _mm512_loadu_si512((const void*)(a + (size_t)blk*32));
        f = _mm512_fmadd_ps(q8_scl2(h2f(w[blk].d)*ad[blk], h2f(w[blk+1].d)*ad[blk+1]),
                            q8_dot(_mm512_abs_epi8(wp), q8_sgnfold(ap, wp)), f);
    }
    if (blk < nb) {
        __m512i wp = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*) w[blk].qs));
        __m512i ap = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)(a + (size_t)blk*32)));
        f = _mm512_fmadd_ps(q8_scl2(h2f(w[blk].d)*ad[blk], 0.f),
                            q8_dot(_mm512_abs_epi8(wp), q8_sgnfold(ap, wp)), f);
    }
    return _mm512_reduce_add_ps(f);
}

/* Phase 1 (requant B -> int8) is the shared scalar jam_q8_0_requant. Phase 2: 512-bit vpdpbusd with an
 * RM×RN register tile (RM weight rows × RN activation cols), processed row-at-a-time so only the current
 * row's weight pair is live alongside RN activation pairs (RM·RN + RN + temps zmm). The RM/RN loops have
 * compile-time bounds so -O3 fully unrolls them and keeps the f[][] accumulators in registers. Tune the
 * tile via -DJAM_Q8_RM / -DJAM_Q8_RN. Output token-major C[(j+c)·ldc + (i+r)]. */
#ifndef JAM_Q8_RM
#define JAM_Q8_RM 4
#endif
#ifndef JAM_Q8_RN
#define JAM_Q8_RN 4
#endif
void jam_mm_q8_0_avx512(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* A = (const char*) J->a;
    float* C = (float*) J->c;
    const int ldc = J->ldc, n = J->n, k = J->k, nb = J->nb;
    const size_t wrow = (size_t)(J->lda / 32);          /* weight-row stride in blocks */
    enum { RM = JAM_Q8_RM, RN = JAM_Q8_RN };
    #define WROW(ii) ((const block_q8_0*) (A + (size_t)(ii) * wrow * sizeof(block_q8_0)))
    int i = rb;
    for (; i + RM <= re; i += RM) {
        const block_q8_0* w[RM];
        for (int r = 0; r < RM; r++) w[r] = WROW(i + r);
        int j = 0;
        for (; j + RN <= n; j += RN) {
            const int8_t* a[RN]; const float* b[RN];
            for (int c = 0; c < RN; c++) { a[c] = J->aq + (size_t)(j+c)*k; b[c] = J->ad + (size_t)(j+c)*nb; }
            __m512 f[RM][RN];
            for (int r = 0; r < RM; r++) for (int c = 0; c < RN; c++) f[r][c] = _mm512_setzero_ps();
            int blk = 0;
            for (; blk + 2 <= nb; blk += 2) {                 /* block-pair: 64 int8 per vpdpbusd */
                __m512i ap[RN]; float al[RN], ah[RN];
                for (int c = 0; c < RN; c++) {
                    ap[c] = _mm512_loadu_si512((const void*)(a[c] + (size_t)blk*32));
                    al[c] = b[c][blk]; ah[c] = b[c][blk+1];
                }
                for (int r = 0; r < RM; r++) {
                    __m512i wp = q8_wpair(&w[r][blk], &w[r][blk+1]), aw = _mm512_abs_epi8(wp);
                    float wl = h2f(w[r][blk].d), wh = h2f(w[r][blk+1].d);
                    for (int c = 0; c < RN; c++)
                        f[r][c] = _mm512_fmadd_ps(q8_scl2(wl*al[c], wh*ah[c]), q8_dot(aw, q8_sgnfold(ap[c], wp)), f[r][c]);
                }
            }
            if (blk < nb) {                                   /* odd block: zero-extend, hi scale = 0 */
                __m512i ap[RN]; float al[RN];
                for (int c = 0; c < RN; c++) {
                    ap[c] = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)(a[c] + (size_t)blk*32)));
                    al[c] = b[c][blk];
                }
                for (int r = 0; r < RM; r++) {
                    __m512i wp = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*) w[r][blk].qs)), aw = _mm512_abs_epi8(wp);
                    float wl = h2f(w[r][blk].d);
                    for (int c = 0; c < RN; c++)
                        f[r][c] = _mm512_fmadd_ps(q8_scl2(wl*al[c], 0.f), q8_dot(aw, q8_sgnfold(ap[c], wp)), f[r][c]);
                }
            }
            for (int r = 0; r < RM; r++) for (int c = 0; c < RN; c++)
                C[(size_t)(j+c)*ldc + (i+r)] = _mm512_reduce_add_ps(f[r][c]);
        }
        for (; j < n; ++j)                                    /* column remainder: RM rows × 1 col */
            for (int r = 0; r < RM; r++)
                C[(size_t)j*ldc + (i+r)] = q8_bp_row(w[r], J->aq+(size_t)j*k, J->ad+(size_t)j*nb, nb);
    }
    for (; i < re; ++i) {                                     /* row remainder: 1 row × all cols */
        const block_q8_0* w = WROW(i);
        for (int j = 0; j < n; ++j)
            C[(size_t)j*ldc+i] = q8_bp_row(w, J->aq+(size_t)j*k, J->ad+(size_t)j*nb, nb);
    }
    #undef WROW
}

/* ---- Q8_0 MATVEC (n==1, decode). Bandwidth-bound: each weight row read once from DRAM, dotted with
 * the single (cached, pre-requantized) activation column. SW-prefetch rows ahead to hide DRAM latency;
 * the row-range fan-out keeps every core streaming a different slice. Output token-major n=1 -> C[i]. */
void jam_mm_q8_0_gemv_avx512(void* arg, int rb, int re, int tid) {
    (void) tid;
    const jam_q8_job* J = (const jam_q8_job*) arg;
    const char* A = (const char*) J->a;
    float* C = (float*) J->c;
    const int nb = J->nb;
    const size_t wstride = (size_t)(J->lda / 32) * sizeof(block_q8_0);   /* bytes per weight row */
    const int8_t* aq = J->aq; const float* ad = J->ad;
    for (int i = rb; i < re; ++i) {
        const char* wp = A + (size_t) i * wstride;
        if (i + 3 < re) {                                              /* prefetch ~3 rows ahead */
            const char* pf = A + (size_t)(i + 3) * wstride;
            _mm_prefetch(pf, _MM_HINT_T0);
            _mm_prefetch(pf + 64, _MM_HINT_T0);
        }
        C[i] = q8_bp_row((const block_q8_0*) wp, aq, ad, nb);
    }
}

/* ---- F16 / BF16 DENSE weight @ F32 -> F32 (ported from jinferjni.c run_dense_gemm). 4×4 register
 * tile, 16-wide; the weight is converted to f32 on the fly (cvtph for F16, <<16 for BF16). Output is
 * token-major C[s*ldc + r] — already jam's layout. k must be a multiple of 16 (else the generic floor). */
static inline __m512 jam_loadw_f16(const uint16_t* p)  { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) p)); }
static inline __m512 jam_loadw_bf16(const uint16_t* p) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) p)), 16)); }

#define JAM_DENSE(NAME, LOADW)                                                                         \
void NAME(void* arg, int rb, int re, int tid) {                                                       \
    (void) tid;                                                                                        \
    const jam_mm_job* J = (const jam_mm_job*) arg;                                                     \
    const uint16_t* W = (const uint16_t*) J->a;                                                        \
    const float* X = (const float*) J->b;                                                              \
    float* C = (float*) J->c;                                                                          \
    const int ldw = J->lda, ldx = J->ldb, ldc = J->ldc, n = J->n, k = J->k;                           \
    int r = rb;                                                                                        \
    for (; r + 4 <= re; r += 4) {                                                                      \
        const uint16_t *w0=W+(int64_t)r*ldw,*w1=w0+ldw,*w2=w1+ldw,*w3=w2+ldw;                          \
        int s = 0;                                                                                     \
        for (; s + 4 <= n; s += 4) {                                                                   \
            const float *x0=X+(int64_t)s*ldx,*x1=x0+ldx,*x2=x1+ldx,*x3=x2+ldx;                         \
            __m512 a00=_mm512_setzero_ps(),a01=a00,a02=a00,a03=a00, a10=a00,a11=a00,a12=a00,a13=a00,   \
                   a20=a00,a21=a00,a22=a00,a23=a00, a30=a00,a31=a00,a32=a00,a33=a00;                   \
            for (int t = 0; t < k; t += 16) {                                                          \
                __m512 wv0=LOADW(w0+t),wv1=LOADW(w1+t),wv2=LOADW(w2+t),wv3=LOADW(w3+t);                \
                __m512 xv0=_mm512_loadu_ps(x0+t),xv1=_mm512_loadu_ps(x1+t),xv2=_mm512_loadu_ps(x2+t),xv3=_mm512_loadu_ps(x3+t); \
                a00=_mm512_fmadd_ps(wv0,xv0,a00);a01=_mm512_fmadd_ps(wv1,xv0,a01);a02=_mm512_fmadd_ps(wv2,xv0,a02);a03=_mm512_fmadd_ps(wv3,xv0,a03); \
                a10=_mm512_fmadd_ps(wv0,xv1,a10);a11=_mm512_fmadd_ps(wv1,xv1,a11);a12=_mm512_fmadd_ps(wv2,xv1,a12);a13=_mm512_fmadd_ps(wv3,xv1,a13); \
                a20=_mm512_fmadd_ps(wv0,xv2,a20);a21=_mm512_fmadd_ps(wv1,xv2,a21);a22=_mm512_fmadd_ps(wv2,xv2,a22);a23=_mm512_fmadd_ps(wv3,xv2,a23); \
                a30=_mm512_fmadd_ps(wv0,xv3,a30);a31=_mm512_fmadd_ps(wv1,xv3,a31);a32=_mm512_fmadd_ps(wv2,xv3,a32);a33=_mm512_fmadd_ps(wv3,xv3,a33); \
            }                                                                                          \
            float *o0=C+(int64_t)s*ldc+r,*o1=o0+ldc,*o2=o1+ldc,*o3=o2+ldc;                             \
            o0[0]=_mm512_reduce_add_ps(a00);o0[1]=_mm512_reduce_add_ps(a01);o0[2]=_mm512_reduce_add_ps(a02);o0[3]=_mm512_reduce_add_ps(a03); \
            o1[0]=_mm512_reduce_add_ps(a10);o1[1]=_mm512_reduce_add_ps(a11);o1[2]=_mm512_reduce_add_ps(a12);o1[3]=_mm512_reduce_add_ps(a13); \
            o2[0]=_mm512_reduce_add_ps(a20);o2[1]=_mm512_reduce_add_ps(a21);o2[2]=_mm512_reduce_add_ps(a22);o2[3]=_mm512_reduce_add_ps(a23); \
            o3[0]=_mm512_reduce_add_ps(a30);o3[1]=_mm512_reduce_add_ps(a31);o3[2]=_mm512_reduce_add_ps(a32);o3[3]=_mm512_reduce_add_ps(a33); \
        }                                                                                              \
        for (; s < n; s++) {                                                                           \
            const float* xs = X+(int64_t)s*ldx;                                                        \
            __m512 b0=_mm512_setzero_ps(),b1=b0,b2=b0,b3=b0;                                           \
            for (int t=0;t<k;t+=16){ __m512 xv=_mm512_loadu_ps(xs+t);                                  \
                b0=_mm512_fmadd_ps(LOADW(w0+t),xv,b0);b1=_mm512_fmadd_ps(LOADW(w1+t),xv,b1);           \
                b2=_mm512_fmadd_ps(LOADW(w2+t),xv,b2);b3=_mm512_fmadd_ps(LOADW(w3+t),xv,b3); }         \
            float* o=C+(int64_t)s*ldc+r;                                                               \
            o[0]=_mm512_reduce_add_ps(b0);o[1]=_mm512_reduce_add_ps(b1);o[2]=_mm512_reduce_add_ps(b2);o[3]=_mm512_reduce_add_ps(b3); \
        }                                                                                              \
    }                                                                                                  \
    for (; r < re; r++) {                                                                              \
        const uint16_t* w = W+(int64_t)r*ldw;                                                          \
        for (int s = 0; s < n; s++) {                                                                  \
            const float* xs = X+(int64_t)s*ldx;                                                        \
            __m512 acc=_mm512_setzero_ps();                                                            \
            for (int t=0;t<k;t+=16) acc=_mm512_fmadd_ps(LOADW(w+t), _mm512_loadu_ps(xs+t), acc);       \
            C[(int64_t)s*ldc+r]=_mm512_reduce_add_ps(acc);                                             \
        }                                                                                              \
    }                                                                                                  \
}
JAM_DENSE(jam_mm_f16_avx512,  jam_loadw_f16)
JAM_DENSE(jam_mm_bf16_avx512, jam_loadw_bf16)
