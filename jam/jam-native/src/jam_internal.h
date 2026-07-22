#ifndef JAM_INTERNAL_H
#define JAM_INTERNAL_H

#include "jam.h"
#include "jam_cpu.h"  /* jam_cpu_plan (core selection, stored on the ctx for the log) */
#include <stddef.h>   /* size_t */
#include <stdint.h>   /* int8_t / int32_t */
#include <stdlib.h>

/* Portable 64-byte-aligned alloc/free. POSIX uses C11 aligned_alloc (size rounded up to a multiple of the
 * alignment, as C11 requires) freed with free(); Windows uses _aligned_malloc, which MUST pair with
 * _aligned_free (never free()). Only the VNNI repack scratch (qs/dw/mw) needs alignment. */
#if defined(_WIN32)
#  include <malloc.h>
static inline void* jam_aligned_alloc(size_t align, size_t size) { return _aligned_malloc(size, align); }
static inline void  jam_aligned_free(void* p) { _aligned_free(p); }
#else
static inline void* jam_aligned_alloc(size_t align, size_t size) {
    return aligned_alloc(align, (size + align - 1) & ~(align - 1));
}
static inline void  jam_aligned_free(void* p) { free(p); }
#endif

/* ---- internal thread pool (used when no host parallel_for is supplied, e.g. the global ctx) ---- */
typedef struct jam_pool jam_pool;
jam_pool* jam_pool_create(int nthreads, const int* cpu, int nprimary); /* nthreads participants incl. the submitter;
                                                          * cpu[] (len nthreads, or NULL) pins worker k to
                                                          * cpu[k+1] best-effort (cpu[0] = unpinned submitter) */
void      jam_pool_destroy(jam_pool* pool);
void      jam_pool_parallel_for(jam_pool* pool, int n, jam_task_fn fn, void* arg);  /* blocks till done */
void      jam_pool_parallel_for_capped(jam_pool* pool, int n, jam_task_fn fn, void* arg, int cap);
int       jam_pool_is_spin(const jam_pool* pool);        /* 1 if JAM_POOL=spin (spin-then-park barrier) */
int       jam_pool_spin_budget(const jam_pool* pool);    /* pauses before a spinning worker parks */

/* Per-worker K-quant weight-repack scratch (VNNI layout). One per pool worker, indexed by jam tid. */
typedef struct { uint8_t* qs; float* dw; float* mw; int cap_blocks; } jam_repack;

/* A weight-repack kernel: packs feature rows [rows0,re) of a quantized weight (block stride `w_stride`
 * bytes) into the 8-feature-wide VNNI rpblock layout at `out`. One per quant family (Q4_K/Q5_K/Q6_K/Q8_0). */
typedef void (*jam_repack_fn)(const void* w, int rows0, int re, int blocks, size_t w_stride, void* out);

/* The 256-element K-quants share one dispatch path (dispatch_kquant); they differ only in a few values.
 * The ISA-bound ones (kernel/requant/repack, set per ISA at create) live in ctx->kq[], indexed by jam_kq;
 * the compile-time ones (band kernel, block bytes, rpblock size, float floor) are jam.c's kquant_info[]. */
enum jam_kq { JAM_KQ_Q4K, JAM_KQ_Q5K, JAM_KQ_Q6K, JAM_KQ_N };
typedef struct { jam_task_fn kernel, requant; jam_repack_fn repack; } jam_kquant;

struct jam_ctx {
    jam_parallel_for parallel_for;   /* host executor; if set, takes precedence over ipool */
    void*            pool;           /* opaque handle for parallel_for */
    jam_pool*        ipool;          /* jam-owned pool (NULL if a host executor was supplied) */
    int              nthreads;
    int              nthreads_bw;    /* fan cap for bandwidth-bound phases (one per physical core) */
    jam_cpu_plan     cpu;            /* the core-selection plan used to size + pin the pool (for the log) */
    jam_isa          active;         /* the bound ISA level (reported by jam_active_isa) */
    char             name[48];       /* optional label for JAM_DEBUG ("" if unnamed) */
    _Atomic int      busy;           /* serial-stream guard: jam_mm try-acquires (1 exchange); else EBUSY */
    jam_task_fn      f32_kernel;     /* best F32 row-range kernel for `active`, resolved at create */
    jam_task_fn      q8_kernel;      /* best Q8_0 matmul (phase 2); NULL -> generic. Resolved at create
                                      * by explicit feature check — AVX-VNNI is orthogonal to the ladder. */
    jam_task_fn      mxfp4_kernel;   /* best MXFP4 matmul; NULL -> generic (float). Same int8 pipeline. */
    jam_task_fn      nvfp4_kernel;   /* best NVFP4 matmul; NULL -> generic (float). No SIMD kernel yet. */
    jam_task_fn      q1_0_kernel;    /* best Q1_0 (1-bit sign) matmul; NULL -> generic (float). Int8 pipeline. */
    jam_task_fn      q4_0_kernel;    /* best Q4_0 matmul; NULL -> generic. Same int8 pipeline. */
    jam_kquant       kq[JAM_KQ_N];   /* Q4_K/Q5_K/Q6_K ISA-bound {kernel,requant,repack}; consts in kquant_info[] */
    struct jam_rpentry { const void* w; int m, k; jam_repack_fn repack; void* buf; } *rp_cache;
                                          /* repacked-weight cache, keyed on (weight ptr, shape, repack fn) */
    int rp_cache_n, rp_cache_cap;
    jam_task_fn      q8_0_rp_kernel; /* avx2 cached-repack Q8_0 (sign-trick maddubs 8-wide); NULL -> q8_kernel */
    jam_repack_fn    q8_0_repack;    /* non-NULL -> cached weight-repack for q8_0_rp_kernel */
    jam_task_fn      mxfp4_rp_kernel; /* avx2 cached-repack MXFP4 (|w|<=12 int16-deferred madd); NULL -> mxfp4_kernel */
    jam_repack_fn    mxfp4_repack;    /* non-NULL -> cached weight-repack for mxfp4_rp_kernel */
    jam_repack_fn    q4_0_repack;    /* non-NULL -> cached weight-repack (Q4_0 raw nibble) for deferred -8 correction */
    jam_task_fn      dense_f16_kernel;   /* AVX-512 F16 dense (k%16==0); NULL -> generic floor */
    jam_task_fn      dense_f32_kernel;   /* row-blocked dense F32 (avx2, k%8==0); NULL where mnpack wins (avx512) */
    jam_task_fn      dense_bf16_kernel;  /* AVX-512 BF16 dense (k%16==0); NULL -> generic floor */
    jam_task_fn      bf16z_kernel;       /* AVX512-BF16 vdpbf16ps tile (k%32==0); NULL -> dense_bf16 */
    jam_task_fn      bf16z_cvt;          /* its phase 1: F32 activations -> bf16 scratch */
    jam_task_fn      bf16zp_kernel;      /* packed-panel vdpbf16ps microkernel (k%2==0, n>=8) */
    jam_task_fn      bf16zp_pack;        /* its phase 1: convert+transpose into token panels */

    /* Q8_0 VNNI activation-requant scratch (context-owned, grown lazily). Assumes a context is used
     * serially (the global ctx by jinfer's forward thread); concurrent jam_mm on ONE ctx would race
     * this — TODO for a concurrent-safe variant. The generic path needs none of it. */
    void*  q_aq;   size_t q_aq_cap;   /* int8 [n*k] requantized activations */
    void*  q_ad;   size_t q_d_cap;   /* float [n*nb] per-block scales (q_asum shares this cap) */
    void*  q_asum;                    /* float [n*nb] per-block Σ int8 acts (K-quant min term) */

    /* BF16 activation-conversion scratch (vdpbf16ps path) — same serial-stream contract as q_aq. */
    void*  bf_x;   size_t bf_x_cap;   /* bf16 [n*k] converted activations */
    void*  f32_xp; size_t f32_xp_cap; /* f32 [npanels*32*k] transposed activation panels */
    jam_task_fn f32p_kernel, f32p_pack;  /* packed-panel F32 path (avx512); NULL -> mnpack */

    void*  metal;                    /* jam_metal* GPU backend, or NULL (Apple, opt-in). Routes before CPU. */

    /* K-quant scratch (Q4_K...): s8 activations (xq/dx/xsum) + per-worker weight repack. Context-owned,
     * grown lazily; serial-stream only (same contract as the Q8 requant scratch). */
    int8_t* kq_xq;  size_t kq_xq_cap;
    float*  kq_dx;  size_t kq_dx_cap;     /* xsum cap = 2*kq_dx_cap */
    float*  kq_xsum;
    jam_repack* kq_repack; int kq_repack_n;
    int     q4k_avail;                    /* AVX-512-VNNI Q4_K kernel bound */
};

/* A matmul job handed to the row-range kernels. The kernel computes output rows [begin, end).
 * Matches jam_task_fn so it can be dispatched directly through a parallel_for. */
typedef struct {
    const void* a; int at; int lda;
    const void* b; int bt; int ldb;
    void*       c; int ct; int ldc;
    int n, k;
} jam_mm_job;

/* Portable-C floor — always built, always available. Per-ISA kernels (jam_kernels_avx2.c,
 * jam_kernels_avx512.c, jam_kernels_neon.c, jam_kernels_sme.c, ...) live in their own TUs compiled
 * with their -m flags and bound at create; this scalar one is the fallback and the reference. */
void jam_mm_f32_generic(void* job, int row_begin, int row_end, int tid);

#ifdef JAM_HAVE_AVX2
void jam_mm_f32_avx2(void* job, int row_begin, int row_end, int tid);
void jam_mm_f32d_avx2(void* job, int row_begin, int row_end, int tid);       /* F32 dense, row-blocked 3x4 tile */
void jam_mm_f16_avx2(void* job, int row_begin, int row_end, int tid);       /* F16 dense, avx2 2×4 tile */
void jam_mm_bf16_avx2(void* job, int row_begin, int row_end, int tid);      /* BF16 dense, avx2 2×4 tile */
#endif
#ifdef JAM_HAVE_AVX512
void jam_mm_f32_avx512(void* job, int row_begin, int row_end, int tid);
void jam_mm_f16_avx512(void* job, int row_begin, int row_end, int tid);     /* F16 dense, 4×4 tile */
void jam_mm_bf16_avx512(void* job, int row_begin, int row_end, int tid);    /* BF16 dense, 4×4 tile */
void jam_f32_pack_avx512(void* job, int p_begin, int p_end, int tid);       /* token-panel transpose */
void jam_mm_f32p_avx512(void* job, int row_begin, int row_end, int tid);    /* packed broadcast 8x32 */
#endif

/* ---- packed-panel F32: phase 1 transposes activations into xp, phase 2 broadcast-FMAs ---- */
typedef struct {
    const float* w; long ldw;
    const float* x; long ldx;
    float*       xp;             /* [ceil(n/32)*32*k] token-major panels */
    void*        c; long ldc;
    int n; long k;
} jam_f32p_job;
#ifdef JAM_HAVE_AVX512BF16
void jam_bf16_cvt_avx512bf16(void* job, int row_begin, int row_end, int tid);  /* F32 -> bf16 scratch */
void jam_mm_bf16_avx512bf16(void* job, int row_begin, int row_end, int tid);   /* vdpbf16ps 4×4 tile */
void jam_bf16_pack_avx512bf16(void* job, int p_begin, int p_end, int tid);     /* panel convert+transpose */
void jam_mm_bf16p_avx512bf16(void* job, int row_begin, int row_end, int tid);  /* packed broadcast 8x32 */
#endif

/* ---- BF16 (weight) @ F32 (activation) via vdpbf16ps: phase 1 converts activations into xb ---- */
typedef struct {
    const uint16_t* w; long ldw;    /* BF16 weight [m×k] */
    const float*    x; long ldx;    /* F32 activation [n×k] */
    uint16_t*       xb;             /* [n*k] bf16-converted activations (phase-1 output) */
    void*           c; long ldc;    /* F32 output, token-major */
    int n; long k;
} jam_bf16_job;

/* ---- Q8_0 (weight) @ F32 (activation) -> F32 ----
 * aq/ad/asum are the requantized-B scratch used ONLY by the VNNI path (requant phase fills them, the
 * matmul phase reads them). The generic path dequantizes the weight on the fly and ignores them. */
#include <stdint.h>
typedef struct {
    const void* a; int lda;     /* Q8_0 weight [m×k] (k, lda multiples of 32) */
    const void* b; int ldb;     /* F32 activation [n×k] */
    void*       c; int ldc;     /* F32 output [m×n] */
    int n, k, nb;               /* nb = k/32 */
    int8_t*  aq;                /* [n*k]  requantized activations (VNNI) */
    float*   ad;                /* [n*nb] per-block activation scales */
    float*   asum;              /* [n*nb] per-block Σ(int8 activations) — K-quant dmin·min term (or NULL) */
    int      m;                 /* #output rows (features); the group-indexed rp kernels read this, NOT ldc
                                 * (the API permits ldc > m, so ldc must not be reused as the row count) */
} jam_q8_job;

void jam_mm_q8_0_f32_generic(void* job, int row_begin, int row_end, int tid);  /* portable floor */
void jam_mm_mxfp4_f32_generic(void* job, int row_begin, int row_end, int tid); /* portable floor */
void jam_mm_nvfp4_f32_generic(void* job, int row_begin, int row_end, int tid); /* portable floor (NVFP4) */
void jam_mm_q1_0_f32_generic(void* job, int row_begin, int row_end, int tid);  /* portable floor (Q1_0) */
void jam_mm_q4k_f32_generic(void* job, int row_begin, int row_end, int tid);   /* portable floor (q8_job) */

/* ---- Q4_K @ F32 (AVX-512-VNNI; ported from jinferjni.c). repack scratch is PER WORKER (jam tid). ---- */
typedef struct {
    const uint8_t* w; int64_t w_stride;   /* Q4_K weights, bytes per row = (k/256)*144 */
    const float* rhs; int rhs_stride;     /* F32 activations [seq×k], stride in elements */
    int8_t* xq; float* dx; float* xsum;   /* phase-1 s8 activations + per-32 scales + per-16 sums */
    float* out; int out_stride;           /* feature-major C[dim0×seq], ldc = out_stride */
    int dim0, dim1, seq, kblocks;
    jam_repack* repack;                   /* [ctx->nthreads] */
} jam_q4k_job;
#ifdef JAM_HAVE_AVX512
void jam_q4k_quant(void* job, int s0, int s1, int tid);   /* phase 1: quantize activations to s8 (SHARED) */
void jam_q4k_band(void* job, int t0, int t1, int tid);    /* phase 2: Q4_K repack + VNNI matmul */
void jam_q6k_band(void* job, int t0, int t1, int tid);    /* phase 2: Q6_K repack + VNNI matmul */
void jam_q8_0_repack_band(void* job, int t0, int t1, int tid); /* phase 2: Q8_0 16-row VNNI repack matmul */
void jam_q1_0_repack_band(void* job, int t0, int t1, int tid); /* phase 2: Q1_0 packed-sign-bit VNNI band */
void jam_q4_0_repack_band(void* job, int t0, int t1, int tid); /* phase 2: Q4_0 16-row VNNI repack matmul */
void jam_mxfp4_repack_band(void* job, int t0, int t1, int tid); /* phase 2: MXFP4 16-row VNNI repack matmul */
void jam_q5k_repack_band(void* job, int t0, int t1, int tid);  /* phase 2: Q5_K 16-row VNNI repack matmul */
#endif
/* 256-bit AVX-VNNI Q8_0 band (8-row groups) — the no-AVX-512 client path. Defined in the avxvnni TU;
 * shares the jam_q4k_job + per-worker repack scratch with the AVX-512 band. Dispatch wiring (activation
 * requant + try-band for active==AVX_VNNI) is the next step. */
void jam_q8_0_repack_band_avxvnni(void* job, int t0, int t1, int tid);
void jam_q8_0_requant_256(void* job, int s0, int s1, int tid);   /* pure-256 phase-1 requant for the band */
void jam_q4_0_repack_band_avxvnni(void* job, int t0, int t1, int tid);
void jam_mm_q6k_f32_generic(void* job, int row_begin, int row_end, int tid);    /* portable floor (q8_job) */
void jam_mm_q5k_f32_generic(void* job, int row_begin, int row_end, int tid);    /* portable floor (no VNNI) */
void jam_mm_f16_f32_generic(void* job, int row_begin, int row_end, int tid);    /* F16 dense portable floor */
void jam_mm_bf16_f32_generic(void* job, int row_begin, int row_end, int tid);   /* BF16 dense portable floor */
void jam_q8_0_requant(void* job, int b_begin, int b_end, int tid);             /* phase 1: A -> int8 (shared) */

void jam_mm_q4_0_f32_generic(void* job, int row_begin, int row_end, int tid);  /* portable floor (q8_job) */
/* Cached-repack rpblock layouts — shared by the avx2 + avx-vnni 8-feature-wide gemms and jam.c dispatch. */
typedef struct { float d[8], dmin[8]; uint8_t sc[64], mn[64], qs[1024]; } jam_q4k_rpblock;
typedef struct { float d[8], dmin[8]; uint8_t sc[64], mn[64], qs[2048]; } jam_q5k_rpblock;
typedef struct { float d[8]; int8_t sc[128]; uint8_t qs[2048]; } jam_q6k_rpblock;
typedef struct { float d[8]; int8_t qs[256]; } jam_q8_0_rpblock;
#ifdef JAM_HAVE_AVX2
void jam_mm_mxfp4_avx2(void* job, int a_begin, int a_end, int tid);        /* maddubs + FP4 decode */
void jam_mm_q4_0_avx2(void* job, int a_begin, int a_end, int tid);         /* maddubs + nibble-8 decode */
void jam_q4k_repack8(const void* w, int rows0, int re, int sblocks, size_t w_stride, void* out);  /* repack 8 features (Q4_K) */
void jam_mm_q4k_rp_avx2(void* job, int rb, int re, int tid);             /* cached-repack Q4_K gemm (rb..re = groups) */
void jam_q5k_repack8(const void* w, int rows0, int re, int sblocks, size_t w_stride, void* out);  /* repack 8 features (Q5_K) */
void jam_mm_q5k_rp_avx2(void* job, int rb, int re, int tid);
void jam_mm_q5k_rp1_avx2(void* job, int rb, int re, int tid);
void jam_q6k_repack8(const void* w, int rows0, int re, int sblocks, size_t w_stride, void* out);  /* repack 8 features (Q6_K) */
void jam_mm_q6k_rp_avx2(void* job, int rb, int re, int tid);
void jam_mm_q6k_rp1_avx2(void* job, int rb, int re, int tid);
void jam_q8_0_repack8(const void* w, int rows0, int re, int nblocks, size_t w_stride, void* out); /* repack 8 features (Q8_0) */
void jam_q4_0_repack8(const void* w, int rows0, int re, int nblocks, size_t w_stride, void* out); /* repack 8 features (Q4_0 -> raw u8 nibble) */
void jam_mm_q8_0_rp_avx2(void* job, int rb, int re, int tid);
void jam_mm_mxfp4_rp_avx2(void* job, int grp_begin, int grp_end, int tid);   /* cached-repack MXFP4 (int16-deferred) */
void jam_mm_mxfp4_rp1_avx2(void* job, int grp_begin, int grp_end, int tid);  /* n==1 sibling */
void jam_mxfp4_repack8(const void* w, int rows0, int re, int blocks, size_t w_stride, void* out);
            /* cached-repack Q8_0 gemm (sign-trick maddubs) */
void jam_mm_q8_0_rp1_avx2(void* job, int rb, int re, int tid);           /* cached-repack Q8_0 gemv/decode */
void jam_mm_q4_0_rp_avx2(void* job, int rb, int re, int tid);            /* cached-repack Q4_0 gemm (unsigned nibble, deferred madd, -8·Σa) */
void jam_mm_q4_0_rp1_avx2(void* job, int rb, int re, int tid);           /* cached-repack Q4_0 gemv/decode */
void jam_q8k_requant(void* job, int rb, int re, int tid);                 /* per-256 (Q8_K) activation requant, per-32 sums */
void jam_q6k_requant(void* job, int rb, int re, int tid);                 /* per-256 requant with per-16 sums (Q6_K bias) */
void jam_mm_nvfp4_avx2(void* job, int rb, int re, int tid);                /* NVFP4: FP4 LUT + per-16 E4M3 */
void jam_mm_q1_0_avx2(void* job, int rb, int re, int tid);                 /* Q1_0: sign-mask xor-negate maddubs */
#endif
#ifdef JAM_HAVE_AVXVNNI
void jam_mm_mxfp4_avxvnni(void* job, int a_begin, int a_end, int tid);     /* vpdpbusd + FP4 decode */
void jam_mm_q4_0_avxvnni(void* job, int a_begin, int a_end, int tid);      /* vpdpbusd + nibble-8 decode */
#endif

#ifdef JAM_HAVE_SSE3
void jam_mm_q8_0_sse3(void* job, int rb, int re, int tid);                 /* 128-bit sign-extend+madd (pre-AVX2 floor) */
void jam_mm_q4_0_sse3(void* job, int rb, int re, int tid);                 /* + arithmetic nibble decode */
void jam_mm_mxfp4_sse3(void* job, int rb, int re, int tid);               /* + scalar FP4-LUT decode (no pshufb) */
void jam_mm_q4k_sse3(void* job, int rb, int re, int tid);                 /* K-quant int8 dot (sign-extend+madd, SSE3 floor) */
void jam_mm_q5k_sse3(void* job, int rb, int re, int tid);
void jam_mm_q6k_sse3(void* job, int rb, int re, int tid);
#endif
#ifdef JAM_HAVE_SSSE3
void jam_mm_q8_0_ssse3(void* job, int rb, int re, int tid);               /* 128-bit maddubs sign-trick (Core 2 floor) */
void jam_mm_q4_0_ssse3(void* job, int rb, int re, int tid);              /* + arithmetic nibble decode */
#endif
#ifdef JAM_HAVE_AVX2
void jam_mm_q8_0_avx2(void* job, int a_begin, int a_end, int tid);         /* phase 2: maddubs matmul */
#endif
#ifdef JAM_HAVE_AVXVNNI
void jam_mm_q8_0_avxvnni(void* job, int a_begin, int a_end, int tid);      /* phase 2: 256-bit vpdpbusd */
void jam_mm_q4k_rp_avxvnni(void* job, int rb, int re, int tid);            /* cached-repack rp kernels, vpdpbusd dot */
void jam_mm_q5k_rp_avxvnni(void* job, int rb, int re, int tid);
void jam_mm_q6k_rp_avxvnni(void* job, int rb, int re, int tid);
void jam_mm_q8_0_rp_avxvnni(void* job, int rb, int re, int tid);
#endif
#ifdef JAM_HAVE_AVX512BW
void jam_mm_q8_0_avx512bw(void* job, int a_begin, int a_end, int tid);     /* phase 2: 512-bit maddubs (no VNNI) */
void jam_mm_nvfp4_avx512(void* job, int rb, int re, int tid);              /* NVFP4 512-bit (in the avx512bw TU) */
#endif
#ifdef JAM_HAVE_AVX512
void jam_mm_q8_0_avx512(void* job, int a_begin, int a_end, int tid);       /* phase 2: 512-bit VNNI matmul */
void jam_mm_q8_0_gemv_avx512(void* job, int row_begin, int row_end, int tid);     /* n==1 matvec (decode) + prefetch */
#endif
#ifdef JAM_HAVE_NEON
void jam_mm_q8_0_neon(void* job, int a_begin, int a_end, int tid);         /* vmull+vpadal (ARMv8 floor) */
void jam_mm_q4_0_neon(void* job, int rb, int re, int tid);                 /* + nibble decode */
void jam_mm_mxfp4_neon(void* job, int rb, int re, int tid);                /* + FP4 table-lookup decode */
void jam_mm_q4k_neon(void* job, int rb, int re, int tid);                 /* K-quant int8 dot (vmull+vpadal) */
void jam_mm_q5k_neon(void* job, int rb, int re, int tid);
void jam_mm_q6k_neon(void* job, int rb, int re, int tid);
void jam_mm_nvfp4_neon(void* job, int rb, int re, int tid);               /* NVFP4: FP4 LUT + per-16 E4M3 */
void jam_mm_q1_0_neon(void* job, int rb, int re, int tid);                /* Q1_0: b2b sign expand, vmull dot */
#endif
#ifdef JAM_HAVE_DOTPROD
void jam_mm_q8_0_dotprod(void* job, int a_begin, int a_end, int tid);      /* vdotq_s32 (sdot) */
void jam_mm_q4_0_dotprod(void* job, int rb, int re, int tid);
void jam_mm_nvfp4_dotprod(void* job, int rb, int re, int tid);            /* NVFP4 sdot */
void jam_mm_q1_0_dotprod(void* job, int rb, int re, int tid);             /* Q1_0 sdot */
void jam_mm_mxfp4_dotprod(void* job, int rb, int re, int tid);
void jam_mm_q4k_dotprod(void* job, int rb, int re, int tid);              /* K-quant sdot */
void jam_mm_q5k_dotprod(void* job, int rb, int re, int tid);
void jam_mm_q6k_dotprod(void* job, int rb, int re, int tid);
#endif
#ifdef JAM_HAVE_I8MM
void jam_mm_q8_0_i8mm(void* job, int a_begin, int a_end, int tid);         /* vmmlaq_s32 (smmla 2x2) */
#endif

/* ---- Metal GPU backend (Apple; opt-in via JAM_ISA=metal). A different executor, not a CPU row-range
 * kernel: jam_mm routes supported dtypes to it before the pool path. Implemented in jam_metal.mm. ---- */
#ifdef JAM_HAVE_METAL
typedef struct jam_metal jam_metal;
#ifdef __cplusplus
extern "C" {              /* jam_metal.mm is Objective-C++: match the extern "C" on its definitions */
#endif
jam_metal* jam_metal_create(void);
void       jam_metal_destroy(jam_metal* m);
jam_status jam_metal_mm(jam_metal* m, const void* a, jam_dtype at, int lda,
                        const void* b, jam_dtype bt, int ldb, void* c, jam_dtype ct, int ldc,
                        int M, int N, int K);
#ifdef __cplusplus
}
#endif
#endif

#endif /* JAM_INTERNAL_H */
