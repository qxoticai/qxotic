#ifndef JAM_H
#define JAM_H

/* jam — the fastest multithreaded CPU matmul. One job: C = A @ Bᵀ, with quantized weights.
 *
 *   op       : ONE call, jam_mm. C = A @ Bᵀ (ggml mul_mat convention — k is the CONTIGUOUS last axis
 *              of both operands, so each output is a dot of two contiguous k-vectors). n==1 (B a
 *              single row — decode) -> dedicated streaming gemv; else register-tiled gemm.
 *   dtypes   : values mirror GGML's ggml_type, so a GGUF loader passes a tensor's type straight
 *              through. A (weight) may be any format and selects the kernel; B and C are float. jam
 *              MULTIPLIES quantized weights (decode + requant happen in the kernel); it never
 *              CONVERTS formats — quantization is done in the host (Java).
 *   context  : jam_mm(NULL, ...) uses the process-global context — lazily built on first use from
 *              JAM_* env vars (OpenMP-style; zero setup). Or pass an explicit context for fine
 *              control (host executor, thread count, ISA cap).
 *   threading: jam_mm is multithreaded WITHIN a call (the context's pool fans the work over its
 *              threads). A jam_ctx is a SERIAL EXECUTION STREAM: mm calls on the SAME context run one
 *              at a time and must NOT be called concurrently (they share the pool and the context's
 *              scratch). For PARALLEL matmul, create and use SEVERAL contexts — one per thread, each
 *              owning its own pool + scratch. The global context is a single serial stream.
 *   dispatch : the per-CPU kernel is resolved ONCE (per context); the op is a table lookup + a few
 *              cheap branches. No per-call search/hashing/codegen.
 *   memory   : operands are caller-owned, BORROWED for the call; a context owns its pool and (for the
 *              quantized paths) a lazily-grown activation-requant scratch — both per-context, which is
 *              why a context is single-stream.
 *   targets  : x86 (SSE2..AVX-512/VNNI/AMX), ARM (NEON..SVE), and a portable-C floor — one fat lib,
 *              best kernel chosen at runtime; builds and runs anywhere. No dependencies.
 *
 *   == Environment (configure the GLOBAL context — read ONCE, lazily, on first jam_mm(NULL,...)) ==
 *     JAM_NUM_THREADS   pool size           (unset or 0 = physical cores)
 *     JAM_ISA           cap the kernel ISA  (a capability name below; unset = best available)
 *   An explicit context (jam_ctx_create) ignores the environment and uses its jam_config.
 *
 *   == Capability names (the user-facing strings for JAM_ISA and jam_isa_name) ==
 *     "auto" "generic" "sse2" "avx2" "avx512" "avx512_vnni" "amx"      (x86)
 *     "neon" "dotprod" "i8mm" "sve"                                    (arm)
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {            /* for C++ CALLERS only — the API is pure C11 */
#endif

#define JAM_ABI_VERSION 1u   /* feeds the soname (libjam.so.1); the loader enforces ABI match */

/* dtype tags — numerically identical to GGML's ggml_type (drop-in for GGUF). A may be any of these;
 * B and C are float (F32/F16/BF16). */
typedef enum {
    JAM_F32   = 0,
    JAM_F16   = 1,
    JAM_Q4_0  = 2,
    JAM_Q4_1  = 3,
    JAM_Q5_0  = 6,
    JAM_Q5_1  = 7,
    JAM_Q8_0  = 8,
    JAM_Q2_K  = 10,
    JAM_Q3_K  = 11,
    JAM_Q4_K  = 12,
    JAM_Q5_K  = 13,
    JAM_Q6_K  = 14,
    JAM_Q8_K  = 15,
    JAM_BF16  = 30,
    JAM_MXFP4 = 39,   /* == GGML_TYPE_MXFP4 — verify the value against the ggml.h you target */
    JAM_NVFP4 = 40,   /* == GGML_TYPE_NVFP4 (llama.cpp): {d[4] UE4M3; qs[32]} interleaved, 64-elem, no global */
} jam_dtype;

typedef enum {
    JAM_OK = 0,
    JAM_EINVAL,        /* null args, mismatched/invalid shapes or dtypes */
    JAM_EUNSUPPORTED,  /* no enabled kernel for this dtype combo on this CPU */
    JAM_EBUSY,         /* another mm is in flight on THIS context (serial stream) — retry or fall back */
} jam_status;

/* ISA capability — an ORDERED level (each implies all below it on its arch). Used both to CAP the
 * engine (jam_config.max_isa) and to REPORT what's live (jam_active_isa). Names via jam_isa_name. */
typedef enum {
    JAM_ISA_AUTO = 0,          /* config only: pick the best available */
    JAM_ISA_GENERIC,           /* portable scalar floor */
    JAM_ISA_SSE2,
    JAM_ISA_SSE3,              /* 128-bit int8 (sign-extend + madd, software fp16); pre-AVX2 x86 floor */
    JAM_ISA_AVX2,
    JAM_ISA_AVX_VNNI,          /* 256-bit VNNI without AVX-512 (client: Alder/Raptor Lake) */
    JAM_ISA_AVX512,
    JAM_ISA_AVX512_VNNI,
    JAM_ISA_AMX,
    /* arm */
    JAM_ISA_NEON,
    JAM_ISA_DOTPROD,
    JAM_ISA_I8MM,
    JAM_ISA_SVE,
    /* GPU backend (not a CPU ISA) — opt-in via JAM_ISA=metal / max_isa; routes matmul to the Apple GPU. */
    JAM_ISA_METAL,
} jam_isa;

const char* jam_isa_name(jam_isa isa);   /* user-facing name, e.g. "avx512_vnni"; static, do not free */

/* ---- threading: drive the caller's executor (e.g. the JVM SpinPool), or let jam own a pool ----
 * jam calls parallel_for(pool, n, fn, arg): run fn over [0,n) split across workers, block till done. */
typedef void (*jam_task_fn)(void* arg, int begin, int end, int tid);
typedef void (*jam_parallel_for)(void* pool, int n, jam_task_fn fn, void* arg);

/* ---- context (optional; ctx==NULL uses the global env-configured one) ---- */
typedef struct {
    jam_parallel_for parallel_for;   /* NULL -> the context owns an internal pool */
    void*            pool;            /* opaque, passed back to parallel_for */
    int32_t          nthreads;        /* internal-pool size; 0 = auto (physical cores) */
    jam_isa          max_isa;         /* JAM_ISA_AUTO = best; else cap here (disables higher levels) */
    const char*      name;            /* optional label for JAM_DEBUG logs (copied; NULL = unnamed) */
} jam_config;

typedef struct jam_ctx jam_ctx;
jam_ctx* jam_ctx_create(const jam_config* cfg);   /* NULL on failure */
void     jam_ctx_destroy(jam_ctx* ctx);

/* ---- the work ----
 * C = W @ Aᵀ   (k is the contiguous last axis of both W and A)
 *     W [m×k] dtype `wt`, row stride ldw   — WEIGHTS, may be quantized (selects the kernel)
 *     A [n×k] dtype `at`, row stride lda   — ACTIVATIONS, float
 *     C       dtype `ct`, row stride ldc   — OUTPUT, TOKEN-major (ggml/llama.cpp layout):
 *                                            C[j*ldc + i] = dot(W[i,:], A[j,:]), i.e. each token j's
 *                                            m-feature vector is contiguous (feature i unit-stride), ldc >= m.
 * Strides in ELEMENTS (k is unit-stride). Pointers borrowed for the call. ctx==NULL -> global.
 * n==1 is the gemv (decode) case. Unhandled dtype combo -> JAM_EUNSUPPORTED; concurrent mm on one ctx -> JAM_EBUSY. */
jam_status jam_mm(jam_ctx* ctx,
                  const void* w, jam_dtype wt, int ldw,   /* weights     [m × k] */
                  const void* a, jam_dtype at, int lda,   /* activations [n × k] */
                  void*       c, jam_dtype ct, int ldc,   /* output      [m × n] */
                  int m, int n, int k);

jam_isa     jam_active_isa(const jam_ctx* ctx);   /* the live kernel level; ctx==NULL -> global */
const char* jam_ctx_name(const jam_ctx* ctx);    /* the context's label ("" if unnamed); ctx==NULL -> global */

/* Drop the internal repacked-weight cache entry for `w` (ctx==NULL -> global). The quant fast path repacks
 * each weight once and caches it keyed on the pointer, reused for the ctx lifetime. Call this BEFORE freeing
 * or overwriting a weight whose address may be reused, else a new weight at that address hits the stale
 * repack. No-op if `w` was never cached. Not safe to call concurrently with jam_mm on the same context. */
void jam_forget_weight(jam_ctx* ctx, const void* w);

/* Destroy the process-global context (the one jam_mm(NULL,...) uses) and free its pool + scratch. A no-op
 * if it was never created; a later jam_mm(NULL,...) lazily re-creates it (idempotent). Most callers never
 * need this — the global is a reachable singleton, not a leak. Call it for a clean teardown: a plugin host
 * BEFORE dlclose (else each load/unload accumulates one), or a JVM host from a shutdown hook. It is NOT run
 * automatically (joining the pool threads from a library destructor is unsafe during VM teardown). NOT safe
 * to call concurrently with any jam_mm(NULL,...). */
void jam_global_destroy(void);

#ifdef __cplusplus
}
#endif
#endif /* JAM_H */
