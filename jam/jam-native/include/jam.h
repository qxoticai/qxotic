/* jam.h - fast matrix multiplication with quantized weights, on the CPU or the Apple GPU. C99, one header.
 *
 *   Errors     A function returns its result; a negative value is a JAM_E* code, NULL for a pointer.
 *              jam_last_error has the message and the code, per thread, until the thread's next jam
 *              call (reading it changes nothing); after a success, "" and JAM_OK. Every call sets it but
 *              jam_version, jam_last_error, jam_isa_name, jam_pack_abi and a free of NULL. A goroutine
 *              or a Java virtual thread may change OS thread between two calls, so it reads the message
 *              pinned to the thread that failed or goes by the returned code, which is always the call's.
 *   Ownership  Operands are borrowed for the call, shared memory until jam_unshare. A context owns
 *              its scratch and its pool, and jam_context_free releases them. A string returned to you
 *              is the library's: jam_version's and jam_isa_name's for good, jam_last_error's until the
 *              calling thread's next jam call.
 *   Threads    Any function may be called from any thread, unless its comment says otherwise. A
 *              context is a serial stream: one jam_mm at a time, a concurrent one is JAM_EBUSY, and
 *              jam_context_free not during any call on it; jam_share and jam_unshare may run beside a
 *              jam_mm. For parallel matmuls, make several contexts.
 *   Fork       A context made before fork refuses jam_mm and jam_share in the child with JAM_ERROR, and
 *              jam_context_free there releases nothing, as its threads stayed in the parent; make a new
 *              one in the child, which stays on the CPU (Apple's GPU is unusable after fork without
 *              exec). The fork handler lives as long as the process, so once loaded libjam stays loaded.
 *   Settings   Through jam_context_options. The environment holds two diagnostic switches, read once
 *              as the library loads, on when set to anything but "", "0", "false" or "no": JAM_DEBUG
 *              prints the dispatch on stderr, and why a Metal setup or call fell back to the CPU;
 *              JAM_METAL_PROFILE prints Metal's per-call timings at jam_context_free and at exit. jam
 *              is otherwise silent.
 *   Work       One call, jam_mm: R = W @ A^T, k the contiguous axis of both W and A. W may be any
 *              type below and selects the kernel; A and R are float. jam multiplies quantized
 *              weights, it never converts them. The kernel level is chosen once per context, from
 *              what the CPU (or GPU) offers, below the context's cap. On the Apple GPU the block quants
 *              (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K) multiply in half precision with float sums, as llama.cpp's
 *              Metal does: an activation beyond +-65504 turns infinite there, where the CPU stays finite.
 *   Stability  Before 1.0 any release may break this API or its ABI, and the soname is the whole
 *              version: a program loads the release it was built against. From 1.0 a PATCH changes
 *              nothing a caller relies on, a MINOR only adds, a MAJOR may break, and every release is
 *              checked.
 */
#ifndef JAM_H
#define JAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(JAM_STATIC)
#  define JAM_API
#elif defined(_WIN32) && defined(JAM_BUILD)
#  define JAM_API __declspec(dllexport)
#elif defined(_WIN32)
#  define JAM_API __declspec(dllimport)
#elif defined(__GNUC__)
#  define JAM_API __attribute__((visibility("default")))
#else
#  define JAM_API
#endif

#define JAM_VERSION_MAJOR 0
#define JAM_VERSION_MINOR 4
#define JAM_VERSION_PATCH 0
/* The release as text: MAJOR.MINOR.PATCH, then -rcN on a release candidate; what jam_version() returns. */
#define JAM_VERSION "0.4.0-rc1"

/* status: the values jam, toknroll and jinfer share */
#define JAM_OK             0
#define JAM_ERROR        (-1)   /* see jam_last_error */
#define JAM_EINVAL       (-2)   /* a NULL, a shape or stride the type cannot have, an unknown flag bit */
#define JAM_EUNSUPPORTED (-4)   /* a valid request no kernel here serves, an unknown type included: fall back */
#define JAM_EBUSY        (-6)   /* another jam_mm is running on this context */
#define JAM_ENOMEM       (-7)

/* operand types: the values of GGML's ggml_type; PQ2_0 and PTQ1_0 are the PrismML fork's */
enum {
    JAM_F32 = 0, JAM_F16 = 1, JAM_BF16 = 30,
    JAM_Q4_0 = 2, JAM_Q5_0 = 6, JAM_Q8_0 = 8,
    JAM_Q4_K = 12, JAM_Q5_K = 13, JAM_Q6_K = 14,
    JAM_MXFP4 = 39, JAM_NVFP4 = 40, JAM_Q1_0 = 41, JAM_Q2_0 = 42,
    JAM_PQ2_0 = 142, JAM_PTQ1_0 = 143
};

/* kernel levels, ordered within each architecture */
enum {
    JAM_ISA_AUTO = 0,                                   /* in options: the best available */
    JAM_ISA_GENERIC = 1,                                /* portable C */
    JAM_ISA_SSE3 = 2, JAM_ISA_SSSE3 = 3, JAM_ISA_AVX2 = 4,
    JAM_ISA_AVX_VNNI = 5, JAM_ISA_AVX512 = 6, JAM_ISA_AVX512_VNNI = 7,
    JAM_ISA_NEON = 16, JAM_ISA_DOTPROD = 17, JAM_ISA_I8MM = 18,
    JAM_ISA_METAL = 19                                  /* the Apple GPU, chosen by AUTO; cap below it for the CPU */
};

JAM_API const char* jam_version(void);               /* JAM_VERSION, the release it was built from */
JAM_API const char* jam_last_error(int32_t* code);   /* the message; *code = the code unless NULL */
JAM_API const char* jam_isa_name(int32_t isa);       /* "avx512_vnni"; "" for an unknown level */

/* ---- threads: jam runs its work on the host's loop, or on a small pool of its own ---- */

/* One slice of the work, [begin, end), on participant slot. */
typedef void (*jam_task_fn)(int32_t begin, int32_t end, int32_t slot, void* task_data);
/* Runs task over [0, count) in slices and returns when all are done. Each slice's slot is below
 * the context's threads and unique among the slices running at once. jam calls it only from the
 * thread inside jam_mm; it must return normally and must not call jam. A slot at or past threads
 * fails that jam_mm with JAM_EINVAL. */
typedef void (*jam_parallel_fn)(int32_t count, jam_task_fn task, void* task_data, void* user_data);

#define JAM_MAX_THREADS 1024

typedef struct {                    /* zero = default in every field; NULL options = all defaults */
    jam_parallel_fn parallel;       /* NULL: the context runs a pool of its own */
    void*           parallel_user;  /* passed back to parallel */
    int32_t         threads;        /* participants, at most JAM_MAX_THREADS; 0 = every online CPU */
    int32_t         max_isa;        /* a cap on this architecture's levels; JAM_ISA_AUTO = no cap */
} jam_context_options;

typedef struct jam_context jam_context;

/* NULL on failure, jam_last_error saying why: JAM_EINVAL for a cap of another architecture or threads
 * out of range, JAM_ENOMEM. */
JAM_API jam_context* jam_context_new(const jam_context_options* options);
/* Releases the context; NULL does nothing, and so does a context made before fork (Fork). */
JAM_API void         jam_context_free(jam_context* context);
/* The level in use; JAM_EINVAL for NULL. */
JAM_API int32_t      jam_context_isa(const jam_context* context);

/* ---- the work: R = W @ A^T ----
 *   W [m x k], type wt, row stride ldw   the weights, any type above (wt | JAM_PACKED: packed)
 *   A [n x k], type at, row stride lda   the activations, JAM_F32
 *   R [n x m], type rt, row stride ldr   the result, JAM_F32, token-major: R[j*ldr + i] = W[i,:] . A[j,:]
 * m, n and k are positive; strides count elements, ldw and lda at least k, ldr at least m. A block
 * type takes k and ldw in whole blocks: 32 elements, NVFP4 64, Q1_0 128, Q4_K, Q5_K and Q6_K 256.
 * Anything else is JAM_EINVAL. JAM_EUNSUPPORTED: a valid call no kernel here serves, such as a type,
 * at or rt jam has no kernel for, or packed weights on a context whose jam_pack_size is 0. */
JAM_API int32_t jam_mm(jam_context* context,
                       const void* w, int32_t wt, int32_t ldw,
                       const void* a, int32_t at, int32_t lda,
                       void*       r, int32_t rt, int32_t ldr,
                       int32_t m, int32_t n, int32_t k);

/* ---- packed weights: an in-memory layout, never a file format ----
 * jam_pack_size gives the bytes of the packed form a context wants for a [m x k] weight; the caller
 * writes that layout (JAM_PACK_ABI, described below) once at load, drops the canonical
 * bytes, and passes wt | JAM_PACKED with ldw == k. Values are exactly the canonical dequant. */
#define JAM_PACKED   0x100   /* OR'd into wt: W holds the packed layout of (wt & ~JAM_PACKED) */
#define JAM_PACK_ABI 1

/*
 * Layout (JAM_PACK_ABI 1): rows in groups of 4 (m % 4 == 0 required); group g starts at
 * base + g * GB, its sections in order (nb = k/32 blocks, sb = k/256 super-blocks; within a
 * section the k-block index is the outer stride, the 4 rows r interleave inside):
 *
 *   Q4_0  GB = 80*nb           [nb x 64B nibble lines: 4 rows x 16B, elem e in the low nibble of
 *                               byte e, elem e+16 in the high nibble (Q4_0 order)]
 *                              [nb x 16B f32 scale[4], lane = row]
 *   MXFP4 GB = 68*nb           [nb x 4B E8M0 scale[4] | 64B nibbles: 4-byte chunks interleaved
 *                               across 4 rows, enabling lane-wise SDOT]
 *   Q4_K  GB = 72*nb + 32*sb   [nb x 64B re-nibbled lines, Q4_0 order]
 *                              [nb x 8B u8 sc[4] | mn[4] (6-bit values, decoded)]
 *                              [sb x 32B f32 d[4] | dmin[4]]
 *   Q5_K  GB = 136*nb + 32*sb  [nb x 128B int8: 4 rows x 32, q5 in 0..31]
 *                              [nb x 8B u8 sc[4] | mn[4]]  [sb x 32B f32 d[4] | dmin[4]]
 *   Q6_K  GB = 136*nb + 16*sb  [nb x 128B int8: 4 rows x 32, value q6 - 32]
 *                              [nb x 8B i8 sc: first-half sc[4] | second-half sc[4] (per-16)]
 *                              [sb x 16B f32 d[4]]
 */

JAM_API int32_t jam_pack_abi(void);   /* the JAM_PACK_ABI this library reads */
/* Bytes of the packed form; 0 = keep the canonical bytes: a type, shape or CPU this cannot pack,
 * a negative m or k included. JAM_EINVAL for a NULL context. */
JAM_API int64_t jam_pack_size(const jam_context* context, int32_t wt, int32_t m, int32_t k);

/* ---- sharing: a hint for GPU backends; results never depend on it ---- */
#define JAM_SHARE_IMMUTABLE 0x1u   /* the bytes never change while shared */

/* [p, p + bytes) goes to many jam_mm calls, so a device maps it once; bytes > 0. JAM_EUNSUPPORTED:
 * ignored (a CPU context). Shares on one context must not overlap (JAM_EINVAL). Unshare before
 * freeing the memory. Both may run during a jam_mm on the context. */
JAM_API int32_t jam_share(jam_context* context, const void* p, int64_t bytes, uint32_t flags);
/* Ends the share starting at p, if any; a NULL context is JAM_EINVAL in jam_last_error. */
JAM_API void    jam_unshare(jam_context* context, const void* p);

#ifdef __cplusplus
}
#endif
#endif /* JAM_H */
