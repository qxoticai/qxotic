// Portable native GEMM entry points, bound by
// com.llama4j.NativeKernels (-Dllama.nativeGemmLib=<path> on the JVM, or statically
// linked into a native image with -Dllama.staticGemm=true).
//
// The file must compile on every platform. Fast kernels are enabled by capability bits; platforms
// without a native implementation still export the JNI symbols and Java falls back to Vector API.
//
// Current fast backends:
//   - x86_64 AVX-512/VNNI: Q8_0, Q4_K, Q6_K (thread-pool, repack)
//   - x86_64 AVX2/FMA/F16C:  Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, BF16, F16, F32
//   - ARM64 NEON:            Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, BF16, F16, F32 + GEMV
//   - Pure C:                all quant types (baseline fallback)
//
// Separately: gemm.metal provides Metal compute kernels for Apple Silicon (GPU).
//
// Build:
//   make libnative   (gcc -O3 -march=native -shared -fPIC -pthread)
//   make arm64-so    (aarch64-linux-gnu-gcc for ARM cross-compile)
//   make metal-lib   (macOS only: xcrun metal -> gemm.metallib)

#define _GNU_SOURCE

#include <jni.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#define LFM25_X86_64 1
#else
#define LFM25_X86_64 0
#endif
#if !defined(_WIN32)
#include <pthread.h>
#include <unistd.h>
#define LFM25_POSIX_THREADS 1
#else
#define LFM25_POSIX_THREADS 0
#endif
#if defined(__linux__)
#include <sched.h>
#endif
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#if LFM25_POSIX_THREADS && LFM25_X86_64 && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__F16C__)
#define LFM25_AVX512_VNNI 1
#else
#define LFM25_AVX512_VNNI 0
#endif

#if LFM25_X86_64 && defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define LFM25_AVX2_FMA 1
#else
#define LFM25_AVX2_FMA 0
#endif

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define LFM25_ARM64 1
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define LFM25_ARM_FP16 1
#else
#define LFM25_ARM_FP16 0
#endif
#else
#define LFM25_ARM64 0
#define LFM25_ARM_FP16 0
#endif
// NEON is mandatory on ARMv8+: always enabled on AArch64.
#if LFM25_ARM64
#define LFM25_ARM_NEON 1
#else
#define LFM25_ARM_NEON 0
#endif

#define LFM25_CAP_Q8_0_GEMM 1
#define LFM25_CAP_Q8_0_GEMV 2
#define LFM25_CAP_Q4_K_GEMM 4
#define LFM25_CAP_Q6_K_GEMM 8
#define LFM25_CAP_Q4_0_GEMM 16
#define LFM25_CAP_Q5_K_GEMM 32
#define LFM25_CAP_BF16_GEMM 64
#define LFM25_CAP_F16_GEMM 128
#define LFM25_CAP_F32_GEMM 256

#if LFM25_X86_64 && (defined(__GNUC__) || defined(__clang__))
static int x86_supports_avx512_vnni(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
    if ((ecx & bit_OSXSAVE) == 0) {
        return 0;
    }
    unsigned int xcr0_lo, xcr0_hi;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    unsigned long long xcr0 = ((unsigned long long) xcr0_hi << 32) | xcr0_lo;
    const unsigned long long zmm_state = (1ULL << 1) | (1ULL << 2) | (1ULL << 5) | (1ULL << 6) | (1ULL << 7);
    if ((xcr0 & zmm_state) != zmm_state) {
        return 0;
    }
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
    return (ebx & bit_AVX512F) && (ebx & bit_AVX512BW) && (ecx & bit_AVX512VNNI);
}

static int x86_supports_avx2_fma_f16c(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
    if ((ecx & bit_OSXSAVE) == 0 || (ecx & bit_AVX) == 0 || (ecx & bit_FMA) == 0 || (ecx & bit_F16C) == 0) {
        return 0;
    }
    unsigned int xcr0_lo, xcr0_hi;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    unsigned long long xcr0 = ((unsigned long long) xcr0_hi << 32) | xcr0_lo;
    const unsigned long long ymm_state = (1ULL << 1) | (1ULL << 2);
    if ((xcr0 & ymm_state) != ymm_state) {
        return 0;
    }
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
    return (ebx & bit_AVX2) != 0;
}
#else
static int x86_supports_avx512_vnni(void) {
    return 0;
}

static int x86_supports_avx2_fma_f16c(void) {
    return 0;
}
#endif

#define QK 32           // elements per Q8_0 block
#define BLOCK_BYTES 34  // 2-byte fp16 scale + 32 int8 quants

#define QKK 256             // elements per K-quant super-block
#define Q4K_BLOCK_BYTES 144 // d(f16) dmin(f16) scales[12] qs[128]
#define Q5K_BLOCK_BYTES 176 // d(f16) dmin(f16) scales[12] qh[32] qs[128]
#define Q6K_BLOCK_BYTES 210 // ql[128] qh[64] scales[16] d(f16)

#define GEMV_ROW_CHUNK 64
#define VNNI_BAND 32        // weight rows per parallel work unit (4 groups of 16)
#define VNNI_MIN_SEQ 8      // below this, activation quantization + repack don't amortize

static float c_fp16_to_f32(uint16_t h) {
    int sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    int mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) {
            union { uint32_t i; float f; } u = {(uint32_t) sign};
            return u.f;
        }
        int e = 127 - 15 + 1;
        while ((mant & 0x0400) == 0) {
            mant <<= 1;
            e--;
        }
        mant &= 0x03FF;
        union { uint32_t i; float f; } u = {(uint32_t) (sign | (e << 23) | (mant << 13))};
        return u.f;
    }
    if (exp == 0x1F) {
        union { uint32_t i; float f; } u = {(uint32_t) (sign | 0x7F800000 | (mant << 13))};
        return u.f;
    }
    union { uint32_t i; float f; } u = {(uint32_t) (sign | ((exp + (127 - 15)) << 23) | (mant << 13))};
    return u.f;
}

static float c_q4_0_dot(const uint8_t *w, const float *x, int kblocks) {
    float sum = 0.0f;
    for (int b = 0; b < kblocks; b++, w += 18, x += QK) {
        float d = c_fp16_to_f32(*(const uint16_t *) w);
        const uint8_t *qs = w + 2;
        for (int i = 0; i < 16; i++) {
            int packed = qs[i];
            sum += (float) ((packed & 0x0F) - 8) * d * x[i];
            sum += (float) (((packed >> 4) & 0x0F) - 8) * d * x[i + 16];
        }
    }
    return sum;
}

static void c_gemm_q4_0(jlong weights, jlong x, jlong x_stride_bytes,
                        jlong out, jlong out_stride_bytes,
                        jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * 18;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = c_q4_0_dot(wbase + (int64_t) row * w_stride, row_x, kblocks);
        }
    }
}

static float c_q8_0_dot(const uint8_t *w, const float *x, int kblocks) {
    float sum = 0.0f;
    for (int b = 0; b < kblocks; b++, w += BLOCK_BYTES, x += QK) {
        float d = c_fp16_to_f32(*(const uint16_t *) w);
        const int8_t *qs = (const int8_t *) (w + 2);
        for (int i = 0; i < QK; i++) {
            sum += (float) qs[i] * d * x[i];
        }
    }
    return sum;
}

static void c_gemm_q8_0(jlong weights, jlong x, jlong x_stride_bytes,
                        jlong out, jlong out_stride_bytes,
                        jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * BLOCK_BYTES;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = c_q8_0_dot(wbase + (int64_t) row * w_stride, row_x, kblocks);
        }
    }
}

static int c_q4k_scale_min(const uint8_t *scales, int j, int is_min) {
    if (j < 4) {
        return scales[is_min ? j + 4 : j] & 63;
    }
    int low = is_min ? (scales[j + 4] >> 4) : (scales[j + 4] & 0xF);
    int high = (scales[is_min ? j : j - 4] >> 6) & 0x3;
    return low | (high << 4);
}

static float c_q4k_dot(const uint8_t *w, const float *x, int blocks) {
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++, w += Q4K_BLOCK_BYTES, x += QKK) {
        float d = c_fp16_to_f32(*(const uint16_t *) w);
        float dmin = c_fp16_to_f32(*(const uint16_t *) (w + 2));
        const uint8_t *scales = w + 4;
        const uint8_t *qs = w + 16;
        for (int g = 0; g < 4; g++) {
            int sc_lo = c_q4k_scale_min(scales, g * 2, 0);
            int mn_lo = c_q4k_scale_min(scales, g * 2, 1);
            int sc_hi = c_q4k_scale_min(scales, g * 2 + 1, 0);
            int mn_hi = c_q4k_scale_min(scales, g * 2 + 1, 1);
            const uint8_t *q = qs + g * 32;
            const float *xlo = x + g * 64;
            const float *xhi = xlo + 32;
            for (int i = 0; i < 32; i++) {
                int packed = q[i];
                sum += (d * sc_lo * (float) (packed & 0x0F) - dmin * mn_lo) * xlo[i];
                sum += (d * sc_hi * (float) ((packed >> 4) & 0x0F) - dmin * mn_hi) * xhi[i];
            }
        }
    }
    return sum;
}

static float c_q5k_dot(const uint8_t *w, const float *x, int blocks) {
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++, w += Q5K_BLOCK_BYTES, x += QKK) {
        float d = c_fp16_to_f32(*(const uint16_t *) w);
        float dmin = c_fp16_to_f32(*(const uint16_t *) (w + 2));
        const uint8_t *scales = w + 4;
        const uint8_t *qh = w + 16;
        const uint8_t *qs = w + 48;
        for (int g = 0; g < 4; g++) {
            int sc_lo = c_q4k_scale_min(scales, g * 2, 0);
            int mn_lo = c_q4k_scale_min(scales, g * 2, 1);
            int sc_hi = c_q4k_scale_min(scales, g * 2 + 1, 0);
            int mn_hi = c_q4k_scale_min(scales, g * 2 + 1, 1);
            const uint8_t *q = qs + g * 32;
            const float *xlo = x + g * 64;
            const float *xhi = xlo + 32;
            for (int i = 0; i < 32; i++) {
                int packed = q[i];
                int hi_bits = qh[i];
                int qlo = (packed & 0x0F) | (((hi_bits >> (2 * g)) & 1) << 4);
                int qhi = ((packed >> 4) & 0x0F) | (((hi_bits >> (2 * g + 1)) & 1) << 4);
                sum += (d * sc_lo * (float) qlo - dmin * mn_lo) * xlo[i];
                sum += (d * sc_hi * (float) qhi - dmin * mn_hi) * xhi[i];
            }
        }
    }
    return sum;
}

static float c_q6k_dot(const uint8_t *w, const float *x, int blocks) {
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++, w += Q6K_BLOCK_BYTES, x += QKK) {
        float d = c_fp16_to_f32(*(const uint16_t *) (w + 208));
        const uint8_t *ql = w;
        const uint8_t *qh = w + 128;
        const int8_t *scales = (const int8_t *) (w + 192);
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_base = ql + half * 64;
            const uint8_t *qh_base = qh + half * 32;
            for (int sub = 0; sub < 4; sub++) {
                int scale = scales[half * 8 + sub * 2];
                int scale2 = scales[half * 8 + sub * 2 + 1];
                for (int i = 0; i < 16; i++) {
                    int idx = sub * 32 + i;
                    int qh_byte = qh_base[i];
                    int qh_byte2 = qh_base[i + 16];
                    int q0, q1;
                    switch (sub) {
                        case 0:
                            q0 = (ql_base[i] & 0x0F) | ((qh_byte & 0x03) << 4);
                            q1 = (ql_base[i + 16] & 0x0F) | ((qh_byte2 & 0x03) << 4);
                            break;
                        case 1:
                            q0 = (ql_base[32 + i] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4);
                            q1 = (ql_base[48 + i] & 0x0F) | (((qh_byte2 >> 2) & 0x03) << 4);
                            break;
                        case 2:
                            q0 = ((ql_base[i] >> 4) & 0x0F) | (((qh_byte >> 4) & 0x03) << 4);
                            q1 = ((ql_base[i + 16] >> 4) & 0x0F) | (((qh_byte2 >> 4) & 0x03) << 4);
                            break;
                        default:
                            q0 = ((ql_base[32 + i] >> 4) & 0x0F) | (((qh_byte >> 6) & 0x03) << 4);
                            q1 = ((ql_base[48 + i] >> 4) & 0x0F) | (((qh_byte2 >> 6) & 0x03) << 4);
                            break;
                    }
                    sum += d * (float) scale * (float) (q0 - 32) * x[half * 128 + idx];
                    sum += d * (float) scale2 * (float) (q1 - 32) * x[half * 128 + idx + 16];
                }
            }
        }
    }
    return sum;
}

static void c_gemm_kquant(float (*dot)(const uint8_t *, const float *, int), int block_bytes,
                          jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int blocks = dim1 / QKK;
    int64_t w_stride = (int64_t) blocks * block_bytes;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = dot(wbase + (int64_t) row * w_stride, row_x, blocks);
        }
    }
}

static float c_bf16_to_f32(uint16_t h) {
    union { uint32_t i; float f; } u = {(uint32_t) h << 16};
    return u.f;
}

static float c_f32_dot(const uint8_t *w, const float *x, int dim1) {
    const float *wf = (const float *) w;
    float sum = 0.0f;
    for (int i = 0; i < dim1; i++) sum += wf[i] * x[i];
    return sum;
}

static float c_f16_dot(const uint8_t *w, const float *x, int dim1) {
    const uint16_t *wh = (const uint16_t *) w;
    float sum = 0.0f;
    for (int i = 0; i < dim1; i++) sum += c_fp16_to_f32(wh[i]) * x[i];
    return sum;
}

static float c_bf16_dot(const uint8_t *w, const float *x, int dim1) {
    const uint16_t *wh = (const uint16_t *) w;
    float sum = 0.0f;
    for (int i = 0; i < dim1; i++) sum += c_bf16_to_f32(wh[i]) * x[i];
    return sum;
}

static void c_gemm_dense(float (*dot)(const uint8_t *, const float *, int), int elem_bytes,
                         jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
                         jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int64_t w_stride = (int64_t) dim1 * elem_bytes;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = dot(wbase + (int64_t) row * w_stride, row_x, dim1);
        }
    }
}

#if LFM25_AVX512_VNNI

typedef struct {
    int kind; // 0 = f32 gemm, 1 = gemv, 2 = quantize activations (u8), 3 = vnni gemm,
              // 4 = quantize activations (s8 + sums), 5 = q4k vnni gemm, 6 = q6k vnni gemm
    const uint8_t *weights;
    const float *rhs;
    float *out;
    int that_stride;
    int out_stride;
    int sequence_length;
    int dim0;
    int dim1;
    int row_tile;
    int seq_tile;
    int tile_count;
    int seq_tile_count;
    int dtype; // for kind 7 (dense gemm): 0 = f32, 1 = f16, 2 = bf16 weights
} gemm_task_t;

// Activations quantized to Q8 (u8, biased +128 so they can be the unsigned vpdpbusd
// operand) once per gemm: xq[s][kblocks][32], dx[s][kblocks].
// K-quant gemms quantize to PLAIN s8 instead (their weight nibbles are the unsigned
// operand) into the same buffers, plus exact f32 sums per 16 elements in xsum_buf
// (Q4_K min terms and the Q6_K -32 offset never enter the integer dot).
static uint8_t *xq_buf = NULL;
static float *dx_buf = NULL;
static float *xsum_buf = NULL;
static size_t xq_cap = 0;
static size_t dx_cap = 0;
static size_t xsum_cap = 0;

// Per-worker repack scratch: one 64-row weight band in vnni layout.
// Per 16-row group g and block b: qs[((g*kblocks)+b)*512 + j*64] holds rows' 4-byte
// K-groups side by side (row r at offset r*4), so one vpdpbusd accumulates 16 rows in
// the 16 i32 lanes. dw = per-row fp32 scales, cw = d*128*sum(quants) bias correction
// (the +128 activation bias contributes 128*sum(w) per block, folded here).
typedef struct {
    uint8_t *qs;
    float *dw;
    float *cw;
    int cap_blocks;
} repack_t;
static repack_t *repack_scratch = NULL; // pool_size + 1 entries (last = dispatching thread)

static pthread_mutex_t pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pool_start = PTHREAD_COND_INITIALIZER;
static pthread_t *pool_threads = NULL;
static int pool_size = 0;
static int pool_stop = 0;
static gemm_task_t current_task;
static atomic_int next_tile;
// Spinning dispatch: decode issues hundreds of sub-100us gemvs per token, so a
// mutex+condvar round trip per call dominates. Workers spin briefly on the generation
// counter before sleeping; the caller participates in the work and only wakes sleepers.
static atomic_int pool_generation;
static atomic_int pool_active;
static atomic_int pool_sleepers;
static int pool_spin = 500; // _mm_pause iterations before a worker sleeps; bigger values starve Java threads

static inline float fp16_to_f32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

static inline __m512 q8_half_block(const uint8_t *q, __m512 scale) {
    __m512i i32 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) q));
    return _mm512_mul_ps(_mm512_cvtepi32_ps(i32), scale);
}

static inline float hsum512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

// One output: row x col over the full K range.
static void tile_1x1(const uint8_t *w, const float *x, int kblocks, float *out) {
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    for (int b = 0; b < kblocks; b++, w += BLOCK_BYTES, x += QK) {
        __m512 s = _mm512_set1_ps(fp16_to_f32(*(const uint16_t *) w));
        c0 = _mm512_fmadd_ps(q8_half_block(w + 2, s), _mm512_loadu_ps(x), c0);
        c1 = _mm512_fmadd_ps(q8_half_block(w + 18, s), _mm512_loadu_ps(x + 16), c1);
    }
    *out = hsum512(_mm512_add_ps(c0, c1));
}

// 4 rows x 4 cols register tile over the full K range.
// out[c * out_stride + r] for r,c in 0..3.
static void tile_4x4(const uint8_t *w, int64_t w_stride,
                     const float *x, int64_t x_stride,
                     int kblocks, float *out, int64_t out_stride) {
    const uint8_t *w0 = w, *w1 = w + w_stride, *w2 = w + 2 * w_stride, *w3 = w + 3 * w_stride;
    const float *x0 = x, *x1 = x + x_stride, *x2 = x + 2 * x_stride, *x3 = x + 3 * x_stride;
    __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps(), c02 = _mm512_setzero_ps(), c03 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps(), c13 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps(), c23 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps(), c32 = _mm512_setzero_ps(), c33 = _mm512_setzero_ps();
    for (int b = 0; b < kblocks; b++) {
        __m512 s0 = _mm512_set1_ps(fp16_to_f32(*(const uint16_t *) w0));
        __m512 s1 = _mm512_set1_ps(fp16_to_f32(*(const uint16_t *) w1));
        __m512 s2 = _mm512_set1_ps(fp16_to_f32(*(const uint16_t *) w2));
        __m512 s3 = _mm512_set1_ps(fp16_to_f32(*(const uint16_t *) w3));
        __m512 w0l = q8_half_block(w0 + 2, s0), w0h = q8_half_block(w0 + 18, s0);
        __m512 w1l = q8_half_block(w1 + 2, s1), w1h = q8_half_block(w1 + 18, s1);
        __m512 w2l = q8_half_block(w2 + 2, s2), w2h = q8_half_block(w2 + 18, s2);
        __m512 w3l = q8_half_block(w3 + 2, s3), w3h = q8_half_block(w3 + 18, s3);

        __m512 al = _mm512_loadu_ps(x0), ah = _mm512_loadu_ps(x0 + 16);
        c00 = _mm512_fmadd_ps(w0l, al, c00); c00 = _mm512_fmadd_ps(w0h, ah, c00);
        c10 = _mm512_fmadd_ps(w1l, al, c10); c10 = _mm512_fmadd_ps(w1h, ah, c10);
        c20 = _mm512_fmadd_ps(w2l, al, c20); c20 = _mm512_fmadd_ps(w2h, ah, c20);
        c30 = _mm512_fmadd_ps(w3l, al, c30); c30 = _mm512_fmadd_ps(w3h, ah, c30);
        al = _mm512_loadu_ps(x1); ah = _mm512_loadu_ps(x1 + 16);
        c01 = _mm512_fmadd_ps(w0l, al, c01); c01 = _mm512_fmadd_ps(w0h, ah, c01);
        c11 = _mm512_fmadd_ps(w1l, al, c11); c11 = _mm512_fmadd_ps(w1h, ah, c11);
        c21 = _mm512_fmadd_ps(w2l, al, c21); c21 = _mm512_fmadd_ps(w2h, ah, c21);
        c31 = _mm512_fmadd_ps(w3l, al, c31); c31 = _mm512_fmadd_ps(w3h, ah, c31);
        al = _mm512_loadu_ps(x2); ah = _mm512_loadu_ps(x2 + 16);
        c02 = _mm512_fmadd_ps(w0l, al, c02); c02 = _mm512_fmadd_ps(w0h, ah, c02);
        c12 = _mm512_fmadd_ps(w1l, al, c12); c12 = _mm512_fmadd_ps(w1h, ah, c12);
        c22 = _mm512_fmadd_ps(w2l, al, c22); c22 = _mm512_fmadd_ps(w2h, ah, c22);
        c32 = _mm512_fmadd_ps(w3l, al, c32); c32 = _mm512_fmadd_ps(w3h, ah, c32);
        al = _mm512_loadu_ps(x3); ah = _mm512_loadu_ps(x3 + 16);
        c03 = _mm512_fmadd_ps(w0l, al, c03); c03 = _mm512_fmadd_ps(w0h, ah, c03);
        c13 = _mm512_fmadd_ps(w1l, al, c13); c13 = _mm512_fmadd_ps(w1h, ah, c13);
        c23 = _mm512_fmadd_ps(w2l, al, c23); c23 = _mm512_fmadd_ps(w2h, ah, c23);
        c33 = _mm512_fmadd_ps(w3l, al, c33); c33 = _mm512_fmadd_ps(w3h, ah, c33);

        w0 += BLOCK_BYTES; w1 += BLOCK_BYTES; w2 += BLOCK_BYTES; w3 += BLOCK_BYTES;
        x0 += QK; x1 += QK; x2 += QK; x3 += QK;
    }
    out[0] = hsum512(c00); out[1] = hsum512(c10); out[2] = hsum512(c20); out[3] = hsum512(c30);
    out += out_stride;
    out[0] = hsum512(c01); out[1] = hsum512(c11); out[2] = hsum512(c21); out[3] = hsum512(c31);
    out += out_stride;
    out[0] = hsum512(c02); out[1] = hsum512(c12); out[2] = hsum512(c22); out[3] = hsum512(c32);
    out += out_stride;
    out[0] = hsum512(c03); out[1] = hsum512(c13); out[2] = hsum512(c23); out[3] = hsum512(c33);
}

// 4 rows sharing one activation vector (gemv): weight streaming is the bottleneck, so run
// 4 parallel row streams with light prefetch ahead of the loads.
static void gemv_rows4(const uint8_t *w, int64_t w_stride, const float *x, int kblocks, float *out) {
    const uint8_t *w0 = w, *w1 = w + w_stride, *w2 = w + 2 * w_stride, *w3 = w + 3 * w_stride;
    __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps(), c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
    for (int b = 0; b < kblocks; b++) {
        _mm_prefetch((const char *) (w0 + 8 * BLOCK_BYTES), _MM_HINT_T0);
        _mm_prefetch((const char *) (w1 + 8 * BLOCK_BYTES), _MM_HINT_T0);
        _mm_prefetch((const char *) (w2 + 8 * BLOCK_BYTES), _MM_HINT_T0);
        _mm_prefetch((const char *) (w3 + 8 * BLOCK_BYTES), _MM_HINT_T0);
        __m512 s0 = _mm512_set1_ps(fp16_to_f32(*(const uint16_t *) w0));
        __m512 s1 = _mm512_set1_ps(fp16_to_f32(*(const uint16_t *) w1));
        __m512 s2 = _mm512_set1_ps(fp16_to_f32(*(const uint16_t *) w2));
        __m512 s3 = _mm512_set1_ps(fp16_to_f32(*(const uint16_t *) w3));
        __m512 al = _mm512_loadu_ps(x), ah = _mm512_loadu_ps(x + 16);
        c0 = _mm512_fmadd_ps(q8_half_block(w0 + 2, s0), al, c0);
        c0 = _mm512_fmadd_ps(q8_half_block(w0 + 18, s0), ah, c0);
        c1 = _mm512_fmadd_ps(q8_half_block(w1 + 2, s1), al, c1);
        c1 = _mm512_fmadd_ps(q8_half_block(w1 + 18, s1), ah, c1);
        c2 = _mm512_fmadd_ps(q8_half_block(w2 + 2, s2), al, c2);
        c2 = _mm512_fmadd_ps(q8_half_block(w2 + 18, s2), ah, c2);
        c3 = _mm512_fmadd_ps(q8_half_block(w3 + 2, s3), al, c3);
        c3 = _mm512_fmadd_ps(q8_half_block(w3 + 18, s3), ah, c3);
        w0 += BLOCK_BYTES; w1 += BLOCK_BYTES; w2 += BLOCK_BYTES; w3 += BLOCK_BYTES;
        x += QK;
    }
    out[0] = hsum512(c0);
    out[1] = hsum512(c1);
    out[2] = hsum512(c2);
    out[3] = hsum512(c3);
}

static void compute_gemv_chunk(const gemm_task_t *task, int tile) {
    const int kblocks = task->dim1 / QK;
    const int64_t w_stride = (int64_t) kblocks * BLOCK_BYTES;
    const uint8_t *wbase = task->weights;
    int row = tile * GEMV_ROW_CHUNK;
    int row_end = row + GEMV_ROW_CHUNK;
    if (row_end > task->dim0) row_end = task->dim0;
    for (; row + 3 < row_end; row += 4) {
        gemv_rows4(wbase + (int64_t) row * kblocks * BLOCK_BYTES, w_stride, task->rhs, kblocks, task->out + row);
    }
    for (; row < row_end; row++) {
        tile_1x1(wbase + (int64_t) row * kblocks * BLOCK_BYTES, task->rhs, kblocks, task->out + row);
    }
}

// Quantize one f32 activation row to Q8 blocks: u8 quants (biased +128) + f32 scale.
static void quantize_row_q8(const float *x, int kblocks, uint8_t *xq, float *dx) {
    const __m512i bias = _mm512_set1_epi32(128);
    for (int b = 0; b < kblocks; b++, x += QK, xq += QK) {
        __m512 a0 = _mm512_loadu_ps(x);
        __m512 a1 = _mm512_loadu_ps(x + 16);
        __m512 amax = _mm512_max_ps(_mm512_abs_ps(a0), _mm512_abs_ps(a1));
        float max = _mm512_reduce_max_ps(amax);
        float d = max / 127.0f;
        float inv = max > 0.0f ? 127.0f / max : 0.0f;
        dx[b] = d;
        __m512 vinv = _mm512_set1_ps(inv);
        __m512i q0 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(a0, vinv)), bias);
        __m512i q1 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(a1, vinv)), bias);
        _mm_storeu_si128((__m128i *) xq, _mm512_cvtusepi32_epi8(q0));
        _mm_storeu_si128((__m128i *) (xq + 16), _mm512_cvtusepi32_epi8(q1));
    }
}

static void compute_quantize_chunk(const gemm_task_t *task, int tile) {
    const int kblocks = task->dim1 / QK;
    int s = tile * 8;
    int s_end = s + 8;
    if (s_end > task->sequence_length) s_end = task->sequence_length;
    for (; s < s_end; s++) {
        quantize_row_q8(task->rhs + (int64_t) s * task->that_stride,
                        kblocks,
                        xq_buf + (int64_t) s * kblocks * QK,
                        dx_buf + (int64_t) s * kblocks);
    }
}

// Per-worker repack scratch, sized by 32-element blocks. All quant formats unpack to one
// byte per element per row (16 bytes/element/group): qs = 512 bytes per 32-block per group.
// dw/cw are sized at PER-16-ELEMENT granularity (Q6_K's scale granularity; Q8_0/Q4_K use
// every other slot's worth, i.e. per-32).
static repack_t *ensure_repack(int worker, int kblocks) {
    repack_t *rp = &repack_scratch[worker];
    if (rp->cap_blocks < kblocks) {
        free(rp->qs);
        free(rp->dw);
        free(rp->cw);
        rp->qs = (uint8_t *) aligned_alloc(64, (size_t) (VNNI_BAND / 16) * kblocks * 512);
        rp->dw = (float *) aligned_alloc(64, (size_t) (VNNI_BAND / 16) * kblocks * 2 * 16 * sizeof(float));
        rp->cw = (float *) aligned_alloc(64, (size_t) (VNNI_BAND / 16) * kblocks * 2 * 16 * sizeof(float));
        rp->cap_blocks = kblocks;
    }
    return rp;
}

// Repack one 16-row weight group into the vnni layout (see repack_t).
static void repack_group16(const uint8_t *wbase, int64_t w_stride, int kblocks,
                           uint8_t *qs, float *dw, float *cw) {
    for (int r = 0; r < 16; r++) {
        const uint8_t *w = wbase + r * w_stride;
        for (int b = 0; b < kblocks; b++, w += BLOCK_BYTES) {
            float d = fp16_to_f32(*(const uint16_t *) w);
            __m512i q = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) (w + 2)));
            __m512i q2 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) (w + 18)));
            int wsum = _mm512_reduce_add_epi32(_mm512_add_epi32(q, q2));
            dw[b * 16 + r] = d;
            cw[b * 16 + r] = d * 128.0f * (float) wsum;
            const uint32_t *src = (const uint32_t *) (w + 2);
            uint8_t *dst = qs + (int64_t) b * 512 + r * 4;
            for (int g = 0; g < 8; g++) {
                *(uint32_t *) (dst + g * 64) = src[g];
            }
        }
    }
}

// 16 rows x 4 cols: lanes are rows, so per-block scales are per-lane vectors and the
// final accumulators store contiguously (no horizontal reductions at all).
static void vnni_tile_16x4(const uint8_t *qs, const float *dw, const float *cw,
                           const uint8_t *xq0, const float *dx0, int64_t xq_col_stride, int64_t dx_col_stride,
                           int kblocks, float *out, int64_t out_stride) {
    const uint8_t *x0 = xq0, *x1 = xq0 + xq_col_stride, *x2 = xq0 + 2 * xq_col_stride, *x3 = xq0 + 3 * xq_col_stride;
    const float *d0 = dx0, *d1 = dx0 + dx_col_stride, *d2 = dx0 + 2 * dx_col_stride, *d3 = dx0 + 3 * dx_col_stride;
    __m512 f0 = _mm512_setzero_ps(), f1 = _mm512_setzero_ps(), f2 = _mm512_setzero_ps(), f3 = _mm512_setzero_ps();
    for (int b = 0; b < kblocks; b++) {
        __m512i i0 = _mm512_setzero_si512(), i1 = _mm512_setzero_si512(), i2 = _mm512_setzero_si512(), i3 = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i w = _mm512_load_si512((const void *) (qs + g * 64));
            i0 = _mm512_dpbusd_epi32(i0, _mm512_set1_epi32(((const int *) x0)[g]), w);
            i1 = _mm512_dpbusd_epi32(i1, _mm512_set1_epi32(((const int *) x1)[g]), w);
            i2 = _mm512_dpbusd_epi32(i2, _mm512_set1_epi32(((const int *) x2)[g]), w);
            i3 = _mm512_dpbusd_epi32(i3, _mm512_set1_epi32(((const int *) x3)[g]), w);
        }
        __m512 dwv = _mm512_load_ps(dw);
        __m512 cwv = _mm512_load_ps(cw);
        __m512 dxb;
        dxb = _mm512_set1_ps(d0[b]);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i0), _mm512_mul_ps(dwv, dxb), f0);
        f0 = _mm512_fnmadd_ps(cwv, dxb, f0);
        dxb = _mm512_set1_ps(d1[b]);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i1), _mm512_mul_ps(dwv, dxb), f1);
        f1 = _mm512_fnmadd_ps(cwv, dxb, f1);
        dxb = _mm512_set1_ps(d2[b]);
        f2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i2), _mm512_mul_ps(dwv, dxb), f2);
        f2 = _mm512_fnmadd_ps(cwv, dxb, f2);
        dxb = _mm512_set1_ps(d3[b]);
        f3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i3), _mm512_mul_ps(dwv, dxb), f3);
        f3 = _mm512_fnmadd_ps(cwv, dxb, f3);
        qs += 512;
        dw += 16;
        cw += 16;
        x0 += QK; x1 += QK; x2 += QK; x3 += QK;
    }
    _mm512_storeu_ps(out, f0);
    _mm512_storeu_ps(out + out_stride, f1);
    _mm512_storeu_ps(out + 2 * out_stride, f2);
    _mm512_storeu_ps(out + 3 * out_stride, f3);
}

static void vnni_tile_16x1(const uint8_t *qs, const float *dw, const float *cw,
                           const uint8_t *x0, const float *d0,
                           int kblocks, float *out) {
    __m512 f0 = _mm512_setzero_ps();
    for (int b = 0; b < kblocks; b++) {
        __m512i i0 = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i w = _mm512_load_si512((const void *) (qs + g * 64));
            i0 = _mm512_dpbusd_epi32(i0, _mm512_set1_epi32(((const int *) x0)[g]), w);
        }
        __m512 dxb = _mm512_set1_ps(d0[b]);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i0), _mm512_mul_ps(_mm512_load_ps(dw), dxb), f0);
        f0 = _mm512_fnmadd_ps(_mm512_load_ps(cw), dxb, f0);
        qs += 512;
        dw += 16;
        cw += 16;
        x0 += QK;
    }
    _mm512_storeu_ps(out, f0);
}

static void compute_vnni_band(const gemm_task_t *task, int tile, int worker) {
    const int kblocks = task->dim1 / QK;
    const int64_t w_stride = (int64_t) kblocks * BLOCK_BYTES;
    const int seq = task->sequence_length;
    int row = tile * VNNI_BAND;
    int row_end = row + VNNI_BAND;
    if (row_end > task->dim0) row_end = task->dim0;

    repack_t *rp = ensure_repack(worker, kblocks);

    const uint8_t *wbase = task->weights;
    int group = 0;
    for (int r = row; r + 15 < row_end; r += 16, group++) {
        uint8_t *qs = rp->qs + (int64_t) group * kblocks * 512;
        float *dw = rp->dw + (int64_t) group * kblocks * 16;
        float *cw = rp->cw + (int64_t) group * kblocks * 16;
        repack_group16(wbase + (int64_t) r * w_stride, w_stride, kblocks, qs, dw, cw);
        int s = 0;
        for (; s + 3 < seq; s += 4) {
            vnni_tile_16x4(qs, dw, cw,
                           xq_buf + (int64_t) s * kblocks * QK,
                           dx_buf + (int64_t) s * kblocks,
                           (int64_t) kblocks * QK, kblocks,
                           kblocks,
                           task->out + (int64_t) s * task->out_stride + r, task->out_stride);
        }
        for (; s < seq; s++) {
            vnni_tile_16x1(qs, dw, cw,
                           xq_buf + (int64_t) s * kblocks * QK,
                           dx_buf + (int64_t) s * kblocks,
                           kblocks,
                           task->out + (int64_t) s * task->out_stride + r);
        }
    }
    // leftover rows (< 16): plain f32 dots
    for (int r = row + (group * 16); r < row_end; r++) {
        for (int s = 0; s < seq; s++) {
            tile_1x1(wbase + (int64_t) r * w_stride,
                     task->rhs + (int64_t) s * task->that_stride, kblocks,
                     task->out + (int64_t) s * task->out_stride + r);
        }
    }
}

// ===================== K-quant (Q4_K / Q6_K) VNNI path =====================
//
// Same band/tile structure as the Q8_0 vnni path, with the vpdpbusd operands SWAPPED:
// K-quant weights unpack to UNSIGNED bytes (nibbles 0..15, or q6+32 in 0..63), so they take
// the unsigned slot and the activations quantize to plain SIGNED int8 — no +128 bias and no
// bias-correction term. The affine parts stay exact in f32: Q4_K's per-sub-block min term is
// dmin*m*sum(x) over the UNQUANTIZED activations, and Q6_K's -32 offset is d*sc*32*sum(x);
// both come from per-16-element activation sums computed during quantization.

// Quantize one f32 row to signed q8 (scale per 32) plus exact f32 sums per 16 elements.
static void quantize_row_q8s(const float *x, int kblocks, int8_t *xq, float *dx, float *xs) {
    for (int b = 0; b < kblocks; b++, x += QK, xq += QK, xs += 2) {
        __m512 a0 = _mm512_loadu_ps(x);
        __m512 a1 = _mm512_loadu_ps(x + 16);
        float max = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a0), _mm512_abs_ps(a1)));
        float d = max / 127.0f;
        float inv = max > 0.0f ? 127.0f / max : 0.0f;
        dx[b] = d;
        xs[0] = _mm512_reduce_add_ps(a0);
        xs[1] = _mm512_reduce_add_ps(a1);
        __m512 vinv = _mm512_set1_ps(inv);
        __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(a0, vinv));
        __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(a1, vinv));
        _mm_storeu_si128((__m128i *) xq, _mm512_cvtsepi32_epi8(q0));
        _mm_storeu_si128((__m128i *) (xq + 16), _mm512_cvtsepi32_epi8(q1));
    }
}

static void compute_quantize_s8_chunk(const gemm_task_t *task, int tile) {
    const int kblocks = task->dim1 / QK;
    int s = tile * 8;
    int s_end = s + 8;
    if (s_end > task->sequence_length) s_end = task->sequence_length;
    for (; s < s_end; s++) {
        quantize_row_q8s(task->rhs + (int64_t) s * task->that_stride,
                         kblocks,
                         (int8_t *) xq_buf + (int64_t) s * kblocks * QK,
                         dx_buf + (int64_t) s * kblocks,
                         xsum_buf + (int64_t) s * kblocks * 2);
    }
}

// The 8 6-bit (scale, min) pairs of a Q4_K super-block.
static inline void q4k_scales_mins(const uint8_t *b, uint8_t *sc, uint8_t *mn) {
    for (int j = 0; j < 4; j++) {
        sc[j] = b[j] & 63;
        mn[j] = b[j + 4] & 63;
        sc[j + 4] = (uint8_t) ((b[j + 8] & 0xF) | ((b[j] >> 6) << 4));
        mn[j + 4] = (uint8_t) ((b[j + 8] >> 4) | ((b[j + 4] >> 6) << 4));
    }
}

// Repack one 16-row Q4_K group: per 32-element sub-block, 8 K-groups of 16 rows x 4 bytes
// (unsigned nibble values); dw = d*scale and mw = dmin*min per row lane.
static void repack_q4k_group16(const uint8_t *wbase, int64_t w_stride, int sblocks,
                               uint8_t *qs, float *dw, float *mw) {
    for (int r = 0; r < 16; r++) {
        const uint8_t *w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += Q4K_BLOCK_BYTES) {
            float d = fp16_to_f32(*(const uint16_t *) w);
            float dmin = fp16_to_f32(*(const uint16_t *) (w + 2));
            uint8_t sc[8], mn[8];
            q4k_scales_mins(w + 4, sc, mn);
            const uint8_t *q = w + 16;
            for (int g = 0; g < 4; g++) {
                int sbLo = B * 8 + g * 2;
                int sbHi = sbLo + 1;
                dw[sbLo * 16 + r] = d * sc[g * 2];
                mw[sbLo * 16 + r] = dmin * mn[g * 2];
                dw[sbHi * 16 + r] = d * sc[g * 2 + 1];
                mw[sbHi * 16 + r] = dmin * mn[g * 2 + 1];
                uint8_t lo[32], hi[32];
                for (int i = 0; i < 32; i++) {
                    lo[i] = q[g * 32 + i] & 0xF;
                    hi[i] = q[g * 32 + i] >> 4;
                }
                uint8_t *dstLo = qs + (int64_t) sbLo * 512 + r * 4;
                uint8_t *dstHi = qs + (int64_t) sbHi * 512 + r * 4;
                for (int k = 0; k < 8; k++) {
                    *(uint32_t *) (dstLo + k * 64) = *(const uint32_t *) (lo + k * 4);
                    *(uint32_t *) (dstHi + k * 64) = *(const uint32_t *) (hi + k * 4);
                }
            }
        }
    }
}

// 16 rows x 4 cols over all 32-element sub-blocks; subs = dim1/32.
static void q4k_vnni_tile_16x4(const uint8_t *qs, const float *dw, const float *mw,
                               const int8_t *xq0, const float *dx0, const float *xs0,
                               int64_t xq_col, int64_t dx_col, int64_t xs_col,
                               int subs, float *out, int64_t out_stride) {
    const int8_t *x0 = xq0, *x1 = xq0 + xq_col, *x2 = xq0 + 2 * xq_col, *x3 = xq0 + 3 * xq_col;
    const float *d0 = dx0, *d1 = dx0 + dx_col, *d2 = dx0 + 2 * dx_col, *d3 = dx0 + 3 * dx_col;
    const float *s0 = xs0, *s1 = xs0 + xs_col, *s2 = xs0 + 2 * xs_col, *s3 = xs0 + 3 * xs_col;
    __m512 f0 = _mm512_setzero_ps(), f1 = _mm512_setzero_ps(), f2 = _mm512_setzero_ps(), f3 = _mm512_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m512i i0 = _mm512_setzero_si512(), i1 = _mm512_setzero_si512(), i2 = _mm512_setzero_si512(), i3 = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i w = _mm512_load_si512((const void *) (qs + g * 64));
            i0 = _mm512_dpbusd_epi32(i0, w, _mm512_set1_epi32(((const int *) x0)[g]));
            i1 = _mm512_dpbusd_epi32(i1, w, _mm512_set1_epi32(((const int *) x1)[g]));
            i2 = _mm512_dpbusd_epi32(i2, w, _mm512_set1_epi32(((const int *) x2)[g]));
            i3 = _mm512_dpbusd_epi32(i3, w, _mm512_set1_epi32(((const int *) x3)[g]));
        }
        __m512 dwv = _mm512_load_ps(dw);
        __m512 mwv = _mm512_load_ps(mw);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i0), _mm512_mul_ps(dwv, _mm512_set1_ps(d0[b])), f0);
        f0 = _mm512_fnmadd_ps(mwv, _mm512_set1_ps(s0[2 * b] + s0[2 * b + 1]), f0);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i1), _mm512_mul_ps(dwv, _mm512_set1_ps(d1[b])), f1);
        f1 = _mm512_fnmadd_ps(mwv, _mm512_set1_ps(s1[2 * b] + s1[2 * b + 1]), f1);
        f2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i2), _mm512_mul_ps(dwv, _mm512_set1_ps(d2[b])), f2);
        f2 = _mm512_fnmadd_ps(mwv, _mm512_set1_ps(s2[2 * b] + s2[2 * b + 1]), f2);
        f3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i3), _mm512_mul_ps(dwv, _mm512_set1_ps(d3[b])), f3);
        f3 = _mm512_fnmadd_ps(mwv, _mm512_set1_ps(s3[2 * b] + s3[2 * b + 1]), f3);
        qs += 512;
        dw += 16;
        mw += 16;
        x0 += QK; x1 += QK; x2 += QK; x3 += QK;
    }
    _mm512_storeu_ps(out, f0);
    _mm512_storeu_ps(out + out_stride, f1);
    _mm512_storeu_ps(out + 2 * out_stride, f2);
    _mm512_storeu_ps(out + 3 * out_stride, f3);
}

static void q4k_vnni_tile_16x1(const uint8_t *qs, const float *dw, const float *mw,
                               const int8_t *x0, const float *d0, const float *s0,
                               int subs, float *out) {
    __m512 f0 = _mm512_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m512i i0 = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i w = _mm512_load_si512((const void *) (qs + g * 64));
            i0 = _mm512_dpbusd_epi32(i0, w, _mm512_set1_epi32(((const int *) x0)[g]));
        }
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i0), _mm512_mul_ps(_mm512_load_ps(dw), _mm512_set1_ps(d0[b])), f0);
        f0 = _mm512_fnmadd_ps(_mm512_load_ps(mw), _mm512_set1_ps(s0[2 * b] + s0[2 * b + 1]), f0);
        qs += 512;
        dw += 16;
        mw += 16;
        x0 += QK;
    }
    _mm512_storeu_ps(out, f0);
}

// Plain f32 dot for leftover rows (< 16 in the last band).
static float q4k_dot_scalar(const uint8_t *w, const float *x, int sblocks) {
    float acc = 0.0f;
    for (int B = 0; B < sblocks; B++, w += Q4K_BLOCK_BYTES, x += QKK) {
        float d = fp16_to_f32(*(const uint16_t *) w);
        float dmin = fp16_to_f32(*(const uint16_t *) (w + 2));
        uint8_t sc[8], mn[8];
        q4k_scales_mins(w + 4, sc, mn);
        const uint8_t *q = w + 16;
        for (int g = 0; g < 4; g++) {
            float dl = d * sc[g * 2], ml = dmin * mn[g * 2];
            float dh = d * sc[g * 2 + 1], mh = dmin * mn[g * 2 + 1];
            for (int i = 0; i < 32; i++) {
                acc += (dl * (q[g * 32 + i] & 0xF) - ml) * x[g * 64 + i];
                acc += (dh * (q[g * 32 + i] >> 4) - mh) * x[g * 64 + 32 + i];
            }
        }
    }
    return acc;
}

static void compute_q4k_band(const gemm_task_t *task, int tile, int worker) {
    const int kblocks = task->dim1 / QK;
    const int sblocks = task->dim1 / QKK;
    const int64_t w_stride = (int64_t) sblocks * Q4K_BLOCK_BYTES;
    const int seq = task->sequence_length;
    int row = tile * VNNI_BAND;
    int row_end = row + VNNI_BAND;
    if (row_end > task->dim0) row_end = task->dim0;

    repack_t *rp = ensure_repack(worker, kblocks);
    const uint8_t *wbase = task->weights;
    int group = 0;
    for (int r = row; r + 15 < row_end; r += 16, group++) {
        uint8_t *qs = rp->qs + (int64_t) group * kblocks * 512;
        float *dw = rp->dw + (int64_t) group * kblocks * 16;
        float *mw = rp->cw + (int64_t) group * kblocks * 16;
        repack_q4k_group16(wbase + (int64_t) r * w_stride, w_stride, sblocks, qs, dw, mw);
        int s = 0;
        for (; s + 3 < seq; s += 4) {
            q4k_vnni_tile_16x4(qs, dw, mw,
                               (const int8_t *) xq_buf + (int64_t) s * kblocks * QK,
                               dx_buf + (int64_t) s * kblocks,
                               xsum_buf + (int64_t) s * kblocks * 2,
                               (int64_t) kblocks * QK, kblocks, (int64_t) kblocks * 2,
                               kblocks,
                               task->out + (int64_t) s * task->out_stride + r, task->out_stride);
        }
        for (; s < seq; s++) {
            q4k_vnni_tile_16x1(qs, dw, mw,
                               (const int8_t *) xq_buf + (int64_t) s * kblocks * QK,
                               dx_buf + (int64_t) s * kblocks,
                               xsum_buf + (int64_t) s * kblocks * 2,
                               kblocks,
                               task->out + (int64_t) s * task->out_stride + r);
        }
    }
    for (int r = row + (group * 16); r < row_end; r++) {
        for (int s = 0; s < seq; s++) {
            task->out[(int64_t) s * task->out_stride + r] =
                    q4k_dot_scalar(wbase + (int64_t) r * w_stride,
                                   task->rhs + (int64_t) s * task->that_stride, sblocks);
        }
    }
}

// Repack one 16-row Q6_K group: per 16-element sub-block, 4 K-groups of 16 rows x 4 bytes
// (q6+32 as unsigned 0..63); dw = d*scale per row lane (scales are per 16 elements).
static void repack_q6k_group16(const uint8_t *wbase, int64_t w_stride, int sblocks,
                               uint8_t *qs, float *dw) {
    for (int r = 0; r < 16; r++) {
        const uint8_t *w = wbase + r * w_stride;
        for (int B = 0; B < sblocks; B++, w += Q6K_BLOCK_BYTES) {
            const uint8_t *ql = w;
            const uint8_t *qh = w + 128;
            const int8_t *sc = (const int8_t *) (w + 192);
            float d = fp16_to_f32(*(const uint16_t *) (w + 208));
            for (int h = 0; h < 2; h++) {
                const uint8_t *qlb = ql + h * 64;
                const uint8_t *qhb = qh + h * 32;
                for (int j = 0; j < 4; j++) {
                    int t0 = B * 16 + h * 8 + j * 2; // two 16-element sub-blocks per 32 values
                    dw[t0 * 16 + r] = d * sc[h * 8 + j * 2];
                    dw[(t0 + 1) * 16 + r] = d * sc[h * 8 + j * 2 + 1];
                    for (int l = 0; l < 32; l++) {
                        int qv;
                        switch (j) {
                            case 0: qv = qlb[l] & 0xF; break;
                            case 1: qv = qlb[32 + l] & 0xF; break;
                            case 2: qv = qlb[l] >> 4; break;
                            default: qv = qlb[32 + l] >> 4; break;
                        }
                        qv |= ((qhb[l] >> (2 * j)) & 3) << 4;
                        int t = t0 + l / 16;
                        int e = l % 16;
                        qs[(int64_t) t * 256 + (e / 4) * 64 + r * 4 + (e % 4)] = (uint8_t) qv;
                    }
                }
            }
        }
    }
}

// 16 rows x 4 cols over all 16-element sub-blocks; subs = dim1/16. The -32 offset of q6
// folds into dw*32*xsum16 (activation scale dx is per 32: index b>>1).
static void q6k_vnni_tile_16x4(const uint8_t *qs, const float *dw,
                               const int8_t *xq0, const float *dx0, const float *xs0,
                               int64_t xq_col, int64_t dx_col, int64_t xs_col,
                               int subs, float *out, int64_t out_stride) {
    const int8_t *x0 = xq0, *x1 = xq0 + xq_col, *x2 = xq0 + 2 * xq_col, *x3 = xq0 + 3 * xq_col;
    const float *d0 = dx0, *d1 = dx0 + dx_col, *d2 = dx0 + 2 * dx_col, *d3 = dx0 + 3 * dx_col;
    const float *s0 = xs0, *s1 = xs0 + xs_col, *s2 = xs0 + 2 * xs_col, *s3 = xs0 + 3 * xs_col;
    __m512 f0 = _mm512_setzero_ps(), f1 = _mm512_setzero_ps(), f2 = _mm512_setzero_ps(), f3 = _mm512_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m512i i0 = _mm512_setzero_si512(), i1 = _mm512_setzero_si512(), i2 = _mm512_setzero_si512(), i3 = _mm512_setzero_si512();
        for (int g = 0; g < 4; g++) {
            __m512i w = _mm512_load_si512((const void *) (qs + g * 64));
            i0 = _mm512_dpbusd_epi32(i0, w, _mm512_set1_epi32(((const int *) x0)[g]));
            i1 = _mm512_dpbusd_epi32(i1, w, _mm512_set1_epi32(((const int *) x1)[g]));
            i2 = _mm512_dpbusd_epi32(i2, w, _mm512_set1_epi32(((const int *) x2)[g]));
            i3 = _mm512_dpbusd_epi32(i3, w, _mm512_set1_epi32(((const int *) x3)[g]));
        }
        __m512 dwv = _mm512_load_ps(dw);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i0), _mm512_mul_ps(dwv, _mm512_set1_ps(d0[b >> 1])), f0);
        f0 = _mm512_fnmadd_ps(dwv, _mm512_set1_ps(32.0f * s0[b]), f0);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i1), _mm512_mul_ps(dwv, _mm512_set1_ps(d1[b >> 1])), f1);
        f1 = _mm512_fnmadd_ps(dwv, _mm512_set1_ps(32.0f * s1[b]), f1);
        f2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i2), _mm512_mul_ps(dwv, _mm512_set1_ps(d2[b >> 1])), f2);
        f2 = _mm512_fnmadd_ps(dwv, _mm512_set1_ps(32.0f * s2[b]), f2);
        f3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i3), _mm512_mul_ps(dwv, _mm512_set1_ps(d3[b >> 1])), f3);
        f3 = _mm512_fnmadd_ps(dwv, _mm512_set1_ps(32.0f * s3[b]), f3);
        qs += 256;
        dw += 16;
        x0 += 16; x1 += 16; x2 += 16; x3 += 16;
    }
    _mm512_storeu_ps(out, f0);
    _mm512_storeu_ps(out + out_stride, f1);
    _mm512_storeu_ps(out + 2 * out_stride, f2);
    _mm512_storeu_ps(out + 3 * out_stride, f3);
}

static void q6k_vnni_tile_16x1(const uint8_t *qs, const float *dw,
                               const int8_t *x0, const float *d0, const float *s0,
                               int subs, float *out) {
    __m512 f0 = _mm512_setzero_ps();
    for (int b = 0; b < subs; b++) {
        __m512i i0 = _mm512_setzero_si512();
        for (int g = 0; g < 4; g++) {
            __m512i w = _mm512_load_si512((const void *) (qs + g * 64));
            i0 = _mm512_dpbusd_epi32(i0, w, _mm512_set1_epi32(((const int *) x0)[g]));
        }
        __m512 dwv = _mm512_load_ps(dw);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i0), _mm512_mul_ps(dwv, _mm512_set1_ps(d0[b >> 1])), f0);
        f0 = _mm512_fnmadd_ps(dwv, _mm512_set1_ps(32.0f * s0[b]), f0);
        qs += 256;
        dw += 16;
        x0 += 16;
    }
    _mm512_storeu_ps(out, f0);
}

static float q6k_dot_scalar(const uint8_t *w, const float *x, int sblocks) {
    float acc = 0.0f;
    for (int B = 0; B < sblocks; B++, w += Q6K_BLOCK_BYTES, x += QKK) {
        const uint8_t *ql = w;
        const uint8_t *qh = w + 128;
        const int8_t *sc = (const int8_t *) (w + 192);
        float d = fp16_to_f32(*(const uint16_t *) (w + 208));
        for (int h = 0; h < 2; h++) {
            const uint8_t *qlb = ql + h * 64;
            const uint8_t *qhb = qh + h * 32;
            for (int j = 0; j < 4; j++) {
                for (int l = 0; l < 32; l++) {
                    int qv;
                    switch (j) {
                        case 0: qv = qlb[l] & 0xF; break;
                        case 1: qv = qlb[32 + l] & 0xF; break;
                        case 2: qv = qlb[l] >> 4; break;
                        default: qv = qlb[32 + l] >> 4; break;
                    }
                    qv |= ((qhb[l] >> (2 * j)) & 3) << 4;
                    float ws = d * sc[h * 8 + j * 2 + l / 16];
                    acc += ws * (qv - 32) * x[h * 128 + j * 32 + l];
                }
            }
        }
    }
    return acc;
}

static void compute_q6k_band(const gemm_task_t *task, int tile, int worker) {
    const int kblocks = task->dim1 / QK;
    const int sblocks = task->dim1 / QKK;
    const int subs16 = task->dim1 / 16;
    const int64_t w_stride = (int64_t) sblocks * Q6K_BLOCK_BYTES;
    const int seq = task->sequence_length;
    int row = tile * VNNI_BAND;
    int row_end = row + VNNI_BAND;
    if (row_end > task->dim0) row_end = task->dim0;

    repack_t *rp = ensure_repack(worker, kblocks);
    const uint8_t *wbase = task->weights;
    int group = 0;
    for (int r = row; r + 15 < row_end; r += 16, group++) {
        uint8_t *qs = rp->qs + (int64_t) group * kblocks * 512;
        float *dw = rp->dw + (int64_t) group * subs16 * 16;
        repack_q6k_group16(wbase + (int64_t) r * w_stride, w_stride, sblocks, qs, dw);
        int s = 0;
        for (; s + 3 < seq; s += 4) {
            q6k_vnni_tile_16x4(qs, dw,
                               (const int8_t *) xq_buf + (int64_t) s * kblocks * QK,
                               dx_buf + (int64_t) s * kblocks,
                               xsum_buf + (int64_t) s * kblocks * 2,
                               (int64_t) kblocks * QK, kblocks, (int64_t) kblocks * 2,
                               subs16,
                               task->out + (int64_t) s * task->out_stride + r, task->out_stride);
        }
        for (; s < seq; s++) {
            q6k_vnni_tile_16x1(qs, dw,
                               (const int8_t *) xq_buf + (int64_t) s * kblocks * QK,
                               dx_buf + (int64_t) s * kblocks,
                               xsum_buf + (int64_t) s * kblocks * 2,
                               subs16,
                               task->out + (int64_t) s * task->out_stride + r);
        }
    }
    for (int r = row + (group * 16); r < row_end; r++) {
        for (int s = 0; s < seq; s++) {
            task->out[(int64_t) s * task->out_stride + r] =
                    q6k_dot_scalar(wbase + (int64_t) r * w_stride,
                                   task->rhs + (int64_t) s * task->that_stride, sblocks);
        }
    }
}

// ---- Dense GEMM: (F32 | F16 | BF16) weight @ F32 activations -> F32, register-tiled 4x4,
// AVX-512, run on the worker pool. Requires dim1 % 16 == 0 (callers fall back to AVX2 otherwise).
// 16 fp32 accumulators reuse each weight row across 4 seq columns and each activation column across
// 4 weight rows; the contraction (dim1) is vectorized 16-wide.
static inline __m512 loadw_f32(const uint8_t *p)  { return _mm512_loadu_ps((const float *) p); }
static inline __m512 loadw_f16(const uint8_t *p)  { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) p)); }
static inline __m512 loadw_bf16(const uint8_t *p) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *) p)), 16)); }

#define DENSE_BLOCK(NAME, LOADW, EB)                                                                        \
static void NAME(const uint8_t *wbase, int64_t w_stride, const float *rhs, int that_stride,                 \
                 float *out, int out_stride, int rs, int re, int ss, int se, int dim1) {                    \
    int r = rs;                                                                                             \
    for (; r + 4 <= re; r += 4) {                                                                           \
        const uint8_t *w0 = wbase + (int64_t) r * w_stride, *w1 = w0 + w_stride,                            \
                      *w2 = w1 + w_stride, *w3 = w2 + w_stride;                                             \
        int s = ss;                                                                                         \
        for (; s + 4 <= se; s += 4) {                                                                       \
            const float *x0 = rhs + (int64_t) s * that_stride, *x1 = x0 + that_stride,                      \
                        *x2 = x1 + that_stride, *x3 = x2 + that_stride;                                     \
            __m512 a00=_mm512_setzero_ps(),a01=a00,a02=a00,a03=a00, a10=a00,a11=a00,a12=a00,a13=a00,        \
                   a20=a00,a21=a00,a22=a00,a23=a00, a30=a00,a31=a00,a32=a00,a33=a00;                        \
            for (int k = 0; k < dim1; k += 16) {                                                            \
                __m512 wv0=LOADW(w0+(int64_t)k*EB), wv1=LOADW(w1+(int64_t)k*EB),                            \
                       wv2=LOADW(w2+(int64_t)k*EB), wv3=LOADW(w3+(int64_t)k*EB);                            \
                __m512 xv0=_mm512_loadu_ps(x0+k), xv1=_mm512_loadu_ps(x1+k),                                \
                       xv2=_mm512_loadu_ps(x2+k), xv3=_mm512_loadu_ps(x3+k);                                \
                a00=_mm512_fmadd_ps(wv0,xv0,a00); a01=_mm512_fmadd_ps(wv1,xv0,a01); a02=_mm512_fmadd_ps(wv2,xv0,a02); a03=_mm512_fmadd_ps(wv3,xv0,a03); \
                a10=_mm512_fmadd_ps(wv0,xv1,a10); a11=_mm512_fmadd_ps(wv1,xv1,a11); a12=_mm512_fmadd_ps(wv2,xv1,a12); a13=_mm512_fmadd_ps(wv3,xv1,a13); \
                a20=_mm512_fmadd_ps(wv0,xv2,a20); a21=_mm512_fmadd_ps(wv1,xv2,a21); a22=_mm512_fmadd_ps(wv2,xv2,a22); a23=_mm512_fmadd_ps(wv3,xv2,a23); \
                a30=_mm512_fmadd_ps(wv0,xv3,a30); a31=_mm512_fmadd_ps(wv1,xv3,a31); a32=_mm512_fmadd_ps(wv2,xv3,a32); a33=_mm512_fmadd_ps(wv3,xv3,a33); \
            }                                                                                               \
            float *o0=out+(int64_t)s*out_stride+r, *o1=o0+out_stride, *o2=o1+out_stride, *o3=o2+out_stride; \
            o0[0]=_mm512_reduce_add_ps(a00); o0[1]=_mm512_reduce_add_ps(a01); o0[2]=_mm512_reduce_add_ps(a02); o0[3]=_mm512_reduce_add_ps(a03); \
            o1[0]=_mm512_reduce_add_ps(a10); o1[1]=_mm512_reduce_add_ps(a11); o1[2]=_mm512_reduce_add_ps(a12); o1[3]=_mm512_reduce_add_ps(a13); \
            o2[0]=_mm512_reduce_add_ps(a20); o2[1]=_mm512_reduce_add_ps(a21); o2[2]=_mm512_reduce_add_ps(a22); o2[3]=_mm512_reduce_add_ps(a23); \
            o3[0]=_mm512_reduce_add_ps(a30); o3[1]=_mm512_reduce_add_ps(a31); o3[2]=_mm512_reduce_add_ps(a32); o3[3]=_mm512_reduce_add_ps(a33); \
        }                                                                                                   \
        for (; s < se; s++) {                                                                               \
            const float *xs = rhs + (int64_t) s * that_stride;                                              \
            __m512 b0=_mm512_setzero_ps(),b1=b0,b2=b0,b3=b0;                                                \
            for (int k=0;k<dim1;k+=16){ __m512 xv=_mm512_loadu_ps(xs+k);                                    \
                b0=_mm512_fmadd_ps(LOADW(w0+(int64_t)k*EB),xv,b0); b1=_mm512_fmadd_ps(LOADW(w1+(int64_t)k*EB),xv,b1); \
                b2=_mm512_fmadd_ps(LOADW(w2+(int64_t)k*EB),xv,b2); b3=_mm512_fmadd_ps(LOADW(w3+(int64_t)k*EB),xv,b3); } \
            float *o=out+(int64_t)s*out_stride+r;                                                           \
            o[0]=_mm512_reduce_add_ps(b0); o[1]=_mm512_reduce_add_ps(b1); o[2]=_mm512_reduce_add_ps(b2); o[3]=_mm512_reduce_add_ps(b3); \
        }                                                                                                   \
    }                                                                                                       \
    for (; r < re; r++) {                                                                                   \
        const uint8_t *w = wbase + (int64_t) r * w_stride;                                                  \
        for (int s = ss; s < se; s++) {                                                                     \
            const float *xs = rhs + (int64_t) s * that_stride;                                              \
            __m512 acc=_mm512_setzero_ps();                                                                 \
            for (int k=0;k<dim1;k+=16) acc=_mm512_fmadd_ps(LOADW(w+(int64_t)k*EB), _mm512_loadu_ps(xs+k), acc); \
            out[(int64_t)s*out_stride+r]=_mm512_reduce_add_ps(acc);                                         \
        }                                                                                                   \
    }                                                                                                       \
}
DENSE_BLOCK(dense_block_f32, loadw_f32, 4)
DENSE_BLOCK(dense_block_f16, loadw_f16, 2)
DENSE_BLOCK(dense_block_bf16, loadw_bf16, 2)

static void compute_dense_chunk(const gemm_task_t *task, int tile) {
    int row_start = (tile / task->seq_tile_count) * task->row_tile;
    int seq_start = (tile % task->seq_tile_count) * task->seq_tile;
    int row_end = row_start + task->row_tile;  if (row_end > task->dim0) row_end = task->dim0;
    int seq_end = seq_start + task->seq_tile;  if (seq_end > task->sequence_length) seq_end = task->sequence_length;
    int64_t w_stride = (int64_t) task->dim1 * (task->dtype == 0 ? 4 : 2);
    if (task->dtype == 0)      dense_block_f32 (task->weights, w_stride, task->rhs, task->that_stride, task->out, task->out_stride, row_start, row_end, seq_start, seq_end, task->dim1);
    else if (task->dtype == 1) dense_block_f16 (task->weights, w_stride, task->rhs, task->that_stride, task->out, task->out_stride, row_start, row_end, seq_start, seq_end, task->dim1);
    else                       dense_block_bf16(task->weights, w_stride, task->rhs, task->that_stride, task->out, task->out_stride, row_start, row_end, seq_start, seq_end, task->dim1);
}

static void compute_tile(const gemm_task_t *task, int tile, int worker) {
    if (task->kind == 7) {
        compute_dense_chunk(task, tile);
        return;
    }
    if (task->kind == 1) {
        compute_gemv_chunk(task, tile);
        return;
    }
    if (task->kind == 2) {
        compute_quantize_chunk(task, tile);
        return;
    }
    if (task->kind == 3) {
        compute_vnni_band(task, tile, worker);
        return;
    }
    if (task->kind == 4) {
        compute_quantize_s8_chunk(task, tile);
        return;
    }
    if (task->kind == 5) {
        compute_q4k_band(task, tile, worker);
        return;
    }
    if (task->kind == 6) {
        compute_q6k_band(task, tile, worker);
        return;
    }
    const int kblocks = task->dim1 / QK;
    const int64_t w_stride = (int64_t) kblocks * BLOCK_BYTES;
    int row_start = (tile / task->seq_tile_count) * task->row_tile;
    int seq_start = (tile % task->seq_tile_count) * task->seq_tile;
    int row_end = row_start + task->row_tile;
    int seq_end = seq_start + task->seq_tile;
    if (row_end > task->dim0) row_end = task->dim0;
    if (seq_end > task->sequence_length) seq_end = task->sequence_length;

    const uint8_t *wbase = task->weights;
    int row = row_start;
    for (; row + 3 < row_end; row += 4) {
        const uint8_t *w = wbase + (int64_t) row * kblocks * BLOCK_BYTES;
        int s = seq_start;
        for (; s + 3 < seq_end; s += 4) {
            tile_4x4(w, w_stride,
                     task->rhs + (int64_t) s * task->that_stride, task->that_stride,
                     kblocks,
                     task->out + (int64_t) s * task->out_stride + row, task->out_stride);
        }
        for (; s < seq_end; s++) {
            const float *x = task->rhs + (int64_t) s * task->that_stride;
            float *o = task->out + (int64_t) s * task->out_stride + row;
            tile_1x1(w, x, kblocks, o);
            tile_1x1(w + w_stride, x, kblocks, o + 1);
            tile_1x1(w + 2 * w_stride, x, kblocks, o + 2);
            tile_1x1(w + 3 * w_stride, x, kblocks, o + 3);
        }
    }
    for (; row < row_end; row++) {
        const uint8_t *w = wbase + (int64_t) row * kblocks * BLOCK_BYTES;
        for (int s = seq_start; s < seq_end; s++) {
            tile_1x1(w, task->rhs + (int64_t) s * task->that_stride, kblocks,
                     task->out + (int64_t) s * task->out_stride + row);
        }
    }
}

static void run_task(const gemm_task_t *task, int worker) {
    for (;;) {
        int tile = atomic_fetch_add_explicit(&next_tile, 1, memory_order_relaxed);
        if (tile >= task->tile_count) {
            return;
        }
        compute_tile(task, tile, worker);
    }
}

static void pin_worker_if_requested(int worker) {
#if defined(__linux__)
    const char *pin = getenv("LFM25_NATIVE_PIN_THREADS");
    if (pin == NULL || pin[0] == '0') {
        return;
    }
    int base = 0;
    int stride = 1;
    const char *base_env = getenv("LFM25_NATIVE_CPU_BASE");
    const char *stride_env = getenv("LFM25_NATIVE_CPU_STRIDE");
    if (base_env != NULL) base = atoi(base_env);
    if (stride_env != NULL) stride = atoi(stride_env);
    if (stride < 1) stride = 1;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(base + worker * stride, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#else
    (void) worker;
#endif
}

static void *pool_worker(void *arg) {
    int worker = (int) (intptr_t) arg;
    int seen_generation = 0;
    pin_worker_if_requested(worker);
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&pool_generation, memory_order_acquire) == seen_generation) {
            if (pool_stop) {
                return NULL;
            }
            if (++spins >= pool_spin) {
                pthread_mutex_lock(&pool_mutex);
                atomic_fetch_add_explicit(&pool_sleepers, 1, memory_order_relaxed);
                while (!pool_stop && atomic_load_explicit(&pool_generation, memory_order_acquire) == seen_generation) {
                    pthread_cond_wait(&pool_start, &pool_mutex);
                }
                atomic_fetch_sub_explicit(&pool_sleepers, 1, memory_order_relaxed);
                pthread_mutex_unlock(&pool_mutex);
                spins = 0;
            } else {
                _mm_pause();
            }
        }
        seen_generation = atomic_load_explicit(&pool_generation, memory_order_acquire);
        run_task(&current_task, worker);
        atomic_fetch_sub_explicit(&pool_active, 1, memory_order_release);
    }
}

// Created once; sized from LFM25_NATIVE_THREADS or the number of online CPUs.
static int ensure_pool_locked(void) {
    if (pool_size > 0) {
        return 1;
    }
    int workers = (int) sysconf(_SC_NPROCESSORS_ONLN);
    const char *env = getenv("LFM25_NATIVE_THREADS");
    if (env != NULL && atoi(env) > 0) {
        workers = atoi(env);
    }
    if (workers < 1) workers = 1;
    workers -= 1; // the dispatching (Java) thread participates in every task
    if (workers < 1) workers = 1;
    const char *spin_env = getenv("LFM25_NATIVE_SPIN");
    if (spin_env != NULL && atoi(spin_env) >= 0) {
        pool_spin = atoi(spin_env);
    }
    pool_threads = (pthread_t *) calloc((size_t) workers, sizeof(pthread_t));
    if (pool_threads == NULL) {
        return 0;
    }
    for (int i = 0; i < workers; i++) {
        if (pthread_create(&pool_threads[i], NULL, pool_worker, (void *) (intptr_t) i) != 0) {
            pool_stop = 1;
            pthread_cond_broadcast(&pool_start);
            pthread_mutex_unlock(&pool_mutex);
            for (int j = 0; j < i; j++) pthread_join(pool_threads[j], NULL);
            pthread_mutex_lock(&pool_mutex);
            free(pool_threads);
            pool_threads = NULL;
            pool_stop = 0;
            return 0;
        }
    }
    repack_scratch = (repack_t *) calloc((size_t) workers + 1, sizeof(repack_t));
    if (repack_scratch == NULL) {
        return 0;
    }
    pool_size = workers;
    return 1;
}

static void destroy_pool_locked(void) {
    if (pool_size == 0) {
        return;
    }
    pool_stop = 1;
    pthread_cond_broadcast(&pool_start);
    pthread_mutex_unlock(&pool_mutex);
    for (int i = 0; i < pool_size; i++) {
        pthread_join(pool_threads[i], NULL);
    }
    pthread_mutex_lock(&pool_mutex);
    free(pool_threads);
    pool_threads = NULL;
    pool_size = 0;
    pool_stop = 0;
}

// current_task must be fully initialized; runs it on the pool with the caller participating.
static void dispatch_task(void) {
    pthread_mutex_lock(&pool_mutex);
    if (!ensure_pool_locked()) {
        pthread_mutex_unlock(&pool_mutex);
        run_task(&current_task, pool_size); // degraded: single-threaded
        return;
    }
    pthread_mutex_unlock(&pool_mutex);

    atomic_store_explicit(&next_tile, 0, memory_order_relaxed);
    if (current_task.tile_count <= 2) {
        run_task(&current_task, pool_size); // not worth waking the pool
        return;
    }
    atomic_store_explicit(&pool_active, pool_size, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool_generation, 1, memory_order_release);
    if (atomic_load_explicit(&pool_sleepers, memory_order_relaxed) > 0) {
        pthread_mutex_lock(&pool_mutex);
        pthread_cond_broadcast(&pool_start);
        pthread_mutex_unlock(&pool_mutex);
    }
    run_task(&current_task, pool_size);
    while (atomic_load_explicit(&pool_active, memory_order_acquire) > 0) {
        _mm_pause();
    }
}

static int ensure_xq_capacity(int sequence_length, int kblocks) {
    size_t need_xq = (size_t) sequence_length * kblocks * QK;
    size_t need_dx = (size_t) sequence_length * kblocks * sizeof(float);
    if (xq_cap < need_xq) {
        free(xq_buf);
        xq_buf = (uint8_t *) aligned_alloc(64, (need_xq + 63) & ~(size_t) 63);
        xq_cap = xq_buf != NULL ? need_xq : 0;
    }
    if (dx_cap < need_dx) {
        free(dx_buf);
        dx_buf = (float *) aligned_alloc(64, (need_dx + 63) & ~(size_t) 63);
        dx_cap = dx_buf != NULL ? need_dx : 0;
    }
    return xq_buf != NULL && dx_buf != NULL;
}

// K-quant gemms additionally need the per-16-element activation sums.
static int ensure_kquant_capacity(int sequence_length, int kblocks) {
    if (!ensure_xq_capacity(sequence_length, kblocks)) {
        return 0;
    }
    size_t need = (size_t) sequence_length * kblocks * 2 * sizeof(float);
    if (xsum_cap < need) {
        free(xsum_buf);
        xsum_buf = (float *) aligned_alloc(64, (need + 63) & ~(size_t) 63);
        xsum_cap = xsum_buf != NULL ? need : 0;
    }
    return xsum_buf != NULL;
}

// Shared driver for the K-quant gemm entries: quantize activations (s8 + sums), then bands.
static void run_kquant_gemm(int band_kind, jlong weights, jlong x, jlong x_stride_bytes,
                            jlong out, jlong out_stride_bytes,
                            jint sequence_length, jint dim0, jint dim1) {
    int kblocks = dim1 / QK;
    if (!ensure_kquant_capacity(sequence_length, kblocks)) {
        return;
    }
    current_task.weights = (const uint8_t *) (uintptr_t) weights;
    current_task.rhs = (const float *) (uintptr_t) x;
    current_task.out = (float *) (uintptr_t) out;
    current_task.that_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    current_task.out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    current_task.sequence_length = sequence_length;
    current_task.dim0 = dim0;
    current_task.dim1 = dim1;
    current_task.kind = 4;
    current_task.tile_count = (sequence_length + 7) / 8;
    dispatch_task();
    current_task.kind = band_kind;
    current_task.tile_count = (dim0 + VNNI_BAND - 1) / VNNI_BAND;
    dispatch_task();
}

// Dense (f32/f16/bf16 weight @ f32) gemm on the worker pool; tiled over rows x seq.
static void run_dense_gemm(int dtype, jlong weights, jlong x, jlong x_stride_bytes,
                           jlong out, jlong out_stride_bytes, jint sequence_length, jint dim0, jint dim1) {
    const int row_tile = 16, seq_tile = 64;
    int seq_tile_count = (sequence_length + seq_tile - 1) / seq_tile;
    int row_tile_count = (dim0 + row_tile - 1) / row_tile;
    int tile_count = seq_tile_count * row_tile_count;
    if (tile_count <= 0) return;
    current_task.weights = (const uint8_t *) (uintptr_t) weights;
    current_task.rhs = (const float *) (uintptr_t) x;
    current_task.out = (float *) (uintptr_t) out;
    current_task.that_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    current_task.out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    current_task.sequence_length = sequence_length;
    current_task.dim0 = dim0;
    current_task.dim1 = dim1;
    current_task.dtype = dtype;
    current_task.row_tile = row_tile;
    current_task.seq_tile = seq_tile;
    current_task.seq_tile_count = seq_tile_count;
    current_task.kind = 7;
    current_task.tile_count = tile_count;
    dispatch_task();
}

#endif // LFM25_AVX512_VNNI

#if LFM25_AVX2_FMA

static inline float avx2_fp16_to_f32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

static inline __m256 avx2_q8_8(const uint8_t *q, __m256 scale) {
    __m128i bytes = _mm_loadl_epi64((const __m128i *) q);
    __m256i i32 = _mm256_cvtepi8_epi32(bytes);
    return _mm256_mul_ps(_mm256_cvtepi32_ps(i32), scale);
}

static inline float avx2_hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static float avx2_q8_dot(const uint8_t *w, const float *x, int kblocks) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (int b = 0; b < kblocks; b++, w += BLOCK_BYTES, x += QK) {
        __m256 scale = _mm256_set1_ps(avx2_fp16_to_f32(*(const uint16_t *) w));
        acc0 = _mm256_fmadd_ps(avx2_q8_8(w + 2, scale), _mm256_loadu_ps(x), acc0);
        acc1 = _mm256_fmadd_ps(avx2_q8_8(w + 10, scale), _mm256_loadu_ps(x + 8), acc1);
        acc2 = _mm256_fmadd_ps(avx2_q8_8(w + 18, scale), _mm256_loadu_ps(x + 16), acc2);
        acc3 = _mm256_fmadd_ps(avx2_q8_8(w + 26, scale), _mm256_loadu_ps(x + 24), acc3);
    }
    return avx2_hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
}

static void avx2_native_gemm(jlong weights, jlong x, jlong x_stride_bytes,
                             jlong out, jlong out_stride_bytes,
                             jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * BLOCK_BYTES;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = avx2_q8_dot(wbase + (int64_t) row * w_stride, row_x, kblocks);
        }
    }
}

static void avx2_native_gemv(jlong weights, jlong x, jlong out, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * BLOCK_BYTES;
    for (int row = 0; row < dim0; row++) {
        dst[row] = avx2_q8_dot(wbase + (int64_t) row * w_stride, rhs, kblocks);
    }
}

// Q4_0 AVX2: 1 fp16 scale + 16 nibble-pair bytes -> 32 i4 values (bias -8) over 32 f32 activations
static float avx2_q4_0_dot(const uint8_t *w, const float *x, int kblocks) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    const __m256i eight = _mm256_set1_epi32(8);
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    for (int b = 0; b < kblocks; b++, w += 18, x += QK) {
        __m256 d = _mm256_set1_ps(avx2_fp16_to_f32(*(const uint16_t *) w));
        __m128i packed = _mm_loadu_si128((const __m128i *) (w + 2));
        __m128i lo = _mm_and_si128(packed, lo_mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);
        __m256i q0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(lo), eight);
        __m256i q1 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(lo, 8)), eight);
        __m256i q2 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(hi), eight);
        __m256i q3 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(hi, 8)), eight);
        acc0 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q0), d), _mm256_loadu_ps(x), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q1), d), _mm256_loadu_ps(x + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q2), d), _mm256_loadu_ps(x + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q3), d), _mm256_loadu_ps(x + 24), acc3);
    }
    return avx2_hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
}

static void avx2_gemm_q4_0(jlong weights, jlong x, jlong x_stride_bytes,
                           jlong out, jlong out_stride_bytes,
                           jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * 18;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = avx2_q4_0_dot(wbase + (int64_t) row * w_stride, row_x, kblocks);
        }
    }
}

// F16 AVX2: packed f16 weights, convert to f32 inline, 4x8 accumulator lanes + scalar tail
static float avx2_f16_dot(const uint8_t *w, const float *x, int dim1) {
    const uint16_t *wh = (const uint16_t *) w;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < dim1; i += 32, wh += 32, x += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) wh)), _mm256_loadu_ps(x), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (wh + 8))), _mm256_loadu_ps(x + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (wh + 16))), _mm256_loadu_ps(x + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (wh + 24))), _mm256_loadu_ps(x + 24), acc3);
    }
    float sum = avx2_hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < dim1; i++) sum += c_fp16_to_f32(wh[i]) * x[i];
    return sum;
}

// BF16 AVX2: packed bf16 weights, shift-left-16 to f32, 4x8 accumulator lanes + scalar tail
static float avx2_bf16_dot(const uint8_t *w, const float *x, int dim1) {
    const uint16_t *wh = (const uint16_t *) w;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < dim1; i += 32, wh += 32, x += 32) {
        __m256i b0 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) wh));
        __m256i b1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) (wh + 8)));
        __m256i b2 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) (wh + 16)));
        __m256i b3 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) (wh + 24)));
        acc0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(b0, 16)), _mm256_loadu_ps(x), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(b1, 16)), _mm256_loadu_ps(x + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(b2, 16)), _mm256_loadu_ps(x + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(b3, 16)), _mm256_loadu_ps(x + 24), acc3);
    }
    float sum = avx2_hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < dim1; i++) {
        union { uint32_t bits; float f; } u = {(uint32_t) wh[i] << 16};
        sum += u.f * x[i];
    }
    return sum;
}

// F32 AVX2: direct fma over f32 weights, 4x8 accumulator lanes + scalar tail
static float avx2_f32_dot(const uint8_t *w, const float *x, int dim1) {
    const float *wf = (const float *) w;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < dim1; i += 32, wf += 32, x += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wf), _mm256_loadu_ps(x), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(wf + 8), _mm256_loadu_ps(x + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(wf + 16), _mm256_loadu_ps(x + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(wf + 24), _mm256_loadu_ps(x + 24), acc3);
    }
    float sum = avx2_hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < dim1; i++) sum += wf[i] * x[i];
    return sum;
}

// Q4_K AVX2: per-super-block decoded scale/min cache, 4 groups x 64 elements,
// 4x8 accumulator lanes accumulating lo+hi sub-blocks into the same lanes.
static inline void avx2_q4k_decode_scales(const uint8_t *b, float *sc, float *mn, float d, float dmin) {
    for (int j = 0; j < 4; j++) {
        sc[j] = d * (b[j] & 63);
        mn[j] = dmin * (b[j + 4] & 63);
        sc[j + 4] = d * (float) ((b[j + 8] & 0xF) | ((b[j] >> 6) << 4));
        mn[j + 4] = dmin * (float) ((b[j + 8] >> 4) | ((b[j + 4] >> 6) << 4));
    }
}

static float avx2_q4k_dot(const uint8_t *w, const float *x, int blocks) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    for (int b = 0; b < blocks; b++, w += Q4K_BLOCK_BYTES, x += QKK) {
        float d = avx2_fp16_to_f32(*(const uint16_t *) w);
        float dmin = avx2_fp16_to_f32(*(const uint16_t *) (w + 2));
        float sc[8], mn[8];
        avx2_q4k_decode_scales(w + 4, sc, mn, d, dmin);
        const uint8_t *qs = w + 16;
        for (int g = 0; g < 4; g++) {
            __m256 sLo = _mm256_set1_ps(sc[g * 2]);
            __m256 sHi = _mm256_set1_ps(sc[g * 2 + 1]);
            __m256 mLo = _mm256_set1_ps(mn[g * 2]);
            __m256 mHi = _mm256_set1_ps(mn[g * 2 + 1]);
            __m128i p0 = _mm_loadu_si128((const __m128i *) (qs + g * 32));
            __m128i p1 = _mm_loadu_si128((const __m128i *) (qs + g * 32 + 16));
            __m128i lo0 = _mm_and_si128(p0, lo_mask);
            __m128i lo1 = _mm_and_si128(p1, lo_mask);
            __m128i hi0 = _mm_and_si128(_mm_srli_epi16(p0, 4), lo_mask);
            __m128i hi1 = _mm_and_si128(_mm_srli_epi16(p1, 4), lo_mask);
            __m256i ql0 = _mm256_cvtepu8_epi32(lo0);
            __m256i ql1 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(lo0, 8));
            __m256i ql2 = _mm256_cvtepu8_epi32(lo1);
            __m256i ql3 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(lo1, 8));
            __m256i qh0 = _mm256_cvtepu8_epi32(hi0);
            __m256i qh1 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(hi0, 8));
            __m256i qh2 = _mm256_cvtepu8_epi32(hi1);
            __m256i qh3 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(hi1, 8));
            const float *xp = x + g * 64;
            // lo sub-block (32 elements): ql* -> x[0..31]
            acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(ql0), sLo, mLo), _mm256_loadu_ps(xp), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(ql1), sLo, mLo), _mm256_loadu_ps(xp + 8), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(ql2), sLo, mLo), _mm256_loadu_ps(xp + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(ql3), sLo, mLo), _mm256_loadu_ps(xp + 24), acc3);
            // hi sub-block (32 elements): qh* -> x[32..63]
            acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(qh0), sHi, mHi), _mm256_loadu_ps(xp + 32), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(qh1), sHi, mHi), _mm256_loadu_ps(xp + 40), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(qh2), sHi, mHi), _mm256_loadu_ps(xp + 48), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(qh3), sHi, mHi), _mm256_loadu_ps(xp + 56), acc3);
        }
    }
    return avx2_hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
}

// Q5_K AVX2: 5-bit quant, identical structure to Q4_K but extra bit from qh[32] per group.
// Each qh byte holds 8 bits (2 per group), extracted by position 2*g (lo) and 2*g+1 (hi).
// qh is a SINGLE 32-byte array shared across all 4 groups, NOT per-group.
static float avx2_q5k_dot(const uint8_t *w, const float *x, int blocks) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    for (int b = 0; b < blocks; b++, w += Q5K_BLOCK_BYTES, x += QKK) {
        float d = avx2_fp16_to_f32(*(const uint16_t *) w);
        float dmin = avx2_fp16_to_f32(*(const uint16_t *) (w + 2));
        float sc_f[8], mn_f[8];
        avx2_q4k_decode_scales(w + 4, sc_f, mn_f, d, dmin);
        const uint8_t *qh = w + 16;
        const uint8_t *qs = w + 48;
        for (int g = 0; g < 4; g++) {
            __m256 sLo = _mm256_set1_ps(sc_f[g * 2]);
            __m256 sHi = _mm256_set1_ps(sc_f[g * 2 + 1]);
            __m256 mLo = _mm256_set1_ps(mn_f[g * 2]);
            __m256 mHi = _mm256_set1_ps(mn_f[g * 2 + 1]);
            __m128i p0 = _mm_loadu_si128((const __m128i *) (qs + g * 32));
            __m128i p1 = _mm_loadu_si128((const __m128i *) (qs + g * 32 + 16));
            __m128i qh0 = _mm_loadu_si128((const __m128i *) qh);
            __m128i qh1 = _mm_loadu_si128((const __m128i *) (qh + 16));
            __m128i lo0 = _mm_and_si128(p0, lo_mask);
            __m128i lo1 = _mm_and_si128(p1, lo_mask);
            __m128i hi0 = _mm_and_si128(_mm_srli_epi16(p0, 4), lo_mask);
            __m128i hi1 = _mm_and_si128(_mm_srli_epi16(p1, 4), lo_mask);
            // 5th bit extraction: shift qh bytes right by (2*g) for lo, (2*g+1) for hi
            __m128i bit_lo0 = _mm_and_si128(_mm_srli_epi16(qh0, 2 * g), _mm_set1_epi8(1));
            __m128i bit_lo1 = _mm_and_si128(_mm_srli_epi16(qh1, 2 * g), _mm_set1_epi8(1));
            __m128i bit_hi0 = _mm_and_si128(_mm_srli_epi16(qh0, 2 * g + 1), _mm_set1_epi8(1));
            __m128i bit_hi1 = _mm_and_si128(_mm_srli_epi16(qh1, 2 * g + 1), _mm_set1_epi8(1));
            __m128i lo5_0 = _mm_or_si128(lo0, _mm_slli_epi32(bit_lo0, 4));
            __m128i lo5_1 = _mm_or_si128(lo1, _mm_slli_epi32(bit_lo1, 4));
            __m128i hi5_0 = _mm_or_si128(hi0, _mm_slli_epi32(bit_hi0, 4));
            __m128i hi5_1 = _mm_or_si128(hi1, _mm_slli_epi32(bit_hi1, 4));
            __m256i ql0 = _mm256_cvtepu8_epi32(lo5_0);
            __m256i ql1 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(lo5_0, 8));
            __m256i ql2 = _mm256_cvtepu8_epi32(lo5_1);
            __m256i ql3 = _mm256_cvtepu8_epi32(_mm_bsrli_si128(lo5_1, 8));
            __m256i qh0i = _mm256_cvtepu8_epi32(hi5_0);
            __m256i qh1i = _mm256_cvtepu8_epi32(_mm_bsrli_si128(hi5_0, 8));
            __m256i qh2i = _mm256_cvtepu8_epi32(hi5_1);
            __m256i qh3i = _mm256_cvtepu8_epi32(_mm_bsrli_si128(hi5_1, 8));
            const float *xp = x + g * 64;
            acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(ql0), sLo, mLo), _mm256_loadu_ps(xp), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(ql1), sLo, mLo), _mm256_loadu_ps(xp + 8), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(ql2), sLo, mLo), _mm256_loadu_ps(xp + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(ql3), sLo, mLo), _mm256_loadu_ps(xp + 24), acc3);
            acc0 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(qh0i), sHi, mHi), _mm256_loadu_ps(xp + 32), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(qh1i), sHi, mHi), _mm256_loadu_ps(xp + 40), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(qh2i), sHi, mHi), _mm256_loadu_ps(xp + 48), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_fmsub_ps(_mm256_cvtepi32_ps(qh3i), sHi, mHi), _mm256_loadu_ps(xp + 56), acc3);
        }
    }
    return avx2_hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
}

// Q6_K AVX2: 6-bit values, 16-element sub-blocks with per-sub scale,
// 4 subs per half x 2 halves = 256 elements per super-block.
// ql[128] holds 4-bit nibbles, qh[64] holds high 2 bits (4 entries per byte).
// Each Q6 value is biased by -32 (integer domain subtraction).
static float avx2_q6k_dot(const uint8_t *w, const float *x, int blocks) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m256i bias32 = _mm256_set1_epi32(32);
    for (int b = 0; b < blocks; b++, w += Q6K_BLOCK_BYTES, x += QKK) {
        float d = avx2_fp16_to_f32(*(const uint16_t *) (w + 208));
        const uint8_t *ql = w;
        const uint8_t *qh_data = w + 128;
        const int8_t *sc = (const int8_t *) (w + 192);
        for (int half = 0; half < 2; half++) {
            const uint8_t *qlb = ql + half * 64;
            const uint8_t *qhb = qh_data + half * 32;
            for (int sub = 0; sub < 4; sub++) {
                float ws0 = d * sc[half * 8 + sub * 2];
                float ws1 = d * sc[half * 8 + sub * 2 + 1];
                __m256 s0 = _mm256_set1_ps(ws0);
                __m256 s1 = _mm256_set1_ps(ws1);
                int ql_off = (sub < 2 ? sub * 32 : (sub - 2) * 32);
                int use_hi = (sub >= 2);
                __m128i ql_lo = _mm_loadu_si128((const __m128i *) (qlb + ql_off));
                __m128i ql_hi = _mm_loadu_si128((const __m128i *) (qlb + ql_off + 16));
                __m128i qh_lo = _mm_loadu_si128((const __m128i *) qhb);
                __m128i qh_hi = _mm_loadu_si128((const __m128i *) (qhb + 16));
                __m128i nibs0, nibs1;
                if (use_hi) {
                    nibs0 = _mm_and_si128(_mm_srli_epi16(ql_lo, 4), lo_mask);
                    nibs1 = _mm_and_si128(_mm_srli_epi16(ql_hi, 4), lo_mask);
                } else {
                    nibs0 = _mm_and_si128(ql_lo, lo_mask);
                    nibs1 = _mm_and_si128(ql_hi, lo_mask);
                }
                __m128i qh_bits0 = _mm_and_si128(_mm_srli_epi16(qh_lo, sub * 2), _mm_set1_epi8(3));
                __m128i qh_bits1 = _mm_and_si128(_mm_srli_epi16(qh_hi, sub * 2), _mm_set1_epi8(3));
                __m128i q6_0 = _mm_or_si128(nibs0, _mm_slli_epi32(qh_bits0, 4));
                __m128i q6_1 = _mm_or_si128(nibs1, _mm_slli_epi32(qh_bits1, 4));
                __m256i q_l0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q6_0), bias32);
                __m256i q_l1 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(q6_0, 8)), bias32);
                __m256i q_l2 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q6_1), bias32);
                __m256i q_l3 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_bsrli_si128(q6_1, 8)), bias32);
                const float *xp = x + half * 128 + sub * 32;
                acc0 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q_l0), s0), _mm256_loadu_ps(xp), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q_l1), s0), _mm256_loadu_ps(xp + 8), acc1);
                acc2 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q_l2), s1), _mm256_loadu_ps(xp + 16), acc2);
                acc3 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q_l3), s1), _mm256_loadu_ps(xp + 24), acc3);
            }
        }
    }
    return avx2_hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
}

// AVX2 GEMM driver wrappers: row-major iteration calling the per-row AVX2 dot kernels.
static void avx2_gemm_f16(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int64_t w_stride = (int64_t) dim1 * 2;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = avx2_f16_dot(wbase + (int64_t) row * w_stride, row_x, dim1);
        }
    }
}

static void avx2_gemm_bf16(jlong weights, jlong x, jlong x_stride_bytes,
                           jlong out, jlong out_stride_bytes,
                           jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int64_t w_stride = (int64_t) dim1 * 2;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = avx2_bf16_dot(wbase + (int64_t) row * w_stride, row_x, dim1);
        }
    }
}

static void avx2_gemm_f32(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int64_t w_stride = (int64_t) dim1 * 4;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = avx2_f32_dot(wbase + (int64_t) row * w_stride, row_x, dim1);
        }
    }
}

static void avx2_gemm_q4k(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int blocks = dim1 / QKK;
    int64_t w_stride = (int64_t) blocks * Q4K_BLOCK_BYTES;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = avx2_q4k_dot(wbase + (int64_t) row * w_stride, row_x, blocks);
        }
    }
}

static void avx2_gemm_q5k(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int blocks = dim1 / QKK;
    int64_t w_stride = (int64_t) blocks * Q5K_BLOCK_BYTES;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = avx2_q5k_dot(wbase + (int64_t) row * w_stride, row_x, blocks);
        }
    }
}

static void avx2_gemm_q6k(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int blocks = dim1 / QKK;
    int64_t w_stride = (int64_t) blocks * Q6K_BLOCK_BYTES;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = avx2_q6k_dot(wbase + (int64_t) row * w_stride, row_x, blocks);
        }
    }
}

#endif // LFM25_AVX2_FMA

#if LFM25_ARM_NEON

// NEON fp16-to-f32: use hardware vcvt if available, otherwise scalar fallback.
#if LFM25_ARM_FP16
static inline float neon_fp16_to_f32(float16_t h) {
    return (float) h;
}
#else
static inline float neon_fp16_to_f32(uint16_t h) {
    return c_fp16_to_f32(h);
}
#endif

static inline float neon_hsum(float32x4_t v) {
    return vaddvq_f32(v);
}

// Q8_0 NEON: fp16 scale + 32 int8 quants per block, 4 accumulator lanes x 4 registers
static float neon_q8_0_dot(const uint8_t *w, const float *x, int kblocks) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    float32x4_t a4 = vdupq_n_f32(0), a5 = vdupq_n_f32(0), a6 = vdupq_n_f32(0), a7 = vdupq_n_f32(0);
    for (int b = 0; b < kblocks; b++, w += BLOCK_BYTES, x += QK) {
        float d = neon_fp16_to_f32(*(const uint16_t *) w);
        float32x4_t s = vdupq_n_f32(d);
        const int8_t *q = (const int8_t *) (w + 2);
        int8x16_t qv0 = vld1q_s8(q);
        int8x16_t qv1 = vld1q_s8(q + 16);
        int16x8_t qs0 = vmovl_s8(vget_low_s8(qv0));
        int16x8_t qs1 = vmovl_s8(vget_high_s8(qv0));
        int16x8_t qs2 = vmovl_s8(vget_low_s8(qv1));
        int16x8_t qs3 = vmovl_s8(vget_high_s8(qv1));
        float32x4_t wf0 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs0))));
        float32x4_t wf1 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs0))));
        float32x4_t wf2 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs1))));
        float32x4_t wf3 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs1))));
        float32x4_t wf4 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs2))));
        float32x4_t wf5 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs2))));
        float32x4_t wf6 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs3))));
        float32x4_t wf7 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs3))));
        a0 = vfmaq_f32(a0, wf0, vld1q_f32(x));
        a1 = vfmaq_f32(a1, wf1, vld1q_f32(x + 4));
        a2 = vfmaq_f32(a2, wf2, vld1q_f32(x + 8));
        a3 = vfmaq_f32(a3, wf3, vld1q_f32(x + 12));
        a4 = vfmaq_f32(a4, wf4, vld1q_f32(x + 16));
        a5 = vfmaq_f32(a5, wf5, vld1q_f32(x + 20));
        a6 = vfmaq_f32(a6, wf6, vld1q_f32(x + 24));
        a7 = vfmaq_f32(a7, wf7, vld1q_f32(x + 28));
    }
    float sum = neon_hsum(a0) + neon_hsum(a1) + neon_hsum(a2) + neon_hsum(a3)
              + neon_hsum(a4) + neon_hsum(a5) + neon_hsum(a6) + neon_hsum(a7);
    return sum;
}

// Q4_0 NEON: 1 fp16 scale + 16 nibble-pair bytes -> 32 i4 values (bias -8)
static float neon_q4_0_dot(const uint8_t *w, const float *x, int kblocks) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    float32x4_t a4 = vdupq_n_f32(0), a5 = vdupq_n_f32(0), a6 = vdupq_n_f32(0), a7 = vdupq_n_f32(0);
    const int8x16_t eight = vdupq_n_s8(8);
    for (int b = 0; b < kblocks; b++, w += 18, x += QK) {
        float d = neon_fp16_to_f32(*(const uint16_t *) w);
        float32x4_t s = vdupq_n_f32(d);
        uint8x16_t packed = vld1q_u8(w + 2);
        uint8x16_t lo = vandq_u8(packed, vdupq_n_u8(0x0F));
        uint8x16_t hi = vshrq_n_u8(packed, 4);
        // bias -8 in i8, then widen: 16 i8 -> 8 i16 -> 8 i32 (4 accumulators)
        int8x16_t ql = vsubq_s8(vreinterpretq_s8_u8(lo), eight);
        int8x16_t qh = vsubq_s8(vreinterpretq_s8_u8(hi), eight);
        int16x8_t ql_l = vmovl_s8(vget_low_s8(ql));
        int16x8_t ql_h = vmovl_s8(vget_high_s8(ql));
        int16x8_t qh_l = vmovl_s8(vget_low_s8(qh));
        int16x8_t qh_h = vmovl_s8(vget_high_s8(qh));
        float32x4_t w0 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(ql_l))));
        float32x4_t w1 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(ql_l))));
        float32x4_t w2 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(ql_h))));
        float32x4_t w3 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(ql_h))));
        float32x4_t w4 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qh_l))));
        float32x4_t w5 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qh_l))));
        float32x4_t w6 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qh_h))));
        float32x4_t w7 = vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qh_h))));
        a0 = vfmaq_f32(a0, w0, vld1q_f32(x));
        a1 = vfmaq_f32(a1, w1, vld1q_f32(x + 4));
        a2 = vfmaq_f32(a2, w2, vld1q_f32(x + 8));
        a3 = vfmaq_f32(a3, w3, vld1q_f32(x + 12));
        a4 = vfmaq_f32(a4, w4, vld1q_f32(x + 16));
        a5 = vfmaq_f32(a5, w5, vld1q_f32(x + 20));
        a6 = vfmaq_f32(a6, w6, vld1q_f32(x + 24));
        a7 = vfmaq_f32(a7, w7, vld1q_f32(x + 28));
    }
    float sum = neon_hsum(a0) + neon_hsum(a1) + neon_hsum(a2) + neon_hsum(a3)
              + neon_hsum(a4) + neon_hsum(a5) + neon_hsum(a6) + neon_hsum(a7);
    return sum;
}

// F16 NEON: packed f16 weights
static float neon_f16_dot(const uint8_t *w, const float *x, int dim1) {
    const uint16_t *wh = (const uint16_t *) w;
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    int i = 0;
#if LFM25_ARM_FP16
    for (; i + 15 < dim1; i += 16, wh += 16, x += 16) {
        float16x8_t f16_0 = vld1q_f16((const __fp16 *) wh);
        float16x8_t f16_1 = vld1q_f16((const __fp16 *) (wh + 8));
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(f16_0)), vld1q_f32(x));
        a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(f16_0)), vld1q_f32(x + 4));
        a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(f16_1)), vld1q_f32(x + 8));
        a3 = vfmaq_f32(a3, vcvt_f32_f16(vget_high_f16(f16_1)), vld1q_f32(x + 12));
    }
    float sum = neon_hsum(a0) + neon_hsum(a1) + neon_hsum(a2) + neon_hsum(a3);
#else
    for (; i + 15 < dim1; i += 16, wh += 16, x += 16) {
        // scalar fp16 conversion inline
        float w_f32[16];
        for (int j = 0; j < 16; j++) w_f32[j] = c_fp16_to_f32(wh[j]);
        a0 = vfmaq_f32(a0, vld1q_f32(w_f32), vld1q_f32(x));
        a1 = vfmaq_f32(a1, vld1q_f32(w_f32 + 4), vld1q_f32(x + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(w_f32 + 8), vld1q_f32(x + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(w_f32 + 12), vld1q_f32(x + 12));
    }
    float sum = neon_hsum(a0) + neon_hsum(a1) + neon_hsum(a2) + neon_hsum(a3);
#endif
    for (; i < dim1; i++) sum += c_fp16_to_f32(wh[i]) * x[i];
    return sum;
}

// BF16 NEON: shift-left-16 to f32
static float neon_bf16_dot(const uint8_t *w, const float *x, int dim1) {
    const uint16_t *wh = (const uint16_t *) w;
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 15 < dim1; i += 16, wh += 16, x += 16) {
        uint16x8_t bf0 = vld1q_u16(wh);
        uint16x8_t bf1 = vld1q_u16(wh + 8);
        uint32x4_t w32_0 = vshll_n_u16(vget_low_u16(bf0), 16);
        uint32x4_t w32_1 = vshll_n_u16(vget_high_u16(bf0), 16);
        uint32x4_t w32_2 = vshll_n_u16(vget_low_u16(bf1), 16);
        uint32x4_t w32_3 = vshll_n_u16(vget_high_u16(bf1), 16);
        a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(w32_0), vld1q_f32(x));
        a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(w32_1), vld1q_f32(x + 4));
        a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(w32_2), vld1q_f32(x + 8));
        a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(w32_3), vld1q_f32(x + 12));
    }
    float sum = neon_hsum(a0) + neon_hsum(a1) + neon_hsum(a2) + neon_hsum(a3);
    for (; i < dim1; i++) {
        union { uint32_t bits; float f; } u = {(uint32_t) wh[i] << 16};
        sum += u.f * x[i];
    }
    return sum;
}

// F32 NEON: direct fma
static float neon_f32_dot(const uint8_t *w, const float *x, int dim1) {
    const float *wf = (const float *) w;
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 15 < dim1; i += 16, wf += 16, x += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(wf), vld1q_f32(x));
        a1 = vfmaq_f32(a1, vld1q_f32(wf + 4), vld1q_f32(x + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(wf + 8), vld1q_f32(x + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(wf + 12), vld1q_f32(x + 12));
    }
    float sum = neon_hsum(a0) + neon_hsum(a1) + neon_hsum(a2) + neon_hsum(a3);
    for (; i < dim1; i++) sum += wf[i] * x[i];
    return sum;
}

// Q4_K NEON: decode 8 scale/min pairs, 4 groups x 64 elements,
// 8 accumulator lanes x 4 f32 each, shared across lo and hi sub-blocks.
static inline void neon_q4k_decode_scales(const uint8_t *b, float *sc, float *mn, float d, float dmin) {
    for (int j = 0; j < 4; j++) {
        sc[j] = d * (b[j] & 63);
        mn[j] = dmin * (b[j + 4] & 63);
        sc[j + 4] = d * ((b[j + 8] & 0xF) | ((b[j] >> 6) << 4));
        mn[j + 4] = dmin * ((b[j + 8] >> 4) | ((b[j + 4] >> 6) << 4));
    }
}

static float neon_q4k_dot(const uint8_t *w, const float *x, int blocks) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    float32x4_t a4 = vdupq_n_f32(0), a5 = vdupq_n_f32(0), a6 = vdupq_n_f32(0), a7 = vdupq_n_f32(0);
    for (int b = 0; b < blocks; b++, w += Q4K_BLOCK_BYTES, x += QKK) {
        float d = neon_fp16_to_f32(*(const uint16_t *) w);
        float dmin = neon_fp16_to_f32(*(const uint16_t *) (w + 2));
        float sc[8], mn[8];
        neon_q4k_decode_scales(w + 4, sc, mn, d, dmin);
        const uint8_t *qs = w + 16;
        for (int g = 0; g < 4; g++) {
            float32x4_t sLo = vdupq_n_f32(sc[g * 2]);
            float32x4_t sHi = vdupq_n_f32(sc[g * 2 + 1]);
            float32x4_t mLo = vdupq_n_f32(mn[g * 2]);
            float32x4_t mHi = vdupq_n_f32(mn[g * 2 + 1]);
            uint8x16_t p0 = vld1q_u8(qs + g * 32);
            uint8x16_t p1 = vld1q_u8(qs + g * 32 + 16);
            uint8x16_t lo0 = vandq_u8(p0, vdupq_n_u8(0x0F));
            uint8x16_t lo1 = vandq_u8(p1, vdupq_n_u8(0x0F));
            uint8x16_t hi0 = vshrq_n_u8(p0, 4);
            uint8x16_t hi1 = vshrq_n_u8(p1, 4);
            const float *xp = x + g * 64;
            // lo sub-block: nibble values from 2x16 bytes -> 32 f32 weight contributions
            #define NQK_LOOP(nibs, scv, mnv, xoff) do { \
                uint16x8_t w16_0 = vmovl_u8(vget_low_u8(nibs)); \
                uint16x8_t w16_1 = vmovl_u8(vget_high_u8(nibs)); \
                float32x4_t w0 = vsubq_f32(vmulq_f32(scv, vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_0)))), mnv); \
                float32x4_t w1 = vsubq_f32(vmulq_f32(scv, vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_0)))), mnv); \
                float32x4_t w2 = vsubq_f32(vmulq_f32(scv, vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_1)))), mnv); \
                float32x4_t w3 = vsubq_f32(vmulq_f32(scv, vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_1)))), mnv); \
                a0 = vfmaq_f32(a0, w0, vld1q_f32(xp + xoff)); \
                a1 = vfmaq_f32(a1, w1, vld1q_f32(xp + xoff + 4)); \
                a2 = vfmaq_f32(a2, w2, vld1q_f32(xp + xoff + 8)); \
                a3 = vfmaq_f32(a3, w3, vld1q_f32(xp + xoff + 12)); \
            } while(0)
            NQK_LOOP(lo0, sLo, mLo, 0);
            NQK_LOOP(lo1, sLo, mLo, 16);
            NQK_LOOP(hi0, sHi, mHi, 32);
            NQK_LOOP(hi1, sHi, mHi, 48);
            #undef NQK_LOOP
        }
    }
    return neon_hsum(a0) + neon_hsum(a1) + neon_hsum(a2) + neon_hsum(a3)
         + neon_hsum(a4) + neon_hsum(a5) + neon_hsum(a6) + neon_hsum(a7);
}

// Q5_K NEON: identical Q4_K structure but with 5th bit from shared qh[32]
static float neon_q5k_dot(const uint8_t *w, const float *x, int blocks) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    float32x4_t a4 = vdupq_n_f32(0), a5 = vdupq_n_f32(0), a6 = vdupq_n_f32(0), a7 = vdupq_n_f32(0);
    for (int b = 0; b < blocks; b++, w += Q5K_BLOCK_BYTES, x += QKK) {
        float d = neon_fp16_to_f32(*(const uint16_t *) w);
        float dmin = neon_fp16_to_f32(*(const uint16_t *) (w + 2));
        float sc[8], mn[8];
        neon_q4k_decode_scales(w + 4, sc, mn, d, dmin);
        const uint8_t *qh = w + 16;
        const uint8_t *qs = w + 48;
        for (int g = 0; g < 4; g++) {
            float32x4_t sLo = vdupq_n_f32(sc[g * 2]);
            float32x4_t sHi = vdupq_n_f32(sc[g * 2 + 1]);
            float32x4_t mLo = vdupq_n_f32(mn[g * 2]);
            float32x4_t mHi = vdupq_n_f32(mn[g * 2 + 1]);
            uint8x16_t p0 = vld1q_u8(qs + g * 32);
            uint8x16_t p1 = vld1q_u8(qs + g * 32 + 16);
            uint8x16_t qh0 = vld1q_u8(qh);
            uint8x16_t qh1 = vld1q_u8(qh + 16);
            uint8x16_t lo0 = vandq_u8(p0, vdupq_n_u8(0x0F));
            uint8x16_t lo1 = vandq_u8(p1, vdupq_n_u8(0x0F));
            uint8x16_t hi0 = vshrq_n_u8(p0, 4);
            uint8x16_t hi1 = vshrq_n_u8(p1, 4);
            // q5 bit extraction: bit at position (2*g) for lo, (2*g+1) for hi
            uint8x16_t bitlo0 = vandq_u8(vshrq_n_u8(qh0, 2 * g), vdupq_n_u8(1));
            uint8x16_t bitlo1 = vandq_u8(vshrq_n_u8(qh1, 2 * g), vdupq_n_u8(1));
            uint8x16_t bithi0 = vandq_u8(vshrq_n_u8(qh0, 2 * g + 1), vdupq_n_u8(1));
            uint8x16_t bithi1 = vandq_n_u8(vshrq_n_u8(qh1, 2 * g + 1), 1);
            // Combine: q5 = nibble | (bit << 4)
            uint8x16_t lo5_0 = vorrq_u8(lo0, vshlq_n_u8(bitlo0, 4));
            uint8x16_t lo5_1 = vorrq_u8(lo1, vshlq_n_u8(bitlo1, 4));
            uint8x16_t hi5_0 = vorrq_u8(hi0, vshlq_n_u8(bithi0, 4));
            uint8x16_t hi5_1 = vorrq_u8(hi1, vshlq_n_u8(bithi1, 4));
            const float *xp = x + g * 64;
            #define NQ5K_LOOP(nibs, scv, mnv, xoff) do { \
                uint16x8_t w16_0 = vmovl_u8(vget_low_u8(nibs)); \
                uint16x8_t w16_1 = vmovl_u8(vget_high_u8(nibs)); \
                float32x4_t w0 = vsubq_f32(vmulq_f32(scv, vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_0)))), mnv); \
                float32x4_t w1 = vsubq_f32(vmulq_f32(scv, vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_0)))), mnv); \
                float32x4_t w2 = vsubq_f32(vmulq_f32(scv, vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_1)))), mnv); \
                float32x4_t w3 = vsubq_f32(vmulq_f32(scv, vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_1)))), mnv); \
                a0 = vfmaq_f32(a0, w0, vld1q_f32(xp + xoff)); \
                a1 = vfmaq_f32(a1, w1, vld1q_f32(xp + xoff + 4)); \
                a2 = vfmaq_f32(a2, w2, vld1q_f32(xp + xoff + 8)); \
                a3 = vfmaq_f32(a3, w3, vld1q_f32(xp + xoff + 12)); \
            } while(0)
            NQ5K_LOOP(lo5_0, sLo, mLo, 0);
            NQ5K_LOOP(lo5_1, sLo, mLo, 16);
            NQ5K_LOOP(hi5_0, sHi, mHi, 32);
            NQ5K_LOOP(hi5_1, sHi, mHi, 48);
            #undef NQ5K_LOOP
        }
    }
    return neon_hsum(a0) + neon_hsum(a1) + neon_hsum(a2) + neon_hsum(a3)
         + neon_hsum(a4) + neon_hsum(a5) + neon_hsum(a6) + neon_hsum(a7);
}

// Q6_K NEON: 6-bit values with per-16-element scale, -32 bias
static float neon_q6k_dot(const uint8_t *w, const float *x, int blocks) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    float32x4_t a4 = vdupq_n_f32(0), a5 = vdupq_n_f32(0), a6 = vdupq_n_f32(0), a7 = vdupq_n_f32(0);
    const int8x16_t bias32 = vdupq_n_s8(32);
    for (int b = 0; b < blocks; b++, w += Q6K_BLOCK_BYTES, x += QKK) {
        float d = neon_fp16_to_f32(*(const uint16_t *) (w + 208));
        const uint8_t *ql = w;
        const uint8_t *qh_data = w + 128;
        const int8_t *sc = (const int8_t *) (w + 192);
        for (int half = 0; half < 2; half++) {
            const uint8_t *qb = ql + half * 64;
            const uint8_t *qhb = qh_data + half * 32;
            for (int sub = 0; sub < 4; sub++) {
                float ws0 = d * sc[half * 8 + sub * 2];
                float ws1 = d * sc[half * 8 + sub * 2 + 1];
                float32x4_t s0 = vdupq_n_f32(ws0);
                float32x4_t s1 = vdupq_n_f32(ws1);
                int ql_off = (sub < 2 ? sub * 32 : (sub - 2) * 32);
                int use_hi = (sub >= 2);
                uint8x16_t ql0 = vld1q_u8(qb + ql_off);
                uint8x16_t ql1 = vld1q_u8(qb + ql_off + 16);
                uint8x16_t qh0 = vld1q_u8(qhb);
                uint8x16_t qh1 = vld1q_u8(qhb + 16);
                uint8x16_t nibs0, nibs1;
                if (use_hi) {
                    nibs0 = vshrq_n_u8(ql0, 4);
                    nibs1 = vshrq_n_u8(ql1, 4);
                } else {
                    nibs0 = vandq_u8(ql0, vdupq_n_u8(0x0F));
                    nibs1 = vandq_u8(ql1, vdupq_n_u8(0x0F));
                }
                uint8x16_t qhbits0 = vandq_u8(vshrq_n_u8(qh0, sub * 2), vdupq_n_u8(3));
                uint8x16_t qhbits1 = vandq_u8(vshrq_n_u8(qh1, sub * 2), vdupq_n_u8(3));
                uint8x16_t q6_0 = vorrq_u8(nibs0, vshlq_n_u8(qhbits0, 4));
                uint8x16_t q6_1 = vorrq_u8(nibs1, vshlq_n_u8(qhbits1, 4));
                // bias -32: q6 value is 0..63, biased to -32..31
                int8x16_t q6s_0 = vsubq_s8(vreinterpretq_s8_u8(q6_0), bias32);
                int8x16_t q6s_1 = vsubq_s8(vreinterpretq_s8_u8(q6_1), bias32);
                const float *xp = x + half * 128 + sub * 32;
                #define NQ6K_LOOP(nibs, scv, xoff) do { \
                    int16x8_t w16_0 = vmovl_s8(vget_low_s8(nibs)); \
                    int16x8_t w16_1 = vmovl_s8(vget_high_s8(nibs)); \
                    float32x4_t w0 = vmulq_f32(scv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_0)))); \
                    float32x4_t w1 = vmulq_f32(scv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_0)))); \
                    float32x4_t w2 = vmulq_f32(scv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_1)))); \
                    float32x4_t w3 = vmulq_f32(scv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_1)))); \
                    a0 = vfmaq_f32(a0, w0, vld1q_f32(xp + xoff)); \
                    a1 = vfmaq_f32(a1, w1, vld1q_f32(xp + xoff + 4)); \
                    a2 = vfmaq_f32(a2, w2, vld1q_f32(xp + xoff + 8)); \
                    a3 = vfmaq_f32(a3, w3, vld1q_f32(xp + xoff + 12)); \
                } while(0)
                NQ6K_LOOP(q6s_0, s0, 0);
                NQ6K_LOOP(q6s_1, s1, 16);
                #undef NQ6K_LOOP
            }
        }
    }
    return neon_hsum(a0) + neon_hsum(a1) + neon_hsum(a2) + neon_hsum(a3)
         + neon_hsum(a4) + neon_hsum(a5) + neon_hsum(a6) + neon_hsum(a7);
}

// Generic NEON GEMM driver wrappers: iterate rows and call the per-row dot kernel.

static void neon_gemm_q8_0(jlong weights, jlong x, jlong x_stride_bytes,
                           jlong out, jlong out_stride_bytes,
                           jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * BLOCK_BYTES;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = neon_q8_0_dot(wbase + (int64_t) row * w_stride, row_x, kblocks);
        }
    }
}

static void neon_gemm_q4_0(jlong weights, jlong x, jlong x_stride_bytes,
                           jlong out, jlong out_stride_bytes,
                           jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * 18;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = neon_q4_0_dot(wbase + (int64_t) row * w_stride, row_x, kblocks);
        }
    }
}

static void neon_gemm_f16(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int64_t w_stride = (int64_t) dim1 * 2;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = neon_f16_dot(wbase + (int64_t) row * w_stride, row_x, dim1);
        }
    }
}

static void neon_gemm_bf16(jlong weights, jlong x, jlong x_stride_bytes,
                           jlong out, jlong out_stride_bytes,
                           jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int64_t w_stride = (int64_t) dim1 * 2;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = neon_bf16_dot(wbase + (int64_t) row * w_stride, row_x, dim1);
        }
    }
}

static void neon_gemm_f32(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int64_t w_stride = (int64_t) dim1 * 4;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = neon_f32_dot(wbase + (int64_t) row * w_stride, row_x, dim1);
        }
    }
}

static void neon_gemm_q4k(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int blocks = dim1 / QKK;
    int64_t w_stride = (int64_t) blocks * Q4K_BLOCK_BYTES;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = neon_q4k_dot(wbase + (int64_t) row * w_stride, row_x, blocks);
        }
    }
}

static void neon_gemm_q5k(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int blocks = dim1 / QKK;
    int64_t w_stride = (int64_t) blocks * Q5K_BLOCK_BYTES;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = neon_q5k_dot(wbase + (int64_t) row * w_stride, row_x, blocks);
        }
    }
}

static void neon_gemm_q6k(jlong weights, jlong x, jlong x_stride_bytes,
                          jlong out, jlong out_stride_bytes,
                          jint sequence_length, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int x_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
    int out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
    int blocks = dim1 / QKK;
    int64_t w_stride = (int64_t) blocks * Q6K_BLOCK_BYTES;
    for (int s = 0; s < sequence_length; s++) {
        const float *row_x = rhs + (int64_t) s * x_stride;
        float *row_out = dst + (int64_t) s * out_stride;
        for (int row = 0; row < dim0; row++) {
            row_out[row] = neon_q6k_dot(wbase + (int64_t) row * w_stride, row_x, blocks);
        }
    }
}

static void neon_native_gemv(jlong weights, jlong x, jlong out, jint dim0, jint dim1) {
    const uint8_t *wbase = (const uint8_t *) (uintptr_t) weights;
    const float *rhs = (const float *) (uintptr_t) x;
    float *dst = (float *) (uintptr_t) out;
    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * BLOCK_BYTES;
    for (int row = 0; row < dim0; row++) {
        dst[row] = neon_q8_0_dot(wbase + (int64_t) row * w_stride, rhs, kblocks);
    }
}

#endif // LFM25_ARM_NEON


// Direct JNI-mangled exports: the JVM binds native methods to these by symbol name when the
// .so is loaded, and a native image statically links them (LFM25StaticGemmFeature) — no
// JNI_OnLoad / RegisterNatives needed in either mode. The Java native methods live in
// com.llama4j.NativeKernels; renaming that class requires renaming these symbols.
//
// The boundary is pure pointers and bytes: `weights` already points at the first Q8_0 block
// of the operated row range, `x`/`out` at the first activation/output row, and the row
// strides are in BYTES (quant-layout and element-size math live on the Java side).
// Activations/outputs live in native MemorySegments, so no GetPrimitiveArrayCritical
// (no GC interaction at all on this path).

JNIEXPORT jint JNICALL Java_com_llama4j_NativeKernels_nativeCapabilities(JNIEnv *env, jclass cls) {
    (void) env;
    (void) cls;
    int caps = LFM25_CAP_Q8_0_GEMM | LFM25_CAP_Q4_0_GEMM | LFM25_CAP_Q4_K_GEMM | LFM25_CAP_Q5_K_GEMM | LFM25_CAP_Q6_K_GEMM
            | LFM25_CAP_BF16_GEMM | LFM25_CAP_F16_GEMM | LFM25_CAP_F32_GEMM;
#if LFM25_ARM_NEON
    caps |= LFM25_CAP_Q8_0_GEMV;
    return caps;
#endif
#if LFM25_AVX512_VNNI
    if (x86_supports_avx512_vnni()) {
        caps |= LFM25_CAP_Q8_0_GEMV;
        return caps;
    }
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        caps |= LFM25_CAP_Q8_0_GEMV;
        return caps;
    }
#endif
    return caps;
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemm(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1, jint row_tile, jint seq_tile) {
    (void) env;
    (void) cls;
#if LFM25_AVX512_VNNI
    if (x86_supports_avx512_vnni()) {
        if (row_tile < 4) row_tile = 4;
        if (seq_tile < 4) seq_tile = 4;
        int seq_tile_count = (sequence_length + seq_tile - 1) / seq_tile;
        int row_tile_count = (dim0 + row_tile - 1) / row_tile;
        int tile_count = seq_tile_count * row_tile_count;
        if (tile_count <= 0) {
            return;
        }
        int kblocks = dim1 / QK;
        current_task.weights = (const uint8_t *) (uintptr_t) weights;
        current_task.rhs = (const float *) (uintptr_t) x;
        current_task.out = (float *) (uintptr_t) out;
        current_task.that_stride = (int) (x_stride_bytes / (jlong) sizeof(float));
        current_task.out_stride = (int) (out_stride_bytes / (jlong) sizeof(float));
        current_task.sequence_length = sequence_length;
        current_task.dim0 = dim0;
        current_task.dim1 = dim1;
        current_task.row_tile = row_tile;
        current_task.seq_tile = seq_tile;
        current_task.seq_tile_count = seq_tile_count;
        if (sequence_length >= VNNI_MIN_SEQ && ensure_xq_capacity(sequence_length, kblocks)) {
            current_task.kind = 2;
            current_task.tile_count = (sequence_length + 7) / 8;
            dispatch_task();
            current_task.kind = 3;
            current_task.tile_count = (dim0 + VNNI_BAND - 1) / VNNI_BAND;
            dispatch_task();
        } else {
            current_task.kind = 0;
            current_task.tile_count = tile_count;
            dispatch_task();
        }
        return;
    }
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        avx2_native_gemm(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
    (void) row_tile; (void) seq_tile;
#if LFM25_ARM_NEON
    neon_gemm_q8_0(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
    return;
#endif
    c_gemm_q8_0(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemv(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong out, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
#if LFM25_ARM_NEON
    neon_native_gemv(weights, x, out, dim0, dim1);
    return;
#endif
#if LFM25_AVX512_VNNI
    if (x86_supports_avx512_vnni()) {
        int tile_count = (dim0 + GEMV_ROW_CHUNK - 1) / GEMV_ROW_CHUNK;
        if (tile_count <= 0) {
            return;
        }
        current_task.kind = 1;
        current_task.weights = (const uint8_t *) (uintptr_t) weights;
        current_task.rhs = (const float *) (uintptr_t) x;
        current_task.out = (float *) (uintptr_t) out;
        current_task.dim0 = dim0;
        current_task.dim1 = dim1;
        current_task.tile_count = tile_count;
        dispatch_task();
        return;
    }
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        avx2_native_gemv(weights, x, out, dim0, dim1);
        return;
    }
#endif
    (void) weights; (void) x; (void) out; (void) dim0; (void) dim1;
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemmQ40(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
#if LFM25_ARM_NEON
    neon_gemm_q4_0(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
    return;
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        avx2_gemm_q4_0(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
    c_gemm_q4_0(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemmQ4K(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
#if LFM25_ARM_NEON
    neon_gemm_q4k(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
    return;
#endif
#if LFM25_AVX512_VNNI
    if (x86_supports_avx512_vnni()) {
        run_kquant_gemm(5, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        avx2_gemm_q4k(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
    c_gemm_kquant(c_q4k_dot, Q4K_BLOCK_BYTES, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemmQ5K(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
#if LFM25_ARM_NEON
    neon_gemm_q5k(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
    return;
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        avx2_gemm_q5k(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
    c_gemm_kquant(c_q5k_dot, Q5K_BLOCK_BYTES, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemmQ6K(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
#if LFM25_ARM_NEON
    neon_gemm_q6k(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
    return;
#endif
#if LFM25_AVX512_VNNI
    if (x86_supports_avx512_vnni()) {
        run_kquant_gemm(6, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        avx2_gemm_q6k(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
    c_gemm_kquant(c_q6k_dot, Q6K_BLOCK_BYTES, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemmBF16(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
#if LFM25_ARM_NEON
    neon_gemm_bf16(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
    return;
#endif
#if LFM25_AVX512_VNNI
    if ((dim1 % 16) == 0 && x86_supports_avx512_vnni()) {
        run_dense_gemm(2, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        avx2_gemm_bf16(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
    c_gemm_dense(c_bf16_dot, 2, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemmF16(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
#if LFM25_ARM_NEON
    neon_gemm_f16(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
    return;
#endif
#if LFM25_AVX512_VNNI
    if ((dim1 % 16) == 0 && x86_supports_avx512_vnni()) {
        run_dense_gemm(1, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        avx2_gemm_f16(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
    c_gemm_dense(c_f16_dot, 2, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemmF32(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
#if LFM25_ARM_NEON
    neon_gemm_f32(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
    return;
#endif
#if LFM25_AVX512_VNNI
    if ((dim1 % 16) == 0 && x86_supports_avx512_vnni()) {
        run_dense_gemm(0, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
#if LFM25_AVX2_FMA
    if (x86_supports_avx2_fma_f16c()) {
        avx2_gemm_f32(weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
        return;
    }
#endif
    c_gemm_dense(c_f32_dot, 4, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    (void) vm;
    (void) reserved;
#if LFM25_AVX512_VNNI
    pthread_mutex_lock(&pool_mutex);
    destroy_pool_locked();
    pthread_mutex_unlock(&pool_mutex);
#endif
}
