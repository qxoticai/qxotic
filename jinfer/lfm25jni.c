// Native AVX-512 GEMM for Q8_0 weights @ F32 activations -> F32, bound by
// com.llama4j.NativeKernels (-Dllama.nativeGemmLib=<path> on the JVM, or statically
// linked into a native image with -Dllama.staticGemm=true).
//
// Kernel: 4 weight rows x 4 activation columns per register tile (16 zmm accumulators +
// 8 pre-scaled weight vectors), accumulating over the full K dimension so each output is
// reduced horizontally exactly once. Weights are decoded (sign-extend + fp16 scale) once
// per block and reused by all 4 columns.
//
// Threading: a persistent pthread pool (LFM25_NATIVE_THREADS, default: online CPUs) with
// an atomic tile counter for work stealing; tiles are rowTile x seqTile output blocks.
//
// Build: make libnative   (gcc -O3 -march=native -shared -fPIC -pthread)

#define _GNU_SOURCE

#include <jni.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#define QK 32           // elements per Q8_0 block
#define BLOCK_BYTES 34  // 2-byte fp16 scale + 32 int8 quants

#define QKK 256             // elements per K-quant super-block
#define Q4K_BLOCK_BYTES 144 // d(f16) dmin(f16) scales[12] qs[128]
#define Q6K_BLOCK_BYTES 210 // ql[128] qh[64] scales[16] d(f16)

#define GEMV_ROW_CHUNK 64
#define VNNI_BAND 32        // weight rows per parallel work unit (4 groups of 16)
#define VNNI_MIN_SEQ 8      // below this, activation quantization + repack don't amortize

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

static void compute_tile(const gemm_task_t *task, int tile, int worker) {
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

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemm(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1, jint row_tile, jint seq_tile) {
    (void) env;
    (void) cls;
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
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemv(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong out, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
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
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemmQ4K(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
    run_kquant_gemm(5, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL Java_com_llama4j_NativeKernels_nativeGemmQ6K(JNIEnv *env, jclass cls,
        jlong weights, jlong x, jlong x_stride_bytes, jlong out, jlong out_stride_bytes,
        jint sequence_length, jint dim0, jint dim1) {
    (void) env;
    (void) cls;
    run_kquant_gemm(6, weights, x, x_stride_bytes, out, out_stride_bytes, sequence_length, dim0, dim1);
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    (void) vm;
    (void) reserved;
    pthread_mutex_lock(&pool_mutex);
    destroy_pool_locked();
    pthread_mutex_unlock(&pool_mutex);
}
