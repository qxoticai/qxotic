// Metal compute kernels for LFM25 native GEMM backend.
// Target: Apple Silicon (M1+). Compile with:
//   xcrun -sdk macosx metal -c gemm.metal -o gemm.air
//   xcrun -sdk macosx metallib gemm.air -o gemm.metallib
//
// Dispatch dimensions: grid = (dim0, sequence_length), threadgroup = (TG_WIDTH, TG_HEIGHT).
// Each threadgroup processes TG_WIDTH x TG_HEIGHT output tiles.
//
// Threadgroup memory: shared tile of activations for cache reuse across rows.

#ifdef __METAL_VERSION__

#include <metal_stdlib>
using namespace metal;

#define QK 32
#define BLOCK_BYTES 34

// fp16 -> f32 conversion helper (Metal has native f16 support)
static inline float fp16_to_f32(half h) {
    return float(h);
}

// --- Q8_0 GEMM kernel --------------------------------------------------
// Each thread computes one output element: dot(w[row], x[col]) over all K-blocks.
// Weights are Q8_0 blocks: 2 bytes fp16 scale + 32 int8 quants per block.
// Threadgroup: TG x TG threads, each thread handles 1 output.
// Optimization: each threadgroup loads TG rows of the current K-block into threadgroup
// memory so that TG columns of activations are reused.

kernel void gemm_q8_0(
    device const uint8_t *weights [[buffer(0)]],
    device const float   *x       [[buffer(1)]],
    device float         *out     [[buffer(2)]],
    constant int         &dim0    [[buffer(3)]],
    constant int         &dim1    [[buffer(4)]],
    constant int         &seq_stride  [[buffer(5)]],
    constant int         &out_stride  [[buffer(6)]],
    uint                tid [[thread_position_in_threadgroup]],
    uint                gid [[thread_position_in_grid]],
    uint                tg_size [[threads_per_threadgroup]])
{
    int row = gid % dim0;
    int col = gid / dim0;
    if (row >= dim0 || col >= seq_stride) return;

    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * BLOCK_BYTES;
    const uint8_t *w = weights + (int64_t) row * w_stride;
    const float *xp = x + (int64_t) col * dim1;

    float acc = 0.0f;
    for (int b = 0; b < kblocks; b++, w += BLOCK_BYTES, xp += QK) {
        half d = *(const device half *) w;
        float scale = float(d);
        const device char *q = (const device char *) (w + 2);
        for (int i = 0; i < QK; i++) {
            acc += float(q[i]) * scale * xp[i];
        }
    }
    out[(int64_t) col * out_stride + row] = acc;
}

// Q4_0 layout: 18 bytes per block (2 fp16 scale + 16 nibble-pair bytes = 32 i4 values)
kernel void gemm_q4_0(
    device const uint8_t *weights [[buffer(0)]],
    device const float   *x       [[buffer(1)]],
    device float         *out     [[buffer(2)]],
    constant int         &dim0    [[buffer(3)]],
    constant int         &dim1    [[buffer(4)]],
    constant int         &seq_stride  [[buffer(5)]],
    constant int         &out_stride  [[buffer(6)]],
    uint                tid [[thread_position_in_threadgroup]],
    uint                gid [[thread_position_in_grid]])
{
    int row = gid % dim0;
    int col = gid / dim0;
    if (row >= dim0 || col >= seq_stride) return;

    int kblocks = dim1 / QK;
    int64_t w_stride = (int64_t) kblocks * 18;
    const uint8_t *w = weights + (int64_t) row * w_stride;
    const float *xp = x + (int64_t) col * dim1;

    float acc = 0.0f;
    for (int b = 0; b < kblocks; b++, w += 18, xp += QK) {
        half d = *(const device half *) w;
        float scale = float(d);
        const device uint8_t *qs = w + 2;
        for (int i = 0; i < 16; i++) {
            int lo = (qs[i] & 0x0F) - 8;
            int hi = ((qs[i] >> 4) & 0x0F) - 8;
            acc += float(lo) * scale * xp[i];
            acc += float(hi) * scale * xp[i + 16];
        }
    }
    out[(int64_t) col * out_stride + row] = acc;
}

// F16 GEMM: simple dense fp16 -> f32 dot product.
kernel void gemm_f16(
    device const half    *weights [[buffer(0)]],
    device const float   *x       [[buffer(1)]],
    device float         *out     [[buffer(2)]],
    constant int         &dim0    [[buffer(3)]],
    constant int         &dim1    [[buffer(4)]],
    constant int         &seq_stride  [[buffer(5)]],
    constant int         &out_stride  [[buffer(6)]],
    uint                tid [[thread_position_in_threadgroup]],
    uint                gid [[thread_position_in_grid]])
{
    int row = gid % dim0;
    int col = gid / dim0;
    if (row >= dim0 || col >= seq_stride) return;

    const device half *w = weights + (int64_t) row * dim1;
    const float *xp = x + (int64_t) col * dim1;

    float acc = 0.0f;
    for (int i = 0; i < dim1; i++) {
        acc += float(w[i]) * xp[i];
    }
    out[(int64_t) col * out_stride + row] = acc;
}

// F32 GEMM: direct f32 dot product.
kernel void gemm_f32(
    device const float   *weights [[buffer(0)]],
    device const float   *x       [[buffer(1)]],
    device float         *out     [[buffer(2)]],
    constant int         &dim0    [[buffer(3)]],
    constant int         &dim1    [[buffer(4)]],
    constant int         &seq_stride  [[buffer(5)]],
    constant int         &out_stride  [[buffer(6)]],
    uint                tid [[thread_position_in_threadgroup]],
    uint                gid [[thread_position_in_grid]])
{
    int row = gid % dim0;
    int col = gid / dim0;
    if (row >= dim0 || col >= seq_stride) return;

    const device float *w = weights + (int64_t) row * dim1;
    const float *xp = x + (int64_t) col * dim1;

    float acc = 0.0f;
    for (int i = 0; i < dim1; i++) {
        acc = fma(w[i], xp[i], acc);
    }
    out[(int64_t) col * out_stride + row] = acc;
}

#endif // __METAL_VERSION__
