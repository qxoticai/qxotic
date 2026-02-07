#include <hip/hip_runtime.h>

extern "C" __global__ void vec_add(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

extern "C" __global__ void vec_sub(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] - b[idx];
  }
}

extern "C" __global__ void vec_mul(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] * b[idx];
  }
}

extern "C" __global__ void vec_div(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] / b[idx];
  }
}

extern "C" __global__ void vec_add_meta(
    const float *a, const float *b, float *out, const long *meta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  long n = meta[0];
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}
