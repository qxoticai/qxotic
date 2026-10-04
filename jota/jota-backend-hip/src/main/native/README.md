HIP JNI library

This builds the JNI library behind `com.qxotic.jota.runtime.hip.HipRuntime`.
It links against `libamdhip64` and calls the HIP runtime directly (`hipMalloc`, module loading, kernel launches).

Build:

```bash
export JAVA_HOME=/path/to/jdk
export ROCM_PATH=/opt/rocm
cmake -B build && cmake --build build
```

The output is `build/libjota_hip.so`.
The Maven build of `jota-backend-hip` runs the same CMake project.

HSACO smoke test example

Create a file `vec_add.hip`:

```cpp
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
```

Compile to HSACO:

```bash
hipcc --genco -O2 vec_add.hip -o vec_add.hsaco
```

Run tests:

```bash
mvnd -f jota/pom.xml -pl jota-backend-hip -am -Phip test
```
