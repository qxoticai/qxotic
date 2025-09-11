/*
 * Copyright 2024 LLM4J
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Unary operators
#define OP_IDENTITY 0
#define OP_SQUARE 1
#define OP_EXP 2

// Binary operators
#define OP_SUM 0
#define OP_PRODUCT 1
#define OP_DIVIDE 2
#define OP_SUBTRACT 3
#define OP_MAX 4
#define OP_MIN 5

__device__ float apply_unary(float val, int op_id) {
    switch (op_id) {
        case OP_IDENTITY: return val;
        case OP_SQUARE: return val * val;
        case OP_EXP: return expf(val);
        default: return val; // Should not happen
    }
}

__device__ float apply_binary(float v1, float v2, int op_id) {
    switch (op_id) {
        case OP_SUM: return v1 + v2;
        case OP_PRODUCT: return v1 * v2;
        case OP_DIVIDE: return v1 / v2;
        case OP_SUBTRACT: return v1 - v2;
        case OP_MAX: return fmaxf(v1, v2);
        case OP_MIN: return fminf(v1, v2);
        default: return 0.0f; // Should not happen
    }
}

extern "C" __global__ void elementwise_unary_contiguous(const float* in, float* out, int n, int op_id) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = apply_unary(in[index], op_id);
    }
}

extern "C" __global__ void elementwise2_scalar_contiguous(const float* in, float scalar, float* out, int n, int op_id) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = apply_binary(in[index], scalar, op_id);
    }
}

extern "C" __global__ void elementwise2_contiguous(const float* left, const float* right, float* out, int n, int op_id) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = apply_binary(left[index], right[index], op_id);
    }
}

// Strided Kernels

__device__ void get_offsets(int index, int rank, const long* shape, const long* strides, long& offset) {
    int temp_index = index;
    for (int i = rank - 1; i >= 0; i--) {
        long coord = temp_index % shape[i];
        temp_index /= shape[i];
        if (shape[i] != 1) { // Handle broadcasting
            offset += coord * strides[i];
        }
    }
}

extern "C" __global__ void elementwise_unary_strided(
    const char* in_ptr, const long* in_shape, const long* in_strides,
    char* out_ptr, const long* out_shape, const long* out_strides,
    int n, int rank, int op_id) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    long in_offset = 0;
    long out_offset = 0;
    get_offsets(index, rank, out_shape, in_strides, in_offset);
    get_offsets(index, rank, out_shape, out_strides, out_offset);

    const float* in = (const float*)(in_ptr + in_offset);
    float* out = (float*)(out_ptr + out_offset);

    *out = apply_unary(*in, op_id);
}

extern "C" __global__ void elementwise2_scalar_strided(
    const char* in_ptr, const long* in_shape, const long* in_strides,
    float scalar,
    char* out_ptr, const long* out_shape, const long* out_strides,
    int n, int rank, int op_id) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    long in_offset = 0;
    long out_offset = 0;
    get_offsets(index, rank, out_shape, in_strides, in_offset);
    get_offsets(index, rank, out_shape, out_strides, out_offset);

    const float* in = (const float*)(in_ptr + in_offset);
    float* out = (float*)(out_ptr + out_offset);

    *out = apply_binary(*in, scalar, op_id);
}

extern "C" __global__ void elementwise2_strided(
    const char* left_ptr, const long* left_shape, const long* left_strides,
    const char* right_ptr, const long* right_shape, const long* right_strides,
    char* out_ptr, const long* out_shape, const long* out_strides,
    int n, int rank, int op_id) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    long left_offset = 0;
    long right_offset = 0;
    long out_offset = 0;
    get_offsets(index, rank, out_shape, left_strides, left_offset);
    get_offsets(index, rank, out_shape, right_strides, right_offset);
    get_offsets(index, rank, out_shape, out_strides, out_offset);

    const float* left = (const float*)(left_ptr + left_offset);
    const float* right = (const float*)(right_ptr + right_offset);
    float* out = (float*)(out_ptr + out_offset);

    *out = apply_binary(*left, *right, op_id);
}

// Reduction Kernels

extern "C" __global__ void fold_all_contiguous(
    const float* in, float* out, int n, float initial_value, int op_id) {

    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : initial_value;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = apply_binary(sdata[tid], sdata[tid + s], op_id);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void reduce_contiguous(
    const float* in, float* out, int n_out, int rank, const long* shape, int axis, int op_id) {

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n_out; i += blockDim.x * gridDim.x) {
        long in_idx = 0;
        long out_idx = i;
        long inner_stride = 1;
        for (int d = rank - 1; d > axis; --d) {
            in_idx += (out_idx % shape[d]) * inner_stride;
            out_idx /= shape[d];
            inner_stride *= shape[d];
        }
        long outer_stride = inner_stride * shape[axis];
        in_idx += (out_idx % shape[axis]) * inner_stride;
        out_idx /= shape[axis];
        in_idx += out_idx * outer_stride;

        float result = in[in_idx];
        in_idx += inner_stride;
        for (int j = 1; j < shape[axis]; ++j) {
            result = apply_binary(result, in[in_idx], op_id);
            in_idx += inner_stride;
        }
        out[i] = result;
    }
}

extern "C" __global__ void fill_with_unary_scalar_contiguous(float scalar_in, float* out, int n, int op_id) {
    float value = apply_unary(scalar_in, op_id);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = value;
    }
}

extern "C" __global__ void fill_with_unary_scalar_strided(
    float scalar_in,
    char* out_ptr, const long* out_shape, const long* out_strides,
    int n, int rank, int op_id) {

    float value = apply_unary(scalar_in, op_id);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    long out_offset = 0;
    get_offsets(index, rank, out_shape, out_strides, out_offset);

    float* out = (float*)(out_ptr + out_offset);
    *out = value;
}
