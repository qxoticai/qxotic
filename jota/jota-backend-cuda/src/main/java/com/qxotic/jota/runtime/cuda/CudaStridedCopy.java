package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

final class CudaStridedCopy {

    private static final CudaKernelBackend BACKEND = new CudaKernelBackend();
    private static final ConcurrentMap<String, KernelExecutable> CACHE = new ConcurrentHashMap<>();

    private CudaStridedCopy() {}

    static void copy(MemoryView<CudaDevicePtr> src, MemoryView<CudaDevicePtr> dst) {
        int rank = src.shape().flatRank();
        int elemBytes = (int) src.dataType().byteSize();
        long totalElements = src.shape().size();
        if (totalElements == 0) {
            return;
        }

        String cacheId = "strided-copy-r" + rank + "-e" + elemBytes;
        KernelExecutable exec =
                CACHE.computeIfAbsent(
                        cacheId,
                        id -> {
                            String source = generateKernel(rank, elemBytes);
                            KernelProgram program =
                                    KernelProgram.source("cuda", source, "strided_copy");
                            return BACKEND.compile(program, KernelCacheKey.of(id));
                        });

        KernelArgs args = new KernelArgs();
        args.addBuffer(src);
        args.addBuffer(dst);
        args.addScalarBits(src.byteOffset(), DataType.I64);
        args.addScalarBits(dst.byteOffset(), DataType.I64);
        args.addScalarBits(totalElements, DataType.I64);

        long[] shapeDims = src.shape().toArray();
        long[] srcByteStrides = src.byteStride().toArray();
        long[] dstByteStrides = dst.byteStride().toArray();
        for (int i = 0; i < rank; i++) {
            args.addScalarBits(shapeDims[i], DataType.I64);
        }
        for (int i = 0; i < rank; i++) {
            args.addScalarBits(srcByteStrides[i], DataType.I64);
        }
        for (int i = 0; i < rank; i++) {
            args.addScalarBits(dstByteStrides[i], DataType.I64);
        }

        int threads = 256;
        int blocks = (int) ((totalElements + threads - 1) / threads);
        LaunchConfig config = LaunchConfig.grid(blocks).block(threads);
        ExecutionStream stream = new ExecutionStream(src.memory().device(), null, true);
        exec.launch(config, args, stream);
    }

    private static String generateKernel(int rank, int elemBytes) {
        String copyType =
                switch (elemBytes) {
                    case 1 -> "char";
                    case 2 -> "short";
                    case 4 -> "int";
                    case 8 -> "long long";
                    default ->
                            throw new IllegalArgumentException(
                                    "Unsupported element size: " + elemBytes);
                };
        StringBuilder sb = new StringBuilder();
        sb.append("#include <cuda_runtime.h>\n\n");
        sb.append("extern \"C\" __global__\n");
        sb.append("void strided_copy(\n");
        sb.append("    const char* __restrict__ src,\n");
        sb.append("    char* __restrict__ dst,\n");
        sb.append("    long src_offset, long dst_offset,\n");
        sb.append("    long n");
        for (int i = 0; i < rank; i++) {
            sb.append(",\n    long d").append(i);
        }
        for (int i = 0; i < rank; i++) {
            sb.append(",\n    long ss").append(i);
        }
        for (int i = 0; i < rank; i++) {
            sb.append(",\n    long ds").append(i);
        }
        sb.append(")\n{\n");
        sb.append("    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;\n");
        sb.append("    if (idx >= n) return;\n");
        // Decompose linear index into multi-dim coordinates (innermost first).
        for (int i = rank - 1; i >= 1; i--) {
            sb.append("    long i")
                    .append(i)
                    .append(" = idx % d")
                    .append(i)
                    .append("; idx /= d")
                    .append(i)
                    .append(";\n");
        }
        sb.append("    long i0 = idx;\n");
        // Compute byte offsets.
        sb.append("    long src_off = src_offset");
        for (int i = 0; i < rank; i++) {
            sb.append(" + i").append(i).append(" * ss").append(i);
        }
        sb.append(";\n");
        sb.append("    long dst_off = dst_offset");
        for (int i = 0; i < rank; i++) {
            sb.append(" + i").append(i).append(" * ds").append(i);
        }
        sb.append(";\n");
        sb.append("    *(")
                .append(copyType)
                .append("*)(dst + dst_off) = *(const ")
                .append(copyType)
                .append("*)(src + src_off);\n");
        sb.append("}\n");
        return sb.toString();
    }
}
