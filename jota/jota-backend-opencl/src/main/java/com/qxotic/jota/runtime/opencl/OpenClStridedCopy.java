package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

final class OpenClStridedCopy {

    private static final OpenClKernelBackend BACKEND = new OpenClKernelBackend();
    private static final ConcurrentMap<String, KernelExecutable> CACHE = new ConcurrentHashMap<>();

    private OpenClStridedCopy() {}

    static void copy(MemoryView<OpenClDevicePtr> src, MemoryView<OpenClDevicePtr> dst) {
        int rank = src.shape().flatRank();
        int elemBytes = (int) src.dataType().byteSize();
        long totalElements = src.shape().size();
        if (totalElements == 0) {
            return;
        }

        String cacheId = "opencl-strided-copy-r" + rank + "-e" + elemBytes;
        KernelExecutable exec =
                CACHE.computeIfAbsent(
                        cacheId,
                        id -> {
                            String source = generateKernel(rank, elemBytes);
                            KernelProgram program =
                                    KernelProgram.source("opencl", source, "strided_copy");
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
        ExecutionStream stream = new ExecutionStream(Device.OPENCL, 0L, true);
        exec.launch(config, args, stream);
    }

    private static String generateKernel(int rank, int elemBytes) {
        String copyType =
                switch (elemBytes) {
                    case 1 -> "uchar";
                    case 2 -> "ushort";
                    case 4 -> "uint";
                    case 8 -> "ulong";
                    default ->
                            throw new IllegalArgumentException(
                                    "Unsupported element size: " + elemBytes);
                };

        StringBuilder sb = new StringBuilder();
        sb.append("__kernel void strided_copy(\n");
        sb.append("    __global const uchar* src,\n");
        sb.append("    __global uchar* dst,\n");
        sb.append("    long src_offset,\n");
        sb.append("    long dst_offset,\n");
        sb.append("    long n");
        int slot = 5;
        for (int i = 0; i < rank; i++, slot++) {
            sb.append(",\n    long d").append(i);
        }
        for (int i = 0; i < rank; i++, slot++) {
            sb.append(",\n    long ss").append(i);
        }
        for (int i = 0; i < rank; i++, slot++) {
            sb.append(",\n    long ds").append(i);
        }
        sb.append(") {\n");
        sb.append("    long idx = (long)get_global_id(0);\n");
        sb.append("    if (idx >= n) return;\n");
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
        sb.append("    *((__global ")
                .append(copyType)
                .append("*)(dst + dst_off)) = *((const __global ")
                .append(copyType)
                .append("*)(src + src_off));\n");
        sb.append("}\n");
        return sb.toString();
    }
}
