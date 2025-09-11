package com.llm4j.jota.cuda;


import com.llm4j.jota.memory.Memory;
import com.llm4j.jota.memory.MemoryOperations;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUresult;

import java.lang.foreign.MemorySegment;

import static jcuda.driver.JCudaDriver.*;

public final class CUDAMemoryOperations implements MemoryOperations<CUdeviceptr> {

    private static final CUDAMemoryOperations INSTANCE = new CUDAMemoryOperations();

    public static MemoryOperations<CUdeviceptr> instance() {
        return INSTANCE;
    }

    private CUDAMemoryOperations() {
    }

    @Override
    public void copy(Memory<CUdeviceptr> src, long srcByteOffset,
                     Memory<CUdeviceptr> dst, long dstByteOffset,
                     long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        int result = cuMemcpy(
                dst.base().withByteOffset(dstByteOffset),
                src.base().withByteOffset(srcByteOffset),
                byteSize
        );
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA copy failed: " + result);
        }
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset,
                               Memory<CUdeviceptr> dst, long dstByteOffset,
                               long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        int result = cuMemcpyHtoD(
                dst.base().withByteOffset(dstByteOffset),
                Pointer.to(src.base().asSlice(srcByteOffset, byteSize).asByteBuffer()),
                byteSize
        );
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA copy from host failed: " + result);
        }
    }

    @Override
    public void copyToNative(Memory<CUdeviceptr> src, long srcByteOffset,
                             Memory<MemorySegment> dst, long dstByteOffset,
                             long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        int result = cuMemcpyDtoH(
                Pointer.to(dst.base().asSlice(dstByteOffset, byteSize).asByteBuffer()),
                src.base().withByteOffset(srcByteOffset),
                byteSize
        );
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA copy to host failed: " + result);
        }
    }

    @Override
    public void fillByte(Memory<CUdeviceptr> memory, long byteOffset, long byteSize, byte byteValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        int result = cuMemsetD8(
                memory.base().withByteOffset(byteOffset),
                byteValue,
                byteSize
        );
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA memset failed: " + result);
        }
    }

    @Override
    public void fillShort(Memory<CUdeviceptr> memory, long byteOffset, long byteSize, short shortValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        if (byteSize % 2 != 0) {
            throw new IllegalArgumentException("byteSize must be a multiple of 2 for short fill");
        }
        int result = cuMemsetD16(
                memory.base().withByteOffset(byteOffset),
                shortValue,
                byteSize / 2
        );
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA memset failed: " + result);
        }
    }

    @Override
    public void fillInt(Memory<CUdeviceptr> memory, long byteOffset, long byteSize, int intValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        if (byteSize % 4 != 0) {
            throw new IllegalArgumentException("byteSize must be a multiple of 4 for int fill");
        }
        int result = cuMemsetD32(
                memory.base().withByteOffset(byteOffset),
                intValue,
                byteSize / 4
        );
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA memset failed: " + result);
        }
    }

    @Override
    public void fillLong(Memory<CUdeviceptr> memory, long byteOffset, long byteSize, long longValue) {
        throw new UnsupportedOperationException("CUDA does not support 64-bit fill. A custom kernel is required.");
    }
}