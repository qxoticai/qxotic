package com.llm4j.jota.cuda;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.ScopedMemory;
import com.llm4j.jota.memory.ScopedMemoryAllocator;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUresult;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;

public final class CUDAScopedMemoryAllocator implements ScopedMemoryAllocator<CUdeviceptr> {

    public static final long ALLOC_ALIGNMENT = 256;

    private static final CUDAScopedMemoryAllocator INSTANCE = new CUDAScopedMemoryAllocator();

    public static ScopedMemoryAllocator<CUdeviceptr> instance() {
        return INSTANCE;
    }

    @Override
    public Device device() {
        return CUDAContext.CUDA;
    }

    @Override
    public ScopedMemory<CUdeviceptr> allocateMemory(long byteSize, long byteAlignment) {
        if (ALLOC_ALIGNMENT % byteAlignment != 0) {
            throw new UnsupportedOperationException();
        }
        CUdeviceptr devicePtr = new CUdeviceptr();
        int result = cuMemAlloc(devicePtr, byteSize);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to allocate CUDA memory: " + result);
        }
        return new CUDAScopedMemory(devicePtr, byteSize);
    }

    private static final class CUDAScopedMemory implements ScopedMemory<CUdeviceptr> {
        private final CUdeviceptr devicePtr;
        private final long byteSize;
        private boolean closed = false;

        CUDAScopedMemory(CUdeviceptr devicePtr, long byteSize) {
            this.devicePtr = devicePtr;
            this.byteSize = byteSize;
        }

        @Override
        public long byteSize() {
            return byteSize;
        }

        @Override
        public boolean isReadOnly() {
            return false;
        }

        @Override
        public Device device() {
            return CUDAContext.CUDA;
        }

        @Override
        public CUdeviceptr base() {
            return devicePtr;
        }

        @Override
        public void close() {
            if (!closed) {
                cuMemFree(devicePtr);
                closed = true;
            }
        }

        @Override
        public String toString() {
            return "CUDAScopedMemory{" +
                    "size=" + byteSize +
                    ", device=" + device() +
                    '}';
        }
    }
}