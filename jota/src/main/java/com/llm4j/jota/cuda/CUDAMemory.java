package com.llm4j.jota.cuda;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.ScopedMemory;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;

public final class CUDAMemory implements ScopedMemory<CUdeviceptr> {

    private final CUdeviceptr pointer;
    private final long byteSize;

    CUDAMemory(CUdeviceptr pointer, long byteSize) {
        this.pointer = pointer;
        this.byteSize = byteSize;
    }

    @Override
    public long byteSize() {
        return this.byteSize;
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
        return pointer;
    }

    @Override
    public void close() {
        JCuda.cudaFree(this.pointer);
    }
}
