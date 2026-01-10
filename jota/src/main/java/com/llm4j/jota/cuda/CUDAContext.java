package com.llm4j.jota.cuda;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.*;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;

public class CUDAContext implements Context<CUdeviceptr> {

    public static final Device CUDA = Device.GPU.child("cuda");

    private final cublasHandle cublasHandle;

    public CUDAContext() {
        JCuda.initialize();
        JCudaDriver.cuInit(0);
        cublasHandle = new cublasHandle();
        JCublas2.cublasCreate(cublasHandle);
    }

    @Override
    public Device device() {
        return CUDA;
    }

    @Override
    public MemoryAllocator<CUdeviceptr> memoryAllocator() {
        return CUDAScopedMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<CUdeviceptr> memoryAccess() {
        return CUDAMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<CUdeviceptr> memoryOperations() {
        return CUDAMemoryOperations.instance();
    }

    @Override
    public FloatOperations<CUdeviceptr> floatOperations() {
        throw new UnsupportedOperationException(); // return CUDAFloatOperations.instance();
    }

    public cublasHandle getCublasHandle() {
        return cublasHandle;
    }

    @Override
    public void close() {
        JCublas2.cublasDestroy(cublasHandle);
    }
}
