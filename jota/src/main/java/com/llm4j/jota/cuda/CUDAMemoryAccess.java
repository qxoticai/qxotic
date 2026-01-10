package com.llm4j.jota.cuda;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

public class CUDAMemoryAccess implements MemoryAccess<CUdeviceptr> {

    private static final CUDAMemoryAccess INSTANCE = new CUDAMemoryAccess();

    public static MemoryAccess<CUdeviceptr> instance() {
        return INSTANCE;
    }

    private CUDAMemoryAccess() {}

    @Override
    public byte readByte(Memory<CUdeviceptr> memory, long byteOffset) {
        byte[] hostBuffer = new byte[1];
        int result = JCudaDriver.cuMemcpyDtoH(Pointer.to(hostBuffer), memory.base().withByteOffset(byteOffset), 1);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from device to host: " + CUresult.stringFor(result));
        }
        return hostBuffer[0];
    }

    @Override
    public void writeByte(Memory<CUdeviceptr> memory, long byteOffset, byte value) {
        byte[] hostBuffer = new byte[]{value};
        int result = JCudaDriver.cuMemcpyHtoD(memory.base().withByteOffset(byteOffset), Pointer.to(hostBuffer), 1);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from host to device: " + CUresult.stringFor(result));
        }
    }

    @Override
    public short readShort(Memory<CUdeviceptr> memory, long byteOffset) {
        short[] hostBuffer = new short[1];
        int result = JCudaDriver.cuMemcpyDtoH(Pointer.to(hostBuffer), memory.base().withByteOffset(byteOffset), 2);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from device to host: " + CUresult.stringFor(result));
        }
        return hostBuffer[0];
    }

    @Override
    public void writeShort(Memory<CUdeviceptr> memory, long byteOffset, short value) {
        short[] hostBuffer = new short[]{value};
        int result = JCudaDriver.cuMemcpyHtoD(memory.base().withByteOffset(byteOffset), Pointer.to(hostBuffer), 2);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from host to device: " + CUresult.stringFor(result));
        }
    }

    @Override
    public int readInt(Memory<CUdeviceptr> memory, long byteOffset) {
        int[] hostBuffer = new int[1];
        int result = JCudaDriver.cuMemcpyDtoH(Pointer.to(hostBuffer), memory.base().withByteOffset(byteOffset), 4);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from device to host: " + CUresult.stringFor(result));
        }
        return hostBuffer[0];
    }

    @Override
    public void writeInt(Memory<CUdeviceptr> memory, long byteOffset, int value) {
        int[] hostBuffer = new int[]{value};
        int result = JCudaDriver.cuMemcpyHtoD(memory.base().withByteOffset(byteOffset), Pointer.to(hostBuffer), 4);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from host to device: " + CUresult.stringFor(result));
        }
    }

    @Override
    public long readLong(Memory<CUdeviceptr> memory, long byteOffset) {
        long[] hostBuffer = new long[1];
        int result = JCudaDriver.cuMemcpyDtoH(Pointer.to(hostBuffer), memory.base().withByteOffset(byteOffset), 8);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from device to host: " + CUresult.stringFor(result));
        }
        return hostBuffer[0];
    }

    @Override
    public void writeLong(Memory<CUdeviceptr> memory, long byteOffset, long value) {
        long[] hostBuffer = new long[]{value};
        int result = JCudaDriver.cuMemcpyHtoD(memory.base().withByteOffset(byteOffset), Pointer.to(hostBuffer), 8);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from host to device: " + CUresult.stringFor(result));
        }
    }

    @Override
    public float readFloat(Memory<CUdeviceptr> memory, long byteOffset) {
        float[] hostBuffer = new float[1];
        int result = JCudaDriver.cuMemcpyDtoH(Pointer.to(hostBuffer), memory.base().withByteOffset(byteOffset), 4);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from device to host: " + CUresult.stringFor(result));
        }
        return hostBuffer[0];
    }

    @Override
    public void writeFloat(Memory<CUdeviceptr> memory, long byteOffset, float value) {
        float[] hostBuffer = new float[]{value};
        int result = JCudaDriver.cuMemcpyHtoD(memory.base().withByteOffset(byteOffset), Pointer.to(hostBuffer), 4);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from host to device: " + CUresult.stringFor(result));
        }
    }

    @Override
    public double readDouble(Memory<CUdeviceptr> memory, long byteOffset) {
        double[] hostBuffer = new double[1];
        int result = JCudaDriver.cuMemcpyDtoH(Pointer.to(hostBuffer), memory.base().withByteOffset(byteOffset), 8);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from device to host: " + CUresult.stringFor(result));
        }
        return hostBuffer[0];
    }

    @Override
    public void writeDouble(Memory<CUdeviceptr> memory, long byteOffset, double value) {
        double[] hostBuffer = new double[]{value};
        int result = JCudaDriver.cuMemcpyHtoD(memory.base().withByteOffset(byteOffset), Pointer.to(hostBuffer), 8);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to copy memory from host to device: " + CUresult.stringFor(result));
        }
    }
}
