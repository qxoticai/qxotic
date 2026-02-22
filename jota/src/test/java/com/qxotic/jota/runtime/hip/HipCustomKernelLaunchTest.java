package com.qxotic.jota.runtime.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.tensor.KernelProgram;
import com.qxotic.jota.tensor.LaunchConfig;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class HipCustomKernelLaunchTest {

    static HipDeviceRuntime runtime;

    @BeforeAll
    static void setUp() {
        HipTestAssumptions.assumeHipReady();
        runtime = new HipDeviceRuntime();
    }

    @Test
    void launchGemvKernel() {
        // language=c++
        String source =
                """
                #include <hip/hip_runtime.h>

                extern "C" __global__
                void gemv(const float* A, const float* x, float* y, int M, int N) {
                    int row = blockIdx.x * blockDim.x + threadIdx.x;
                    if (row < M) {
                        float dot = 0.0f;
                        for (int col = 0; col < N; col++) {
                            dot += A[row * N + col] * x[col];
                        }
                        y[row] = dot;
                    }
                }
                """;

        runtime.registerKernel("gemv", KernelProgram.source(KernelProgram.HIP, source, "gemv"));

        int M = 3, N = 4;

        // Allocate on device
        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devA = allocateDevice(device, DataType.FP32, Shape.of(M, N));
        MemoryView<HipDevicePtr> devX = allocateDevice(device, DataType.FP32, Shape.flat(N));
        MemoryView<HipDevicePtr> devY = allocateDevice(device, DataType.FP32, Shape.flat(M));

        // Prepare host data
        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        MemoryView<MemorySegment> hostA = allocateHost(host, DataType.FP32, Shape.of(M, N));
        MemoryView<MemorySegment> hostX = allocateHost(host, DataType.FP32, Shape.flat(N));

        // A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        writeFloats(host, hostA, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        // x = [1, 0, 2, -1]
        writeFloats(host, hostX, 1, 0, 2, -1);

        // Copy to device
        copyToDevice(device, hostA, devA);
        copyToDevice(device, hostX, devX);

        // Launch: one thread per row
        runtime.launchKernel("gemv", LaunchConfig.grid(1).block(M), devA, devX, devY, M, N);

        // Copy result back
        MemoryView<MemorySegment> hostY = allocateHost(host, DataType.FP32, Shape.flat(M));
        copyToHost(device, devY, hostY);

        // y = A * x = [3, 11, 19]
        assertFloats(host, hostY, 3f, 11f, 19f);
    }

    // ── helpers ──────────────────────────────────────────────────

    private static MemoryView<HipDevicePtr> allocateDevice(
            HipMemoryDomain device, DataType dtype, Shape shape) {
        var mem = device.memoryAllocator().allocateMemory(dtype, shape.size());
        return MemoryView.of(mem, dtype, Layout.rowMajor(shape));
    }

    private static MemoryView<MemorySegment> allocateHost(
            MemoryDomain<MemorySegment> host, DataType dtype, Shape shape) {
        var mem = host.memoryAllocator().allocateMemory(dtype, shape.size());
        return MemoryView.of(mem, dtype, Layout.rowMajor(shape));
    }

    private static void writeFloats(
            MemoryDomain<MemorySegment> host, MemoryView<MemorySegment> view, float... values) {
        MemoryAccess<MemorySegment> access = host.directAccess();
        for (int i = 0; i < values.length; i++) {
            access.writeFloat(view.memory(), view.byteOffset() + (long) i * Float.BYTES, values[i]);
        }
    }

    private static void copyToDevice(
            HipMemoryDomain device, MemoryView<MemorySegment> src, MemoryView<HipDevicePtr> dst) {
        long bytes = src.dataType().byteSizeFor(src.shape());
        device.memoryOperations()
                .copyFromNative(src.memory(), src.byteOffset(), dst.memory(), 0, bytes);
    }

    private static void copyToHost(
            HipMemoryDomain device, MemoryView<HipDevicePtr> src, MemoryView<MemorySegment> dst) {
        long bytes = src.dataType().byteSizeFor(src.shape());
        device.memoryOperations()
                .copyToNative(
                        src.memory(), src.byteOffset(), dst.memory(), dst.byteOffset(), bytes);
    }

    private static void assertFloats(
            MemoryDomain<MemorySegment> host, MemoryView<MemorySegment> view, float... expected) {
        MemoryAccess<MemorySegment> access = host.directAccess();
        for (int i = 0; i < expected.length; i++) {
            float actual =
                    access.readFloat(view.memory(), view.byteOffset() + (long) i * Float.BYTES);
            assertEquals(expected[i], actual, 0.0001f, "mismatch at index " + i);
        }
    }

    // HIP runtime/device assumptions are centralized in HipTestAssumptions.
}
