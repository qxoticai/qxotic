package com.qxotic.jota.testutil;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

public abstract class AbstractCustomKernelGemmTest extends AbstractCustomKernelMatrixTest {

    private static final String KERNEL_NAME = "gemm";

    @Test
    final void launchGemmKernel() {
        assumeRuntimeReady();
        DeviceRuntime runtime = createRuntime();

        int m = m();
        int n = n();
        int k = k();

        MemoryDomain<MemorySegment> host = hostDomain();
        MemoryDomain<?> device = runtime.memoryDomain();

        MemoryView<MemorySegment> hostA = allocateF32(host, Shape.of(m, k));
        MemoryView<MemorySegment> hostB = allocateF32(host, Shape.of(k, n));
        MemoryView<MemorySegment> hostC = allocateF32(host, Shape.of(m, n));

        fillInputs(host, hostA, hostB, m, n, k);

        MemoryView<?> devA = allocateF32(device, Shape.of(m, k));
        MemoryView<?> devB = allocateF32(device, Shape.of(k, n));
        MemoryView<?> devC = allocateF32(device, Shape.of(m, n));

        copy(host, hostA, device, devA);
        copy(host, hostB, device, devB);

        registerAndLaunch(
                runtime,
                new KernelTestSpec(
                        KERNEL_NAME,
                        kernelProgram(KERNEL_NAME),
                        launchConfig(m, n, k),
                        new Object[] {devA, devB, devC},
                        kernelScalars(m, n, k)));

        copy(device, devC, host, hostC);
        assertMatchesReference(host, hostA, hostB, hostC, m, n, k);
    }

    protected abstract KernelProgram kernelProgram(String kernelName);

    protected LaunchConfig launchConfig(int m, int n, int k) {
        return LaunchConfig.auto();
    }

    protected int m() {
        return 23;
    }

    protected int n() {
        return 19;
    }

    protected int k() {
        return 17;
    }

    protected float tolerance() {
        return 1e-4f;
    }

    protected Object[] kernelScalars(int m, int n, int k) {
        return new Object[] {m, n, k};
    }

    private static void fillInputs(
            MemoryDomain<MemorySegment> host,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> b,
            int m,
            int n,
            int k) {
        for (int row = 0; row < m; row++) {
            for (int kk = 0; kk < k; kk++) {
                float value = aValue(row, kk);
                long index = (long) row * k + kk;
                writeF32(host, a, index, value);
            }
        }
        for (int kk = 0; kk < k; kk++) {
            for (int col = 0; col < n; col++) {
                float value = bValue(kk, col);
                long index = (long) kk * n + col;
                writeF32(host, b, index, value);
            }
        }
    }

    private void assertMatchesReference(
            MemoryDomain<MemorySegment> host,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> b,
            MemoryView<MemorySegment> c,
            int m,
            int n,
            int k) {
        float[] expected =
                ReferenceKernels.gemm(toFlatF32Array(host, a), toFlatF32Array(host, b), m, n, k);
        float[] actual = toFlatF32Array(host, c);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tolerance(), "mismatch at flat index=" + i);
        }
    }

    private static float aValue(int row, int kk) {
        return ((row % 7) - 3) * 0.37f + ((kk % 5) - 2) * 0.19f;
    }

    private static float bValue(int kk, int col) {
        return ((kk % 11) - 5) * 0.11f - ((col % 6) - 3) * 0.23f;
    }
}
