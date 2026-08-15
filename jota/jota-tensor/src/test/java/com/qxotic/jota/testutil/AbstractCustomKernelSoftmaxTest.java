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

public abstract class AbstractCustomKernelSoftmaxTest extends AbstractCustomKernelMatrixTest {

    private static final String KERNEL_NAME = "softmax";

    @Test
    final void launchSoftmaxKernel() {
        assumeRuntimeReady();
        DeviceRuntime runtime = createRuntime();

        int rows = rows();
        int cols = cols();

        MemoryDomain<MemorySegment> host = hostDomain();
        MemoryDomain<?> device = runtime.memoryDomain();

        MemoryView<MemorySegment> hostX = allocateF32(host, Shape.of(rows, cols));
        MemoryView<MemorySegment> hostY = allocateF32(host, Shape.of(rows, cols));
        fillInput(host, hostX, rows, cols);

        MemoryView<?> devX = allocateF32(device, Shape.of(rows, cols));
        MemoryView<?> devY = allocateF32(device, Shape.of(rows, cols));

        copy(host, hostX, device, devX);

        registerAndLaunch(
                runtime,
                new KernelTestSpec(
                        KERNEL_NAME,
                        kernelProgram(KERNEL_NAME),
                        launchConfig(rows, cols),
                        new Object[] {devX, devY},
                        kernelScalars(rows, cols)));

        copy(device, devY, host, hostY);
        assertMatchesReference(host, hostX, hostY, rows, cols);
    }

    protected abstract KernelProgram kernelProgram(String kernelName);

    protected LaunchConfig launchConfig(int rows, int cols) {
        return LaunchConfig.auto();
    }

    protected Object[] kernelScalars(int rows, int cols) {
        return new Object[] {rows, cols};
    }

    protected int rows() {
        return 29;
    }

    protected int cols() {
        return 67;
    }

    protected float tolerance() {
        return 1e-4f;
    }

    private static void fillInput(
            MemoryDomain<MemorySegment> host, MemoryView<MemorySegment> x, int rows, int cols) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                float value =
                        ((row % 7) - 3) * 0.71f
                                + ((col % 11) - 5) * 0.29f
                                + (float) Math.sin(0.13 * (row + 1) * (col + 1));
                writeF32(host, x, (long) row * cols + col, value);
            }
        }
    }

    private void assertMatchesReference(
            MemoryDomain<MemorySegment> host,
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> y,
            int rows,
            int cols) {
        float[] expected = ReferenceKernels.softmaxRows(toFlatF32Array(host, x), rows, cols);
        float[] actual = toFlatF32Array(host, y);

        for (int row = 0; row < rows; row++) {
            float rowSum = 0.0f;
            for (int col = 0; col < cols; col++) {
                int idx = row * cols + col;
                rowSum += actual[idx];
                assertEquals(
                        expected[idx],
                        actual[idx],
                        tolerance(),
                        "softmax mismatch at (row=" + row + ", col=" + col + ")");
            }
            assertEquals(1.0f, rowSum, 5e-4f, "softmax row sum mismatch at row=" + row);
        }
    }
}
