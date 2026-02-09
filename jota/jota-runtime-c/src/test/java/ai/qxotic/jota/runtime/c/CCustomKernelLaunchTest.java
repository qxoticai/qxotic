package ai.qxotic.jota.runtime.c;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.KernelProgram;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class CCustomKernelLaunchTest {

    static CDeviceRuntime runtime;

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(CNative.isAvailable(), "C JNI runtime not available");
        assumeGccAvailable();
        runtime = new CDeviceRuntime();
    }

    /**
     * C kernel calling convention: void fn(void **buffers, uint64_t *scalars, uint64_t scratch)
     *
     * <p>buffers[0] = A, buffers[1] = x, buffers[2] = y scalars[0] = M, scalars[1] = N
     */
    @Test
    void launchGemvKernel() {
        // language=c
        String source =
                """
                #include <stdint.h>

                void gemv(void **buffers, uint64_t *scalars, uint64_t scratch) {
                    const float *A = (const float *)buffers[0];
                    const float *x = (const float *)buffers[1];
                    float *y       = (float *)buffers[2];
                    int M = (int)scalars[0];
                    int N = (int)scalars[1];

                    for (int row = 0; row < M; row++) {
                        float dot = 0.0f;
                        for (int col = 0; col < N; col++) {
                            dot += A[row * N + col] * x[col];
                        }
                        y[row] = dot;
                    }
                }
                """;

        runtime.registerKernel(
                "gemv", KernelProgram.source(KernelProgram.C, source, "gemv"));

        int M = 3, N = 4;
        MemoryDomain<MemorySegment> domain = runtime.memoryDomain();

        MemoryView<MemorySegment> A = allocate(domain, DataType.FP32, Shape.of(M, N));
        MemoryView<MemorySegment> x = allocate(domain, DataType.FP32, Shape.flat(N));
        MemoryView<MemorySegment> y = allocate(domain, DataType.FP32, Shape.flat(M));

        // A = [[1, 2, 3, 4],
        //      [5, 6, 7, 8],
        //      [9, 10, 11, 12]]
        writeFloats(domain, A, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        // x = [1, 0, 2, -1]
        writeFloats(domain, x, 1, 0, 2, -1);

        // y = A * x = [3, 11, 19]
        runtime.launchKernel("gemv", A, x, y, M, N);

        assertFloats(domain, y, 3f, 11f, 19f);
    }

    // ── helpers ──────────────────────────────────────────────────

    private static MemoryView<MemorySegment> allocate(
            MemoryDomain<MemorySegment> domain, DataType dtype, Shape shape) {
        var mem = domain.memoryAllocator().allocateMemory(dtype, shape.size());
        return MemoryView.of(mem, dtype, Layout.rowMajor(shape));
    }

    private static void writeFloats(
            MemoryDomain<MemorySegment> domain, MemoryView<MemorySegment> view, float... values) {
        MemoryAccess<MemorySegment> access = domain.directAccess();
        for (int i = 0; i < values.length; i++) {
            access.writeFloat(view.memory(), view.byteOffset() + (long) i * Float.BYTES, values[i]);
        }
    }

    private static void assertFloats(
            MemoryDomain<MemorySegment> domain, MemoryView<MemorySegment> view, float... expected) {
        MemoryAccess<MemorySegment> access = domain.directAccess();
        for (int i = 0; i < expected.length; i++) {
            float actual =
                    access.readFloat(view.memory(), view.byteOffset() + (long) i * Float.BYTES);
            assertEquals(expected[i], actual, 0.0001f, "mismatch at index " + i);
        }
    }

    private static void assumeGccAvailable() {
        try {
            Process process = new ProcessBuilder("gcc", "--version").start();
            int code = process.waitFor();
            Assumptions.assumeTrue(code == 0, "gcc not available");
        } catch (Exception e) {
            Assumptions.assumeTrue(false, "gcc not available");
        }
    }
}
