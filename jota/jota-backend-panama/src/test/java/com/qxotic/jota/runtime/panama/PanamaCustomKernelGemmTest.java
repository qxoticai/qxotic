package com.qxotic.jota.runtime.panama;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.testutil.AbstractCustomKernelGemmTest;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import org.junit.jupiter.api.Assumptions;

class PanamaCustomKernelGemmTest extends AbstractCustomKernelGemmTest {

    @Override
    protected void assumeRuntimeReady() {
        Assumptions.assumeTrue(
                ConfiguredTestDevice.resolve()
                        .equals(ConfiguredTestDevice.resolve(DeviceType.PANAMA)),
                "PanamaCustomKernelGemmTest requires the Panama runtime");
    }

    @Override
    protected DeviceRuntime createRuntime() {
        return new PanamaDeviceRuntime();
    }

    @Override
    protected KernelProgram kernelProgram(String kernelName) {
        // language=java
        String source =
                """
                package com.qxotic.jota.runtime.jit;

                import com.qxotic.jota.memory.MemoryAccess;
                import com.qxotic.jota.memory.MemoryDomain;
                import com.qxotic.jota.memory.MemoryView;
                import com.qxotic.jota.runtime.JavaKernel;
                import com.qxotic.jota.runtime.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class GemmKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> A = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> B = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> C = (MemoryView<MemorySegment>) args.getBuffer(2);
                        int M = args.getInt(3);
                        int N = args.getInt(4);
                        int K = args.getInt(5);

                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        final int TILE_M = 8;
                        final int TILE_N = 8;
                        final int TILE_K = 8;

                        for (int rowBase = 0; rowBase < M; rowBase += TILE_M) {
                            int rowLimit = Math.min(rowBase + TILE_M, M);
                            for (int colBase = 0; colBase < N; colBase += TILE_N) {
                                int colLimit = Math.min(colBase + TILE_N, N);
                                for (int row = rowBase; row < rowLimit; row++) {
                                    long aRowBytes = A.byteOffset() + (long) row * K * Float.BYTES;
                                    long cRowBytes = C.byteOffset() + (long) row * N * Float.BYTES;
                                    for (int col = colBase; col < colLimit; col += 2) {
                                        boolean hasCol1 = col + 1 < colLimit;
                                        float acc0 = 0.0f;
                                        float acc1 = 0.0f;

                                        for (int kBase = 0; kBase < K; kBase += TILE_K) {
                                            int kEnd = Math.min(kBase + TILE_K, K);
                                            int kk = kBase;
                                            for (; kk + 3 < kEnd; kk += 4) {
                                                float a0 =
                                                        access.readFloat(
                                                                A.memory(),
                                                                aRowBytes
                                                                        + (long) kk * Float.BYTES);
                                                float a1 =
                                                        access.readFloat(
                                                                A.memory(),
                                                                aRowBytes
                                                                        + (long) (kk + 1)
                                                                                * Float.BYTES);
                                                float a2 =
                                                        access.readFloat(
                                                                A.memory(),
                                                                aRowBytes
                                                                        + (long) (kk + 2)
                                                                                * Float.BYTES);
                                                float a3 =
                                                        access.readFloat(
                                                                A.memory(),
                                                                aRowBytes
                                                                        + (long) (kk + 3)
                                                                                * Float.BYTES);

                                                long b0 =
                                                        B.byteOffset()
                                                                + ((long) kk * N + col)
                                                                        * Float.BYTES;
                                                long b1 =
                                                        B.byteOffset()
                                                                + ((long) (kk + 1) * N + col)
                                                                        * Float.BYTES;
                                                long b2 =
                                                        B.byteOffset()
                                                                + ((long) (kk + 2) * N + col)
                                                                        * Float.BYTES;
                                                long b3 =
                                                        B.byteOffset()
                                                                + ((long) (kk + 3) * N + col)
                                                                        * Float.BYTES;

                                                acc0 += a0 * access.readFloat(B.memory(), b0);
                                                acc0 += a1 * access.readFloat(B.memory(), b1);
                                                acc0 += a2 * access.readFloat(B.memory(), b2);
                                                acc0 += a3 * access.readFloat(B.memory(), b3);

                                                if (hasCol1) {
                                                    acc1 += a0 * access.readFloat(B.memory(), b0 + Float.BYTES);
                                                    acc1 += a1 * access.readFloat(B.memory(), b1 + Float.BYTES);
                                                    acc1 += a2 * access.readFloat(B.memory(), b2 + Float.BYTES);
                                                    acc1 += a3 * access.readFloat(B.memory(), b3 + Float.BYTES);
                                                }
                                            }

                                            for (; kk < kEnd; kk++) {
                                                float av =
                                                        access.readFloat(
                                                                A.memory(),
                                                                aRowBytes
                                                                        + (long) kk * Float.BYTES);
                                                long bOff =
                                                        B.byteOffset()
                                                                + ((long) kk * N + col)
                                                                        * Float.BYTES;
                                                acc0 += av * access.readFloat(B.memory(), bOff);
                                                if (hasCol1) {
                                                    acc1 +=
                                                            av
                                                                    * access.readFloat(
                                                                            B.memory(),
                                                                            bOff + Float.BYTES);
                                                }
                                            }
                                        }

                                        access.writeFloat(
                                                C.memory(), cRowBytes + (long) col * Float.BYTES, acc0);
                                        if (hasCol1) {
                                            access.writeFloat(
                                                    C.memory(),
                                                    cRowBytes
                                                            + (long) (col + 1) * Float.BYTES,
                                                    acc1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                """;
        return KernelProgram.source("java", source, "GemmKernel");
    }
}
