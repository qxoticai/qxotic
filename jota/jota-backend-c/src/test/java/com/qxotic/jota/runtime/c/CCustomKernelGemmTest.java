package com.qxotic.jota.runtime.c;

import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.testutil.AbstractCustomKernelGemmTest;
import com.qxotic.jota.testutil.ExternalToolChecks;
import org.junit.jupiter.api.Assumptions;

class CCustomKernelGemmTest extends AbstractCustomKernelGemmTest {

    @Override
    protected void assumeRuntimeReady() {
        Assumptions.assumeTrue(CNative.isAvailable(), "C JNI runtime not available");
        Assumptions.assumeTrue(ExternalToolChecks.hasVersionCommand("gcc"), "gcc not available");
    }

    @Override
    protected DeviceRuntime createRuntime() {
        return new CDeviceRuntime();
    }

    @Override
    protected KernelProgram kernelProgram(String kernelName) {
        String source =
                """
                #include <stdint.h>

                void gemm(void **buffers, uint64_t *scalars, uint64_t scratch) {
                    const float *A = (const float *)buffers[0];
                    const float *B = (const float *)buffers[1];
                    float *C = (float *)buffers[2];
                    int M = (int)scalars[0];
                    int N = (int)scalars[1];
                    int K = (int)scalars[2];

                    const int TILE_M = 8;
                    const int TILE_N = 8;
                    const int TILE_K = 8;

                    for (int i = 0; i < M * N; i++) {
                        C[i] = 0.0f;
                    }

                    for (int ii = 0; ii < M; ii += TILE_M) {
                        int iMax = ii + TILE_M;
                        if (iMax > M) {
                            iMax = M;
                        }

                        for (int jj = 0; jj < N; jj += TILE_N) {
                            int jMax = jj + TILE_N;
                            if (jMax > N) {
                                jMax = N;
                            }

                            for (int kkBase = 0; kkBase < K; kkBase += TILE_K) {
                                int kMax = kkBase + TILE_K;
                                if (kMax > K) {
                                    kMax = K;
                                }

                                for (int row = ii; row < iMax; row++) {
                                    const float *aRow = A + row * K;
                                    float *cRow = C + row * N;

                                    for (int col = jj; col < jMax; col++) {
                                        float acc = cRow[col];
                                        int kk = kkBase;

                                        for (; kk + 3 < kMax; kk += 4) {
                                            acc += aRow[kk] * B[kk * N + col];
                                            acc += aRow[kk + 1] * B[(kk + 1) * N + col];
                                            acc += aRow[kk + 2] * B[(kk + 2) * N + col];
                                            acc += aRow[kk + 3] * B[(kk + 3) * N + col];
                                        }
                                        for (; kk < kMax; kk++) {
                                            acc += aRow[kk] * B[kk * N + col];
                                        }
                                        cRow[col] = acc;
                                    }
                                }
                            }
                        }
                    }
                }
                """;
        return KernelProgram.source("c", source, kernelName);
    }
}
