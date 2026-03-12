package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.testutil.AbstractCustomKernelGemmTest;

class OpenClCustomKernelGemmTest extends AbstractCustomKernelGemmTest {

    @Override
    protected void assumeRuntimeReady() {
        OpenClTestAssumptions.assumeOpenClReady();
    }

    @Override
    protected DeviceRuntime createRuntime() {
        return new OpenClDeviceRuntime();
    }

    @Override
    protected LaunchConfig launchConfig(int m, int n, int k) {
        int total = m * ((n + 1) / 2);
        int block = 128;
        int grid = (total + block - 1) / block;
        return LaunchConfig.grid(grid).block(block);
    }

    @Override
    protected KernelProgram kernelProgram(String kernelName) {
        String source =
                """
                __kernel void gemm(
                        __global const float* A,
                        __global const float* B,
                        __global float* C,
                        int M,
                        int N,
                        int K) {
                    int idx = (int)get_global_id(0);
                    int cols2 = (N + 1) / 2;
                    int total = M * cols2;
                    if (idx >= total) {
                        return;
                    }

                    int row = idx / cols2;
                    int col = (idx % cols2) * 2;
                    int hasCol1 = col + 1 < N;
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;

                    for (int kBase = 0; kBase < K; kBase += 8) {
                        int kEnd = min(kBase + 8, K);
                        int kk = kBase;

                        for (; kk + 3 < kEnd; kk += 4) {
                            float a0 = A[row * K + kk];
                            float a1 = A[row * K + kk + 1];
                            float a2 = A[row * K + kk + 2];
                            float a3 = A[row * K + kk + 3];

                            acc0 += a0 * B[kk * N + col];
                            acc0 += a1 * B[(kk + 1) * N + col];
                            acc0 += a2 * B[(kk + 2) * N + col];
                            acc0 += a3 * B[(kk + 3) * N + col];

                            if (hasCol1 != 0) {
                                acc1 += a0 * B[kk * N + col + 1];
                                acc1 += a1 * B[(kk + 1) * N + col + 1];
                                acc1 += a2 * B[(kk + 2) * N + col + 1];
                                acc1 += a3 * B[(kk + 3) * N + col + 1];
                            }
                        }

                        for (; kk < kEnd; kk++) {
                            float a = A[row * K + kk];
                            acc0 += a * B[kk * N + col];
                            if (hasCol1 != 0) {
                                acc1 += a * B[kk * N + col + 1];
                            }
                        }
                    }

                    C[row * N + col] = acc0;
                    if (hasCol1 != 0) {
                        C[row * N + col + 1] = acc1;
                    }
                }
                """;
        return KernelProgram.source("opencl", source, kernelName);
    }
}
