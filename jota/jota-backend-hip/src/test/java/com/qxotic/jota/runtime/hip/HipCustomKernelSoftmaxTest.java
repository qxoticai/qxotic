package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.testutil.AbstractCustomKernelSoftmaxTest;

class HipCustomKernelSoftmaxTest extends AbstractCustomKernelSoftmaxTest {

    @Override
    protected void assumeRuntimeReady() {
        HipTestAssumptions.assumeHipReady();
    }

    @Override
    protected DeviceRuntime createRuntime() {
        return new HipDeviceRuntime();
    }

    @Override
    protected LaunchConfig launchConfig(int rows, int cols) {
        int block = 128;
        int grid = (rows + block - 1) / block;
        return LaunchConfig.grid(grid).block(block);
    }

    @Override
    protected KernelProgram kernelProgram(String kernelName) {
        String source =
                """
                #include <hip/hip_runtime.h>
                #include <math.h>

                extern "C" __global__
                void softmax(const float* X, float* Y, int rows, int cols) {
                    int row = blockIdx.x * blockDim.x + threadIdx.x;
                    if (row >= rows) {
                        return;
                    }

                    const float* xRow = X + row * cols;
                    float* yRow = Y + row * cols;

                    float maxVal = xRow[0];
                    int col = 1;
                    for (; col + 1 < cols; col += 2) {
                        float v0 = xRow[col];
                        float v1 = xRow[col + 1];
                        maxVal = fmaxf(maxVal, v0);
                        maxVal = fmaxf(maxVal, v1);
                    }
                    for (; col < cols; col++) {
                        maxVal = fmaxf(maxVal, xRow[col]);
                    }

                    float sumExp = 0.0f;
                    col = 0;
                    for (; col + 1 < cols; col += 2) {
                        float e0 = expf(xRow[col] - maxVal);
                        float e1 = expf(xRow[col + 1] - maxVal);
                        yRow[col] = e0;
                        yRow[col + 1] = e1;
                        sumExp += e0 + e1;
                    }
                    for (; col < cols; col++) {
                        float e = expf(xRow[col] - maxVal);
                        yRow[col] = e;
                        sumExp += e;
                    }

                    float inv = 1.0f / sumExp;
                    col = 0;
                    for (; col + 1 < cols; col += 2) {
                        yRow[col] *= inv;
                        yRow[col + 1] *= inv;
                    }
                    for (; col < cols; col++) {
                        yRow[col] *= inv;
                    }
                }
                """;
        return KernelProgram.source("hip", source, kernelName);
    }
}
