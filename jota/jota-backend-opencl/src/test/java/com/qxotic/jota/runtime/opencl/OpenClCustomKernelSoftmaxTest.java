package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.testutil.AbstractCustomKernelSoftmaxTest;

class OpenClCustomKernelSoftmaxTest extends AbstractCustomKernelSoftmaxTest {

    @Override
    protected void assumeRuntimeReady() {
        OpenClTestAssumptions.assumeOpenClReady();
    }

    @Override
    protected DeviceRuntime createRuntime() {
        return new OpenClDeviceRuntime();
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
                __kernel void softmax(__global const float* X, __global float* Y, int rows, int cols) {
                    int row = (int)get_global_id(0);
                    if (row >= rows) {
                        return;
                    }

                    int base = row * cols;
                    float maxVal = X[base];
                    int col = 1;
                    for (; col + 1 < cols; col += 2) {
                        float v0 = X[base + col];
                        float v1 = X[base + col + 1];
                        maxVal = fmax(maxVal, v0);
                        maxVal = fmax(maxVal, v1);
                    }
                    for (; col < cols; col++) {
                        maxVal = fmax(maxVal, X[base + col]);
                    }

                    float sumExp = 0.0f;
                    col = 0;
                    for (; col + 1 < cols; col += 2) {
                        float e0 = exp(X[base + col] - maxVal);
                        float e1 = exp(X[base + col + 1] - maxVal);
                        Y[base + col] = e0;
                        Y[base + col + 1] = e1;
                        sumExp += e0 + e1;
                    }
                    for (; col < cols; col++) {
                        float e = exp(X[base + col] - maxVal);
                        Y[base + col] = e;
                        sumExp += e;
                    }

                    float inv = 1.0f / sumExp;
                    col = 0;
                    for (; col + 1 < cols; col += 2) {
                        Y[base + col] *= inv;
                        Y[base + col + 1] *= inv;
                    }
                    for (; col < cols; col++) {
                        Y[base + col] *= inv;
                    }
                }
                """;
        return KernelProgram.source("opencl", source, kernelName);
    }
}
