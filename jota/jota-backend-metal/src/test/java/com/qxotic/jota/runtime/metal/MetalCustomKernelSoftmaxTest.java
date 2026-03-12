package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.testutil.AbstractCustomKernelSoftmaxTest;

class MetalCustomKernelSoftmaxTest extends AbstractCustomKernelSoftmaxTest {

    @Override
    protected void assumeRuntimeReady() {
        MetalTestAssumptions.assumeMetalReady();
    }

    @Override
    protected DeviceRuntime createRuntime() {
        return new MetalDeviceRuntime();
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
                #include <metal_stdlib>
                using namespace metal;

                kernel void softmax(
                        const device float* X [[buffer(0)]],
                        device float* Y [[buffer(1)]],
                        constant int* rowsPtr [[buffer(2)]],
                        constant int* colsPtr [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
                    int rows = rowsPtr[0];
                    int cols = colsPtr[0];
                    int row = (int)gid;
                    if (row >= rows) {
                        return;
                    }

                    int base = row * cols;
                    float maxVal = X[base];
                    int col = 1;
                    for (; col + 1 < cols; col += 2) {
                        float v0 = X[base + col];
                        float v1 = X[base + col + 1];
                        maxVal = max(maxVal, v0);
                        maxVal = max(maxVal, v1);
                    }
                    for (; col < cols; col++) {
                        maxVal = max(maxVal, X[base + col]);
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
        return KernelProgram.source("metal", source, kernelName);
    }
}
