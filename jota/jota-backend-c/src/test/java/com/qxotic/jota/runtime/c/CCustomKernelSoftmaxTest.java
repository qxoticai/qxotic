package com.qxotic.jota.runtime.c;

import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.testutil.AbstractCustomKernelSoftmaxTest;
import com.qxotic.jota.testutil.ExternalToolChecks;
import org.junit.jupiter.api.Assumptions;

class CCustomKernelSoftmaxTest extends AbstractCustomKernelSoftmaxTest {

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
                #include <math.h>
                #include <stdint.h>

                void softmax(void **buffers, uint64_t *scalars, uint64_t scratch) {
                    const float *X = (const float *)buffers[0];
                    float *Y = (float *)buffers[1];
                    int rows = (int)scalars[0];
                    int cols = (int)scalars[1];

                    for (int row = 0; row < rows; row++) {
                        const float *xRow = X + row * cols;
                        float *yRow = Y + row * cols;

                        float maxVal = xRow[0];
                        int col = 1;
                        for (; col + 1 < cols; col += 2) {
                            float v0 = xRow[col];
                            float v1 = xRow[col + 1];
                            if (v0 > maxVal) {
                                maxVal = v0;
                            }
                            if (v1 > maxVal) {
                                maxVal = v1;
                            }
                        }
                        for (; col < cols; col++) {
                            if (xRow[col] > maxVal) {
                                maxVal = xRow[col];
                            }
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
                }
                """;
        return KernelProgram.source("c", source, kernelName);
    }
}
