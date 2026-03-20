package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.testutil.AbstractCustomKernelSoftmaxTest;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import org.junit.jupiter.api.Assumptions;

class MojoCustomKernelSoftmaxTest extends AbstractCustomKernelSoftmaxTest {

    @Override
    protected void assumeRuntimeReady() {
        Assumptions.assumeTrue(MojoRuntime.isAvailable(), "libjota_mojo.so is not available");
        Assumptions.assumeTrue(
                ConfiguredTestDevice.hasRuntime(DeviceType.HIP), "HIP runtime is unavailable");
    }

    @Override
    protected DeviceRuntime createRuntime() {
        return new MojoDeviceRuntime();
    }

    @Override
    protected LaunchConfig launchConfig(int rows, int cols) {
        int block = 128;
        int grid = (rows + block - 1) / block;
        return LaunchConfig.grid(grid).block(block);
    }

    @Override
    protected Object[] kernelScalars(int rows, int cols) {
        return new Object[] {(long) rows, (long) cols};
    }

    @Override
    protected KernelProgram kernelProgram(String kernelName) {
        String source =
                """
                from std.gpu import global_idx
                from math import exp

                fn softmax(
                    X: UnsafePointer[Float32, MutAnyOrigin],
                    Y: UnsafePointer[Float32, MutAnyOrigin],
                    rows: Int,
                    cols: Int,
                ):
                    row = Int(global_idx.x)
                    if row >= rows:
                        return

                    base = UInt(row * cols)
                    max_val = X[base]
                    col = 1
                    while col + 1 < cols:
                        v0 = X[base + UInt(col)]
                        v1 = X[base + UInt(col + 1)]
                        if v0 > max_val:
                            max_val = v0
                        if v1 > max_val:
                            max_val = v1
                        col += 2
                    while col < cols:
                        v = X[base + UInt(col)]
                        if v > max_val:
                            max_val = v
                        col += 1

                    sum_exp = Float32(0.0)
                    col = 0
                    while col + 1 < cols:
                        e0 = exp(X[base + UInt(col)] - max_val)
                        e1 = exp(X[base + UInt(col + 1)] - max_val)
                        Y[base + UInt(col)] = e0
                        Y[base + UInt(col + 1)] = e1
                        sum_exp += e0 + e1
                        col += 2
                    while col < cols:
                        e = exp(X[base + UInt(col)] - max_val)
                        Y[base + UInt(col)] = e
                        sum_exp += e
                        col += 1

                    inv = Float32(1.0) / sum_exp
                    col = 0
                    while col + 1 < cols:
                        Y[base + UInt(col)] *= inv
                        Y[base + UInt(col + 1)] *= inv
                        col += 2
                    while col < cols:
                        Y[base + UInt(col)] *= inv
                        col += 1
                """;
        return KernelProgram.source("mojo", source, kernelName);
    }
}
