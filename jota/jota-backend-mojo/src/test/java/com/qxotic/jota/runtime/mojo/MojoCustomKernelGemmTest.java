package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.testutil.AbstractCustomKernelGemmTest;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import org.junit.jupiter.api.Assumptions;

class MojoCustomKernelGemmTest extends AbstractCustomKernelGemmTest {

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
    protected LaunchConfig launchConfig(int m, int n, int k) {
        int total = m * ((n + 1) / 2);
        int block = 128;
        int grid = (total + block - 1) / block;
        return LaunchConfig.grid(grid).block(block);
    }

    @Override
    protected Object[] kernelScalars(int m, int n, int k) {
        return new Object[] {(long) m, (long) n, (long) k};
    }

    @Override
    protected KernelProgram kernelProgram(String kernelName) {
        // language=mojo
        String source =
                """
                from std.gpu import global_idx

                fn gemm(
                    A: UnsafePointer[Float32, MutAnyOrigin],
                    B: UnsafePointer[Float32, MutAnyOrigin],
                    C: UnsafePointer[Float32, MutAnyOrigin],
                    M: Int,
                    N: Int,
                    K: Int,
                ):
                    cols2 = (UInt(N) + 1) // 2
                    idx = global_idx.x
                    total = UInt(M) * cols2
                    if idx < total:
                        row = idx // cols2
                        col0 = (idx % cols2) * 2
                        col1 = col0 + 1
                        has_col1 = col1 < UInt(N)
                        acc0 = Float32(0.0)
                        acc1 = Float32(0.0)

                        for kk in range(K):
                            kk_u = UInt(kk)
                            a = A[row * UInt(K) + kk_u]
                            acc0 += a * B[kk_u * UInt(N) + col0]
                            if has_col1:
                                acc1 += a * B[kk_u * UInt(N) + col1]

                        out0 = row * UInt(N) + col0
                        C[out0] = acc0
                        if has_col1:
                            C[row * UInt(N) + col1] = acc1
                """;
        return KernelProgram.source("mojo", source, kernelName);
    }
}
