package com.qxotic.jota.runtime.panama;

import com.qxotic.jota.Device;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.testutil.AbstractCustomKernelSoftmaxTest;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import org.junit.jupiter.api.Assumptions;

class PanamaCustomKernelSoftmaxTest extends AbstractCustomKernelSoftmaxTest {

    @Override
    protected void assumeRuntimeReady() {
        Assumptions.assumeTrue(
                ConfiguredTestDevice.resolve() == Device.PANAMA,
                "PanamaCustomKernelSoftmaxTest requires the Panama runtime");
    }

    @Override
    protected DeviceRuntime createRuntime() {
        return new PanamaDeviceRuntime();
    }

    @Override
    protected KernelProgram kernelProgram(String kernelName) {
        String source =
                """
                package com.qxotic.jota.runtime.jit;

                import com.qxotic.jota.memory.MemoryAccess;
                import com.qxotic.jota.memory.MemoryDomain;
                import com.qxotic.jota.memory.MemoryView;
                import com.qxotic.jota.runtime.JavaKernel;
                import com.qxotic.jota.runtime.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class SoftmaxKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> X = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> Y = (MemoryView<MemorySegment>) args.getBuffer(1);
                        int rows = args.getInt(2);
                        int cols = args.getInt(3);

                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        for (int row = 0; row < rows; row++) {
                            long rowOffset = X.byteOffset() + (long) row * cols * Float.BYTES;
                            long outOffset = Y.byteOffset() + (long) row * cols * Float.BYTES;

                            float maxVal = access.readFloat(X.memory(), rowOffset);
                            int col = 1;
                            for (; col + 1 < cols; col += 2) {
                                float v0 =
                                        access.readFloat(
                                                X.memory(), rowOffset + (long) col * Float.BYTES);
                                float v1 =
                                        access.readFloat(
                                                X.memory(), rowOffset + (long) (col + 1) * Float.BYTES);
                                if (v0 > maxVal) {
                                    maxVal = v0;
                                }
                                if (v1 > maxVal) {
                                    maxVal = v1;
                                }
                            }
                            for (; col < cols; col++) {
                                float v =
                                        access.readFloat(
                                                X.memory(), rowOffset + (long) col * Float.BYTES);
                                if (v > maxVal) {
                                    maxVal = v;
                                }
                            }

                            float sumExp = 0.0f;
                            col = 0;
                            for (; col + 1 < cols; col += 2) {
                                float e0 =
                                        (float)
                                                Math.exp(
                                                        access.readFloat(
                                                                        X.memory(),
                                                                        rowOffset + (long) col * Float.BYTES)
                                                                - maxVal);
                                float e1 =
                                        (float)
                                                Math.exp(
                                                        access.readFloat(
                                                                        X.memory(),
                                                                        rowOffset
                                                                                + (long) (col + 1)
                                                                                        * Float.BYTES)
                                                                - maxVal);
                                access.writeFloat(
                                        Y.memory(), outOffset + (long) col * Float.BYTES, e0);
                                access.writeFloat(
                                        Y.memory(),
                                        outOffset + (long) (col + 1) * Float.BYTES,
                                        e1);
                                sumExp += e0 + e1;
                            }
                            for (; col < cols; col++) {
                                float e =
                                        (float)
                                                Math.exp(
                                                        access.readFloat(
                                                                        X.memory(),
                                                                        rowOffset + (long) col * Float.BYTES)
                                                                - maxVal);
                                access.writeFloat(
                                        Y.memory(), outOffset + (long) col * Float.BYTES, e);
                                sumExp += e;
                            }

                            float inv = 1.0f / sumExp;
                            col = 0;
                            for (; col + 1 < cols; col += 2) {
                                long out0 = outOffset + (long) col * Float.BYTES;
                                long out1 = outOffset + (long) (col + 1) * Float.BYTES;
                                access.writeFloat(Y.memory(), out0, access.readFloat(Y.memory(), out0) * inv);
                                access.writeFloat(Y.memory(), out1, access.readFloat(Y.memory(), out1) * inv);
                            }
                            for (; col < cols; col++) {
                                long out = outOffset + (long) col * Float.BYTES;
                                access.writeFloat(Y.memory(), out, access.readFloat(Y.memory(), out) * inv);
                            }
                        }
                    }
                }
                """;
        return KernelProgram.source("java", source, "SoftmaxKernel");
    }
}
