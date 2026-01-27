package ai.qxotic.jota.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.BinaryOp;
import ai.qxotic.jota.tensor.ExecutionStream;
import ai.qxotic.jota.tensor.KernelArgs;
import ai.qxotic.jota.tensor.LaunchConfig;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class HipKernelMetadataSmokeTest {

    @Test
    void launchesVecAddMetaKernel() throws Exception {
        Assumptions.assumeTrue(HipRuntime.isAvailable());

        assumeHipccAvailable();

        int n = 1024;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
            b[i] = n - i;
        }

        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMemA = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        var hostMemB = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        var hostMemOut = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostA =
                MemoryView.of(hostMemA, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryView<MemorySegment> hostB =
                MemoryView.of(hostMemB, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryView<MemorySegment> hostOut =
                MemoryView.of(hostMemOut, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> hostAccess = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostA, i);
            hostAccess.writeFloat(hostA.memory(), offset, a[i]);
            hostAccess.writeFloat(hostB.memory(), offset, b[i]);
        }

        HipMemoryContext device = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> devA =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryView<HipDevicePtr> devB =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryView<HipDevicePtr> devOut =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));

        long byteSize = (long) n * Float.BYTES;
        device.memoryOperations()
                .copyFromNative(hostA.memory(), hostA.byteOffset(), devA.memory(), 0, byteSize);
        device.memoryOperations()
                .copyFromNative(hostB.memory(), hostB.byteOffset(), devB.memory(), 0, byteSize);

        HipKernelSpec spec = new HipKernelCompiler().compileMetadata(BinaryOp.ADD);
        byte[] hsaco = java.nio.file.Files.readAllBytes(spec.hsacoPath());
        try (HipModule module = HipModule.load(hsaco)) {
            HipKernelExecutable kernel =
                    new HipKernelExecutable(module.function(spec.kernelName()));

            long[] meta = new long[] {n};
            KernelArgs args =
                    new KernelArgs()
                            .addBuffer(devA)
                            .addBuffer(devB)
                            .addBuffer(devOut)
                            .addMetadata(meta);

            int block = 256;
            int grid = (n + block - 1) / block;
            LaunchConfig config = new LaunchConfig(grid, 1, 1, block, 1, 1, 0, false);
            ExecutionStream stream = new ExecutionStream(Device.HIP, 0L, true);

            kernel.launch(config, args, stream);
        }

        device.memoryOperations()
                .copyToNative(devOut.memory(), 0, hostOut.memory(), hostOut.byteOffset(), byteSize);

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostOut, i);
            float value = access.readFloat(hostOut.memory(), offset);
            assertEquals(a[i] + b[i], value, 0.0001f);
        }
    }

    private static void assumeHipccAvailable() {
        try {
            Process process = new ProcessBuilder("hipcc", "--version").start();
            int code = process.waitFor();
            Assumptions.assumeTrue(code == 0);
        } catch (Exception e) {
            Assumptions.assumeTrue(false, "hipcc not available");
        }
    }
}
