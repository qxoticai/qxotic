package ai.qxotic.jota.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class HipKernelSmokeTest {

    @Test
    void launchesVecAddKernel() throws Exception {
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
        MemoryView<MemorySegment> hostA =
                MemoryView.of(hostMemA, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryView<MemorySegment> hostB =
                MemoryView.of(hostMemB, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

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

        Tensor aTensor = Tensor.of(devA);
        Tensor bTensor = Tensor.of(devB);
        MemoryView<?> output = aTensor.add(bTensor).materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostView = (MemoryView<MemorySegment>) output;

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            float value = access.readFloat(hostView.memory(), offset);
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
