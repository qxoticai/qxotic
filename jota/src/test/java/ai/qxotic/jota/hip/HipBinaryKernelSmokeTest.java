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

class HipBinaryKernelSmokeTest {

    @Test
    void runsSubtractKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 256;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> 1.0f);

        MemoryView<?> output = runBinary(hostA, hostB, (a, b) -> a.subtract(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals((float) (n - 2), access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }

    @Test
    void runsMultiplyKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 128;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> 2.0f);

        MemoryView<?> output = runBinary(hostA, hostB, (a, b) -> a.multiply(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals(
                (float) (n - 1) * 2.0f, access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }

    @Test
    void runsDivideKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 128;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) (i + 1));
        MemoryView<MemorySegment> hostB = hostArray(n, i -> 2.0f);

        MemoryView<?> output = runBinary(hostA, hostB, (a, b) -> a.divide(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals((float) n / 2.0f, access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }

    private static MemoryView<MemorySegment> hostArray(int n, IndexValue supplier) {
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, supplier.value(i));
        }
        return hostView;
    }

    private static MemoryView<?> runBinary(
            MemoryView<MemorySegment> hostA, MemoryView<MemorySegment> hostB, TensorOp op) {
        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> devA =
                MemoryView.of(
                        hipContext
                                .memoryAllocator()
                                .allocateMemory(DataType.FP32, hostA.shape().size()),
                        DataType.FP32,
                        Layout.rowMajor(hostA.shape()));
        MemoryView<HipDevicePtr> devB =
                MemoryView.of(
                        hipContext
                                .memoryAllocator()
                                .allocateMemory(DataType.FP32, hostB.shape().size()),
                        DataType.FP32,
                        Layout.rowMajor(hostB.shape()));
        MemoryContext.copy(ContextFactory.ofMemorySegment(), hostA, hipContext, devA);
        MemoryContext.copy(ContextFactory.ofMemorySegment(), hostB, hipContext, devB);

        Tensor a = Tensor.of(devA);
        Tensor b = Tensor.of(devB);
        return op.apply(a, b).materialize();
    }

    private static MemoryAccess<MemorySegment> hostAccess() {
        return ContextFactory.ofMemorySegment().memoryAccess();
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

    @FunctionalInterface
    private interface TensorOp {
        Tensor apply(Tensor a, Tensor b);
    }

    @FunctionalInterface
    private interface IndexValue {
        float value(int index);
    }
}
