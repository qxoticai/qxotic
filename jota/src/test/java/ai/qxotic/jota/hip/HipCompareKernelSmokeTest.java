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

class HipCompareKernelSmokeTest {

    @Test
    void runsEqualKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 16;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> (float) i);

        MemoryView<?> output = runCompare(hostA, hostB, (a, b) -> a.equal(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals(1, access.readByte(hostOut.memory(), lastOffset));
    }

    @Test
    void runsLessThanKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 16;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> 5.0f);

        MemoryView<?> output = runCompare(hostA, hostB, (a, b) -> a.lessThan(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long offset0 = Indexing.linearToOffset(hostOut, 0);
        long offset7 = Indexing.linearToOffset(hostOut, 7);
        assertEquals(1, access.readByte(hostOut.memory(), offset0));
        assertEquals(0, access.readByte(hostOut.memory(), offset7));
    }

    @Test
    void runsNotEqualKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> (float) i);

        MemoryView<?> output = runCompare(hostA, hostB, (a, b) -> a.notEqual(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals(0, access.readByte(hostOut.memory(), lastOffset));
    }

    @Test
    void runsGreaterThanKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> 3.0f);

        MemoryView<?> output = runCompare(hostA, hostB, (a, b) -> a.greaterThan(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long offset0 = Indexing.linearToOffset(hostOut, 0);
        long offset7 = Indexing.linearToOffset(hostOut, 7);
        assertEquals(0, access.readByte(hostOut.memory(), offset0));
        assertEquals(1, access.readByte(hostOut.memory(), offset7));
    }

    @Test
    void runsLessThanOrEqualKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> 3.0f);

        MemoryView<?> output = runCompare(hostA, hostB, (a, b) -> a.lessThanOrEqual(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long offset3 = Indexing.linearToOffset(hostOut, 3);
        long offset7 = Indexing.linearToOffset(hostOut, 7);
        assertEquals(1, access.readByte(hostOut.memory(), offset3));
        assertEquals(0, access.readByte(hostOut.memory(), offset7));
    }

    @Test
    void runsGreaterThanOrEqualKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> 3.0f);

        MemoryView<?> output = runCompare(hostA, hostB, (a, b) -> a.greaterThanOrEqual(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long offset3 = Indexing.linearToOffset(hostOut, 3);
        long offset0 = Indexing.linearToOffset(hostOut, 0);
        assertEquals(1, access.readByte(hostOut.memory(), offset3));
        assertEquals(0, access.readByte(hostOut.memory(), offset0));
    }

    @Test
    void runsFp64LessThanKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        MemoryView<MemorySegment> hostA = hostArrayFp64(n, i -> (double) i);
        MemoryView<MemorySegment> hostB = hostArrayFp64(n, i -> 4.0);

        MemoryView<?> output = runCompareFp64(hostA, hostB, (a, b) -> a.lessThan(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long offset0 = Indexing.linearToOffset(hostOut, 0);
        long offset7 = Indexing.linearToOffset(hostOut, 7);
        assertEquals(1, access.readByte(hostOut.memory(), offset0));
        assertEquals(0, access.readByte(hostOut.memory(), offset7));
    }

    @Test
    void runsFp64EqualKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        MemoryView<MemorySegment> hostA = hostArrayFp64(n, i -> (double) i);
        MemoryView<MemorySegment> hostB = hostArrayFp64(n, i -> (double) i);

        MemoryView<?> output = runCompareFp64(hostA, hostB, (a, b) -> a.equal(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals(1, access.readByte(hostOut.memory(), lastOffset));
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

    private static MemoryView<?> runCompare(
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

    private static MemoryView<MemorySegment> hostArrayFp64(int n, DoubleIndexValue supplier) {
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP64, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP64, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeDouble(hostView.memory(), offset, supplier.value(i));
        }
        return hostView;
    }

    private static MemoryView<?> runCompareFp64(
            MemoryView<MemorySegment> hostA, MemoryView<MemorySegment> hostB, TensorOp op) {
        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> devA =
                MemoryView.of(
                        hipContext
                                .memoryAllocator()
                                .allocateMemory(DataType.FP64, hostA.shape().size()),
                        DataType.FP64,
                        Layout.rowMajor(hostA.shape()));
        MemoryView<HipDevicePtr> devB =
                MemoryView.of(
                        hipContext
                                .memoryAllocator()
                                .allocateMemory(DataType.FP64, hostB.shape().size()),
                        DataType.FP64,
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

    @FunctionalInterface
    private interface DoubleIndexValue {
        double value(int index);
    }
}
