package ai.qxotic.jota.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.DomainFactory;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class HipBinaryKernelSmokeTest {

    @Test
    void runsTracedAddKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 256;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> 2.0f);

        MemoryView<?> output = runBinary(hostA, hostB, (a, b) -> a.add(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        MemoryView<MemorySegment> hostOut = toHost(DomainFactory.ofMemorySegment(), output);
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals(
                (float) (n - 1) + 2.0f, access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }

    private static MemoryView<MemorySegment> hostArray(int n, IndexValue supplier) {
        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> access = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, supplier.value(i));
        }
        return hostView;
    }

    private static MemoryView<?> runBinary(
            MemoryView<MemorySegment> hostA, MemoryView<MemorySegment> hostB, TensorOp op) {
        HipMemoryDomain hipDomain = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devA =
                MemoryView.of(
                        hipDomain
                                .memoryAllocator()
                                .allocateMemory(DataType.FP32, hostA.shape().size()),
                        DataType.FP32,
                        Layout.rowMajor(hostA.shape()));
        MemoryView<HipDevicePtr> devB =
                MemoryView.of(
                        hipDomain
                                .memoryAllocator()
                                .allocateMemory(DataType.FP32, hostB.shape().size()),
                        DataType.FP32,
                        Layout.rowMajor(hostB.shape()));
        MemoryDomain.copy(DomainFactory.ofMemorySegment(), hostA, hipDomain, devA);
        MemoryDomain.copy(DomainFactory.ofMemorySegment(), hostB, hipDomain, devB);

        Tensor a = Tensor.of(devA);
        Tensor b = Tensor.of(devB);
        Tensor traced = Tracer.trace(a, b, op::apply);
        return traced.materialize();
    }

    private static MemoryView<MemorySegment> toHost(
            MemoryDomain<MemorySegment> host, MemoryView<?> view) {
        if (view.memory().base() instanceof MemorySegment) {
            @SuppressWarnings("unchecked")
            MemoryView<MemorySegment> hostView = (MemoryView<MemorySegment>) view;
            return hostView;
        }
        @SuppressWarnings("unchecked")
        MemoryView<HipDevicePtr> devView = (MemoryView<HipDevicePtr>) view;
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        host.memoryAllocator().allocateMemory(devView.dataType(), devView.shape()),
                        devView.dataType(),
                        devView.layout());
        long byteSize = devView.dataType().byteSizeFor(devView.shape());
        HipMemoryDomain.instance()
                .memoryOperations()
                .copyToNative(
                        devView.memory(),
                        devView.byteOffset(),
                        hostView.memory(),
                        hostView.byteOffset(),
                        byteSize);
        return hostView;
    }

    private static MemoryAccess<MemorySegment> hostAccess() {
        return DomainFactory.ofMemorySegment().directAccess();
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
