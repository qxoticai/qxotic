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
import ai.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class HipUnaryKernelSmokeTest {

    @Test
    void runsTracedSqrtKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 256;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, i);
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        Tensor traced = Tracer.trace(Tensor.of(dev), Tensor::sqrt);
        MemoryView<?> output = traced.materialize();

        MemoryView<MemorySegment> hostOut = toHost(host, hipContext, output);
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals(
                (float) Math.sqrt(n - 1), access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }
    private static MemoryView<MemorySegment> toHost(
            MemoryContext<MemorySegment> host,
            HipMemoryContext device,
            MemoryView<?> view) {
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
        device.memoryOperations()
                .copyToNative(
                        devView.memory(),
                        devView.byteOffset(),
                        hostView.memory(),
                        hostView.byteOffset(),
                        byteSize);
        return hostView;
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
