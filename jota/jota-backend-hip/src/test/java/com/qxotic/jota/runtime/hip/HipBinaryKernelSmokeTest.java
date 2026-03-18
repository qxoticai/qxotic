package com.qxotic.jota.runtime.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class HipBinaryKernelSmokeTest {

    @Test
    void runsTracedAddKernel() {
        HipTestAssumptions.assumeHipReady();

        int n = 256;
        MemoryView<MemorySegment> hostA = hostArray(n, i -> (float) i);
        MemoryView<MemorySegment> hostB = hostArray(n, i -> 2.0f);

        MemoryView<?> output = runBinary(hostA, hostB, (a, b) -> a.add(b));

        MemoryAccess<MemorySegment> access = hostAccess();
        MemoryView<MemorySegment> hostOut = toHost(output);
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

    private static MemoryView<MemorySegment> toHost(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostView =
                (MemoryView<MemorySegment>)
                        Tensor.of(view).to(new Device(DeviceType.PANAMA, 0)).materialize();
        return hostView;
    }

    private static MemoryAccess<MemorySegment> hostAccess() {
        return DomainFactory.ofMemorySegment().directAccess();
    }

    // HIP runtime/device assumptions are centralized in HipTestAssumptions.

    @FunctionalInterface
    private interface TensorOp {
        Tensor apply(Tensor a, Tensor b);
    }

    @FunctionalInterface
    private interface IndexValue {
        float value(int index);
    }
}
