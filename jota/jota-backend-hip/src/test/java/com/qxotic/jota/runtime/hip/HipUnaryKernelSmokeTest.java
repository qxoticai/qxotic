package com.qxotic.jota.runtime.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
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
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class HipUnaryKernelSmokeTest {

    @Test
    void runsTracedSqrtKernel() {
        HipTestAssumptions.assumeHipReady();

        int n = 256;
        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, i);
        }

        HipMemoryDomain hipDomain = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipDomain.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryDomain.copy(host, hostView, hipDomain, dev);

        Tensor traced = Tracer.trace(Tensor.of(dev), Tensor::sqrt);
        MemoryView<?> output = traced.materialize();

        MemoryView<MemorySegment> hostOut = toHost(output);
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals(
                (float) Math.sqrt(n - 1), access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }

    private static MemoryView<MemorySegment> toHost(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostView =
                (MemoryView<MemorySegment>)
                        Tensor.of(view)
                                .to(ConfiguredTestDevice.resolve(DeviceType.PANAMA))
                                .materialize();
        return hostView;
    }

    // HIP runtime/device assumptions are centralized in HipTestAssumptions.
}
