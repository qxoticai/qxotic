package com.qxotic.jota.runtime.opencl;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class OpenClKernelSmokeTest {

    @Test
    void launchesLirKernel() {
        OpenClTestAssumptions.assumeOpenClReady();

        int n = 64;
        Tensor input =
                Tensor.iota(n, DataType.FP32).to(ConfiguredTestDevice.resolve(DeviceType.OPENCL));
        Tensor traced = Tracer.trace(input, t -> t.multiply(2.0f).add(1.0f));
        MemoryView<?> output = traced.materialize();

        MemoryView<MemorySegment> host = toHost(output);
        MemoryAccess<MemorySegment> access = DomainFactory.ofMemorySegment().directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(host, i);
            float value = access.readFloat(host.memory(), offset);
            assertEquals(i * 2.0f + 1.0f, value, 1e-4f);
        }
    }

    @Test
    void copiesHostToDeviceAndBack() {
        OpenClTestAssumptions.assumeOpenClReady();

        int n = 128;
        Tensor src = Tensor.iota(n, DataType.I32);
        Tensor onOpenCl = src.to(ConfiguredTestDevice.resolve(DeviceType.OPENCL));
        MemoryView<MemorySegment> host = toHost(onOpenCl.materialize());

        MemoryAccess<MemorySegment> access = DomainFactory.ofMemorySegment().directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(host, i);
            assertEquals(i, access.readInt(host.memory(), offset));
        }
    }

    @Test
    void copiesHeapBackedFloatMemorySegmentToDeviceAndBack() {
        OpenClTestAssumptions.assumeOpenClReady();

        float[] values = {1.5f, -2.25f, 0.0f, 9.75f, -6.5f};
        Memory<MemorySegment> hostMemory =
                MemoryFactory.ofMemorySegment(MemorySegment.ofArray(values));
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        hostMemory, DataType.FP32, Layout.rowMajor(Shape.flat(values.length)));

        Tensor onOpenCl = Tensor.of(hostView).to(ConfiguredTestDevice.resolve(DeviceType.OPENCL));
        MemoryView<MemorySegment> roundTrip = toHost(onOpenCl.materialize());
        MemoryAccess<MemorySegment> access = DomainFactory.ofMemorySegment().directAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = Indexing.linearToOffset(roundTrip, i);
            assertEquals(values[i], access.readFloat(roundTrip.memory(), offset), 1e-6f);
        }
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
}
