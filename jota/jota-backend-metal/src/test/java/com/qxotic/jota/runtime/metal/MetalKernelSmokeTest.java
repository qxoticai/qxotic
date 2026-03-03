package com.qxotic.jota.runtime.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class MetalKernelSmokeTest {

    @Test
    void launchesLirKernel() {
        MetalTestAssumptions.assumeMetalReady();

        int n = 64;
        Tensor input = Tensor.iota(n, DataType.FP32).to(Device.METAL);
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
        MetalTestAssumptions.assumeMetalReady();

        int n = 128;
        Tensor src = Tensor.iota(n, DataType.I32);
        Tensor onMetal = src.to(Device.METAL);
        MemoryView<MemorySegment> host = toHost(onMetal.materialize());

        MemoryAccess<MemorySegment> access = DomainFactory.ofMemorySegment().directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(host, i);
            assertEquals(i, access.readInt(host.memory(), offset));
        }
    }

    private static MemoryView<MemorySegment> toHost(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostView =
                (MemoryView<MemorySegment>) Tensor.of(view).to(Device.PANAMA).materialize();
        return hostView;
    }
}
