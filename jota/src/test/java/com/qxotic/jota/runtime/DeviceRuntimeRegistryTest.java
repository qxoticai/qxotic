package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.Device;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.runtime.panama.PanamaDeviceRuntime;
import com.qxotic.jota.tensor.ComputeEngine;
import com.qxotic.jota.tensor.Tensor;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class DeviceRuntimeRegistryTest {

    @Test
    void registersNativeRuntimeFor() {
        DefaultRuntimeRegistry registry =
                DefaultRuntimeRegistry.withNative(new PanamaDeviceRuntime());
        assertNotNull(registry.nativeRuntime());
        assertEquals(Device.PANAMA, registry.nativeRuntime().device());
        assertNotNull(registry.runtimeFor(Device.PANAMA));
    }

    @Test
    void registerRejectsDuplicateDeviceRuntime() {
        DefaultRuntimeRegistry registry =
                DefaultRuntimeRegistry.withNative(new PanamaDeviceRuntime());
        assertThrows(
                IllegalStateException.class, () -> registry.register(new PanamaDeviceRuntime()));
    }

    @Test
    void registerNativeRejectsNonPanamaDevice() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        DeviceRuntime nonNative =
                new StubDeviceRuntime(
                        Device.C, DomainFactory.ofMemorySegment(), new NoopEngine(Device.C));
        assertThrows(IllegalArgumentException.class, () -> registry.registerNative(nonNative));
    }

    private record NoopEngine(Device device) implements ComputeEngine {
        @Override
        public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
            throw new UnsupportedOperationException();
        }
    }

    private record StubDeviceRuntime(
            Device device, MemoryDomain<?> memoryDomain, ComputeEngine computeEngine)
            implements DeviceRuntime {
        @Override
        public Optional<KernelService> kernelService() {
            return Optional.empty();
        }
    }
}
