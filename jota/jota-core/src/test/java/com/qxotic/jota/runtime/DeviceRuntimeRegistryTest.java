package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.Device;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.runtime.panama.PanamaDeviceRuntime;
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
        assertSame(registry.nativeRuntime(), registry.runtimeFor(Device.NATIVE));
    }

    @Test
    void registerRejectsDuplicateDeviceRuntime() {
        DefaultRuntimeRegistry registry =
                DefaultRuntimeRegistry.withNative(new PanamaDeviceRuntime());
        assertThrows(
                IllegalStateException.class, () -> registry.register(new PanamaDeviceRuntime()));
    }

    @Test
    void registerNativeRejectsNonNativeDevice() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        DeviceRuntime nonNative =
                new StubDeviceRuntime(
                        Device.HIP, DomainFactory.ofMemorySegment(), new NoopEngine(Device.HIP));
        assertThrows(IllegalArgumentException.class, () -> registry.registerNative(nonNative));
    }

    @Test
    void registerNativeCanSwitchToCBackend() {
        DefaultRuntimeRegistry registry =
                DefaultRuntimeRegistry.withNative(new PanamaDeviceRuntime());
        DeviceRuntime cRuntime =
                new StubDeviceRuntime(Device.C, DomainFactory.ofMemorySegment(), new NoopEngine(Device.C));
        registry.register(cRuntime);

        registry.registerNative(cRuntime);

        assertSame(cRuntime, registry.nativeRuntime());
        assertSame(cRuntime, registry.runtimeFor(Device.NATIVE));
        assertEquals(Device.C, registry.runtimeFor(Device.NATIVE).device());
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
