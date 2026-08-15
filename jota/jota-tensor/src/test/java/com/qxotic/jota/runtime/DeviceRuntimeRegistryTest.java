package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.tensor.Tensor;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class DeviceRuntimeRegistryTest {

    @Test
    void registersFactoryAndResolvesRuntime() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device logical = DeviceType.PANAMA.deviceIndex(0);
        DeviceRuntime runtime = panamaRuntime(logical);
        registry.registerFactory(logical, d -> runtime);

        assertSame(runtime, registry.runtimeFor(logical));
        assertTrue(registry.hasRuntimeFor(logical));
    }

    @Test
    void resolvesRegisteredConcreteDevice() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device logical = DeviceType.PANAMA.deviceIndex(0);
        DeviceRuntime runtime = panamaRuntime(logical);
        registry.registerFactory(logical, d -> runtime);

        assertSame(runtime, registry.runtimeFor(logical));
        assertTrue(registry.hasRuntimeFor(logical));
    }

    @Test
    void factoryIsLazy() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device logical = DeviceType.PANAMA.deviceIndex(0);
        AtomicInteger createCount = new AtomicInteger(0);
        registry.registerFactory(
                logical,
                d -> {
                    createCount.incrementAndGet();
                    return panamaRuntime(logical);
                });

        assertEquals(0, createCount.get());
        registry.runtimeFor(logical);
        assertEquals(1, createCount.get());
        registry.runtimeFor(logical);
        assertEquals(1, createCount.get());
    }

    @Test
    void registerFactoryRejectsDuplicates() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device logical = DeviceType.PANAMA.deviceIndex(0);
        registry.registerFactory(logical, d -> panamaRuntime(logical));
        assertThrows(
                IllegalStateException.class,
                () -> registry.registerFactory(logical, d -> panamaRuntime(logical)));
    }

    @Test
    void hasRuntimeForReturnsFalseForUnknownDevice() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        assertFalse(registry.hasRuntimeFor(DeviceType.CUDA.deviceIndex(0)));
    }

    @Test
    void devicesReturnsOnlyConcreteDevices() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device panamaLogical = DeviceType.PANAMA.deviceIndex(0);
        registry.registerFactory(panamaLogical, d -> panamaRuntime(panamaLogical));

        assertTrue(registry.devices().contains(panamaLogical));
    }

    @Test
    void runtimeFactoryMustReturnRuntimeBoundToRequestedDevice() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device requested = DeviceType.CUDA.deviceIndex(1);
        Device wrong = DeviceType.CUDA.deviceIndex(0);
        registry.registerFactory(
                requested, d -> new StubDeviceRuntime(wrong, null, new NoopEngine(wrong)));

        IllegalStateException error =
                assertThrows(IllegalStateException.class, () -> registry.runtimeFor(requested));
        assertTrue(error.getMessage().contains("returned runtime bound to"));
    }

    @Test
    void runtimeFactoryMustNotReturnNullRuntime() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device requested = DeviceType.CUDA.deviceIndex(0);
        registry.registerFactory(requested, d -> null);

        IllegalStateException error =
                assertThrows(IllegalStateException.class, () -> registry.runtimeFor(requested));
        assertTrue(error.getMessage().contains("returned null"));
    }

    private static DeviceRuntime panamaRuntime(Device device) {
        return new StubDeviceRuntime(
                device, DomainFactory.ofMemorySegment(), new NoopEngine(device));
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
        public boolean supportsNativeRuntimeAlias() {
            return device.belongsTo(DeviceType.PANAMA) || device.belongsTo(DeviceType.C);
        }

        @Override
        public Optional<KernelService> kernelService() {
            return Optional.empty();
        }
    }
}
