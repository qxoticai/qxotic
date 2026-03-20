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
        assertTrue(registry.hasRuntime(logical));
    }

    @Test
    void resolvesAliasToConcreteDevice() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device logical = DeviceType.PANAMA.deviceIndex(0);
        DeviceRuntime runtime = panamaRuntime(logical);
        registry.registerFactory(logical, d -> runtime);

        // Auto-alias: "panama" → Device("panama", 0)
        assertSame(runtime, registry.runtimeFor("panama"));
        assertTrue(registry.hasRuntime("panama"));
    }

    @Test
    void aliasResolutionIsTransitive() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device logical = DeviceType.PANAMA.deviceIndex(0);
        DeviceRuntime runtime = panamaRuntime(logical);
        registry.registerFactory(logical, d -> runtime);

        // native → panama → panama:0
        registry.registerAlias("native", logical);

        assertSame(runtime, registry.runtimeFor("native"));
        assertEquals(logical, registry.resolve("native"));
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
    void hasRuntimeReturnsFalseForUnknownAlias() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        assertFalse(registry.hasRuntime("cuda"));
    }

    @Test
    void registerAliasCanSwitchTarget() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device panamaLogical = DeviceType.PANAMA.deviceIndex(0);
        Device cLogical = DeviceType.C.deviceIndex(0);
        DeviceRuntime panamaRt = panamaRuntime(panamaLogical);
        DeviceRuntime cRt = cRuntime(cLogical);
        registry.registerFactory(panamaLogical, d -> panamaRt);
        registry.registerFactory(cLogical, d -> cRt);

        registry.registerAlias("native", panamaLogical);
        assertSame(panamaRt, registry.runtimeFor("native"));

        registry.registerAlias("native", cLogical);
        assertSame(cRt, registry.runtimeFor("native"));
    }

    @Test
    void devicesReturnsOnlyConcreteDevices() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device panamaLogical = DeviceType.PANAMA.deviceIndex(0);
        registry.registerFactory(panamaLogical, d -> panamaRuntime(panamaLogical));
        registry.registerAlias("native", panamaLogical);

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

    private static DeviceRuntime cRuntime(Device device) {
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
