package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DefaultRuntimeRegistry;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import com.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class EnvironmentTest {

    @Test
    void currentDefaultsToGlobal() {
        assertSame(Environment.global(), Environment.current());
        assertTrue(
                Environment.global().defaultDevice().belongsTo(DeviceType.PANAMA)
                        || Environment.global().defaultDevice().belongsTo(DeviceType.C));
    }

    @Test
    void scopedEnvironmentOverridesDefaults() {
        Device nativeDevice = Environment.global().nativeDevice();
        Environment env =
                Environment.of(nativeDevice, DataType.FP64, Environment.global().runtimes());

        Environment.with(
                env,
                () -> {
                    assertEquals(nativeDevice, Device.defaultDevice());
                    assertEquals(DataType.FP64, DataType.defaultFloat());
                    return null;
                });
    }

    @Test
    void configureGlobalOnlyOnce() {
        Environment env = Environment.global();
        Environment.configureGlobal(env);

        assertSame(env, Environment.global());
        assertThrows(IllegalStateException.class, () -> Environment.configureGlobal(env));
    }

    @Test
    void constructorRejectsNonFloatingDefaultFloat() {
        Device nativeDevice = Environment.global().nativeDevice();
        assertThrows(
                IllegalArgumentException.class,
                () -> Environment.of(nativeDevice, DataType.I32, Environment.global().runtimes()));
    }

    @Test
    void constructorRejectsMissingDefaultDeviceRuntime() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        assertThrows(
                IllegalArgumentException.class,
                () -> Environment.of(DeviceType.C.deviceIndex(0), DataType.FP32, registry));
    }

    @Test
    void registryExposesRegisteredDevices() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        Device javaLogical = DeviceType.JAVA.deviceIndex(0);
        Device panamaLogical = DeviceType.PANAMA.deviceIndex(0);
        registry.registerFactory(
                javaLogical, d -> new StubDeviceRuntime(DomainFactory.ofBytes(), dummyRuntime()));
        registry.registerFactory(
                panamaLogical,
                d -> new StubDeviceRuntime(DomainFactory.ofMemorySegment(), dummyRuntime()));

        assertTrue(registry.devices().contains(javaLogical));
        assertTrue(registry.devices().contains(panamaLogical));
    }

    @Test
    void missingNativeRuntimeMessageIncludesBackendFixesAndDiagnostics() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.addDiagnostic(
                new RuntimeDiagnostic(
                        "panama-core",
                        DeviceType.PANAMA,
                        RuntimeProbe.missingSoftware(
                                "Panama runtime unavailable",
                                "Include com.qxotic:jota-backend-panama")));
        registry.addDiagnostic(
                new RuntimeDiagnostic(
                        "c-runtime",
                        DeviceType.C,
                        RuntimeProbe.missingSoftware(
                                "C backend unavailable",
                                "Include com.qxotic:jota-backend-c and make gcc available")));

        IllegalStateException error =
                EnvironmentImpl.missingNativeRuntimeException(
                        registry, "No compatible runtime available");

        String message = error.getMessage();
        assertNotNull(message);
        assertTrue(message.contains("Unable to configure native runtime"));
        assertTrue(message.contains("Runtime probe diagnostics:"));
        assertTrue(message.contains("panama-core"));
        assertTrue(message.contains("c-runtime"));
        assertTrue(message.contains("com.qxotic:jota-backend-panama"));
        assertTrue(message.contains("com.qxotic:jota-backend-c"));
        assertTrue(message.contains("com.qxotic:jota-graal"));
        assertTrue(message.contains("-Djota.native.backend=<backend-id>"));
    }

    @Test
    void unavailableOverrideMessageIncludesActionableBackendHints() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        String previous = System.getProperty("jota.native.backend");
        System.setProperty("jota.native.backend", "panama");
        try {
            IllegalStateException error =
                    assertThrows(
                            IllegalStateException.class,
                            () -> EnvironmentImpl.selectNativeBackend(registry));
            String message = error.getMessage();
            assertNotNull(message);
            assertTrue(
                    message.contains("Configured jota.native.backend selects unavailable backend"));
            assertTrue(message.contains("com.qxotic:jota-backend-panama"));
            assertTrue(message.contains("com.qxotic:jota-backend-c"));
            assertTrue(message.contains("jota.native.backend: panama"));
        } finally {
            if (previous == null) {
                System.clearProperty("jota.native.backend");
            } else {
                System.setProperty("jota.native.backend", previous);
            }
        }
    }

    @Test
    void missingNativeRuntimeMessageIncludesBackendExcludeProperty() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        String previousInclude = System.getProperty("jota.backends.include");
        String previous = System.getProperty("jota.backends.exclude");
        System.setProperty("jota.backends.include", "c,panama");
        System.setProperty("jota.backends.exclude", "panama,opencl");
        try {
            IllegalStateException error =
                    EnvironmentImpl.missingNativeRuntimeException(
                            registry, "No compatible runtime available");
            String message = error.getMessage();
            assertNotNull(message);
            assertTrue(message.contains("jota.backends.include: c,panama"));
            assertTrue(message.contains("jota.backends.exclude: panama,opencl"));
            assertTrue(message.contains("include/exclude filters"));
        } finally {
            if (previousInclude == null) {
                System.clearProperty("jota.backends.include");
            } else {
                System.setProperty("jota.backends.include", previousInclude);
            }
            if (previous == null) {
                System.clearProperty("jota.backends.exclude");
            } else {
                System.setProperty("jota.backends.exclude", previous);
            }
        }
    }

    @Test
    void excludedBackendIsNotRegisteredEvenWhenProviderIsAvailable() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        DeviceRuntimeProvider provider = new AlwaysAvailableProvider("opencl", DeviceType.OPENCL);

        EnvironmentImpl.registerProvider(registry, provider, Set.of(), Set.of("opencl"));

        assertFalse(registry.hasRuntimeFor(DeviceType.OPENCL.deviceIndex(0)));
        assertTrue(
                registry.diagnostics().stream()
                        .anyMatch(
                                diagnostic ->
                                        diagnostic.providerId().equals("opencl")
                                                && diagnostic
                                                        .probe()
                                                        .message()
                                                        .contains("excluded by configuration")));
    }

    @Test
    void excludeListTakesPriorityOverIncludeList() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        DeviceRuntimeProvider provider = new AlwaysAvailableProvider("opencl", DeviceType.OPENCL);

        EnvironmentImpl.registerProvider(registry, provider, Set.of("opencl"), Set.of("opencl"));

        assertFalse(registry.hasRuntimeFor(DeviceType.OPENCL.deviceIndex(0)));
        assertTrue(
                registry.diagnostics().stream()
                        .anyMatch(
                                diagnostic ->
                                        diagnostic.providerId().equals("opencl")
                                                && diagnostic
                                                        .probe()
                                                        .hint()
                                                        .contains("exclude wins over include")));
    }

    @Test
    void parseNativeBackendOverrideRejectsUnsupportedBackends() {
        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> EnvironmentImpl.parseNativeBackendOverride("hip"));
        assertTrue(error.getMessage().contains("Supported values: auto, native, panama, c"));
    }

    @Test
    void nativeRuntimeAllocatesMemorySegmentBackedMemory() {
        DeviceRuntime nativeRuntime = Environment.nativeRuntime();
        var memory = Environment.nativeMemoryDomain().memoryAllocator().allocateMemory(16);

        assertEquals(Environment.current().nativeDevice(), nativeRuntime.device());
        assertInstanceOf(MemorySegment.class, memory.base());
    }

    private static ComputeEngine dummyRuntime() {
        return new ComputeEngine() {
            @Override
            public Device device() {
                return DeviceType.PANAMA.deviceIndex(0);
            }

            @Override
            public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
                throw new UnsupportedOperationException("No backend execution in this test");
            }
        };
    }

    private static final class StubDeviceRuntime implements DeviceRuntime {
        private final MemoryDomain<?> memoryDomain;
        private final ComputeEngine computeEngine;

        private StubDeviceRuntime(MemoryDomain<?> domain, ComputeEngine computeEngine) {
            this.memoryDomain = domain;
            this.computeEngine = computeEngine;
        }

        @Override
        public Device device() {
            return memoryDomain.device();
        }

        @Override
        public MemoryDomain<?> memoryDomain() {
            return memoryDomain;
        }

        @Override
        public ComputeEngine computeEngine() {
            return computeEngine;
        }

        @Override
        public boolean supportsNativeRuntimeAlias() {
            Device device = device();
            return device.belongsTo(DeviceType.PANAMA) || device.belongsTo(DeviceType.C);
        }

        @Override
        public Optional<KernelService> kernelService() {
            return Optional.empty();
        }
    }

    private static final class AlwaysAvailableProvider extends DeviceRuntimeProvider {
        private final DeviceType deviceType;

        private AlwaysAvailableProvider(String id, DeviceType deviceType) {
            this.deviceType = deviceType;
        }

        @Override
        public DeviceType deviceType() {
            return deviceType;
        }

        @Override
        public RuntimeProbe probe() {
            return RuntimeProbe.available("available");
        }

        @Override
        protected DeviceRuntime createForDevice(Device device) {
            return new StubDeviceRuntime(DomainFactory.ofMemorySegment(), dummyRuntime());
        }
    }
}
