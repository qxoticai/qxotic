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
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import com.qxotic.jota.tensor.Tensor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class EnvironmentTest {

    @Test
    void currentDefaultsToGlobal() {
        assertSame(Environment.global(), Environment.current());
        assertEquals(Device.NATIVE, Environment.global().defaultDevice());
    }

    @Test
    void scopedEnvironmentOverridesDefaults() {
        Environment env =
                new Environment(Device.NATIVE, DataType.FP64, Environment.global().runtimes());

        Environment.with(
                env,
                () -> {
                    assertEquals(Device.NATIVE, Device.defaultDevice());
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
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new Environment(
                                Device.NATIVE, DataType.I32, Environment.global().runtimes()));
    }

    @Test
    void constructorRejectsMissingDefaultDeviceRuntime() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        assertThrows(
                IllegalArgumentException.class,
                () -> new Environment(Device.C, DataType.FP32, registry));
    }

    @Test
    void registryExposesRegisteredDevices() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.register(new StubDeviceRuntime(DomainFactory.ofBytes(), dummyRuntime()));
        registry.register(new StubDeviceRuntime(DomainFactory.ofMemorySegment(), dummyRuntime()));

        assertTrue(registry.devices().contains(Device.PANAMA));
        assertTrue(registry.devices().contains(Device.NATIVE));
    }

    @Test
    void missingNativeRuntimeMessageIncludesBackendFixesAndDiagnostics() throws Exception {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.addDiagnostic(
                new RuntimeDiagnostic(
                        "panama-core",
                        Device.PANAMA,
                        RuntimeProbe.missingSoftware(
                                "Panama runtime unavailable",
                                "Include com.qxotic:jota or com.qxotic:jota-backend-panama")));
        registry.addDiagnostic(
                new RuntimeDiagnostic(
                        "c-runtime",
                        Device.C,
                        RuntimeProbe.missingSoftware(
                                "C backend unavailable",
                                "Include com.qxotic:jota-backend-c and make gcc available")));

        IllegalStateException error =
                invokeMissingNativeRuntimeException(registry, "No compatible runtime available");

        String message = error.getMessage();
        assertNotNull(message);
        assertTrue(message.contains("Unable to configure Device.NATIVE runtime"));
        assertTrue(message.contains("Runtime probe diagnostics:"));
        assertTrue(message.contains("panama-core"));
        assertTrue(message.contains("c-runtime"));
        assertTrue(message.contains("com.qxotic:jota-backend-panama"));
        assertTrue(message.contains("com.qxotic:jota-backend-c"));
        assertTrue(message.contains("com.qxotic:jota-graal"));
        assertTrue(message.contains("-Djota.native.backend=panama or c"));
    }

    @Test
    void unavailableOverrideMessageIncludesActionableBackendHints() throws Exception {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        String previous = System.getProperty("jota.native.backend");
        System.setProperty("jota.native.backend", "panama");
        try {
            IllegalStateException error = invokeSelectNativeBackendFailure(registry);
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

    private static IllegalStateException invokeMissingNativeRuntimeException(
            DefaultRuntimeRegistry registry, String reason) throws Exception {
        Method method =
                Environment.class.getDeclaredMethod(
                        "missingNativeRuntimeException",
                        com.qxotic.jota.runtime.RuntimeRegistry.class,
                        String.class);
        method.setAccessible(true);
        return (IllegalStateException) method.invoke(null, registry, reason);
    }

    private static IllegalStateException invokeSelectNativeBackendFailure(
            DefaultRuntimeRegistry registry) throws Exception {
        Method method =
                Environment.class.getDeclaredMethod(
                        "selectNativeBackend", com.qxotic.jota.runtime.RuntimeRegistry.class);
        method.setAccessible(true);
        try {
            method.invoke(null, registry);
            fail("Expected selectNativeBackend to fail");
            return new IllegalStateException("unreachable");
        } catch (InvocationTargetException e) {
            assertInstanceOf(IllegalStateException.class, e.getCause());
            return (IllegalStateException) e.getCause();
        }
    }

    private ComputeEngine dummyRuntime() {
        return new ComputeEngine() {
            @Override
            public Device device() {
                return Device.PANAMA;
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
            return device.equals(Device.PANAMA) || device.equals(Device.C);
        }

        @Override
        public Optional<KernelService> kernelService() {
            return Optional.empty();
        }
    }
}
