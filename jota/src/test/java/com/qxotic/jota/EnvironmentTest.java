package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.runtime.DefaultRuntimeRegistry;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.tensor.ComputeEngine;
import com.qxotic.jota.tensor.Tensor;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class EnvironmentTest {

    @Test
    void currentDefaultsToGlobal() {
        assertSame(Environment.global(), Environment.current());
    }

    @Test
    void scopedEnvironmentOverridesDefaults() {
        Environment env =
                new Environment(Device.PANAMA, DataType.FP64, Environment.global().runtimes());

        Environment.with(
                env,
                () -> {
                    assertEquals(Device.PANAMA, Device.defaultDevice());
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
                                Device.PANAMA, DataType.I32, Environment.global().runtimes()));
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
        public Optional<KernelService> kernelService() {
            return Optional.empty();
        }
    }
}
