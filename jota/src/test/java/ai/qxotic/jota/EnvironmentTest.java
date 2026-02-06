package ai.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.backend.DefaultBackendRegistry;
import ai.qxotic.jota.backend.DeviceRuntime;
import ai.qxotic.jota.backend.KernelService;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.impl.DomainFactory;
import ai.qxotic.jota.tensor.ComputeEngine;
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
                new Environment(
                        Device.PANAMA,
                        DataType.FP64,
                        Environment.global().backends(),
                        ExecutionMode.LAZY);

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
    void registryExposesRegisteredDevices() {
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubDeviceRuntime(DomainFactory.ofBytes(), dummyBackend()));
        registry.register(new StubDeviceRuntime(DomainFactory.ofMemorySegment(), dummyBackend()));

        assertTrue(registry.devices().contains(Device.PANAMA));
    }

    private ComputeEngine dummyBackend() {
        return new ComputeEngine() {
            @Override
            public Device device() {
                return Device.PANAMA;
            }

            @Override
            public ai.qxotic.jota.memory.MemoryView<?> execute(
                    ai.qxotic.jota.ir.tir.TIRGraph graph,
                    java.util.List<ai.qxotic.jota.tensor.Tensor> inputs) {
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
