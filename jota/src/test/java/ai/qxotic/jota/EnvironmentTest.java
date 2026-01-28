package ai.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.backend.DefaultBackendRegistry;
import ai.qxotic.jota.backend.KernelService;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.ComputeBackend;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.KernelCache;
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
        registry.register(new StubBackend(ContextFactory.ofBytes(), dummyEngine()));
        registry.register(new StubBackend(ContextFactory.ofMemorySegment(), dummyEngine()));

        assertTrue(registry.devices().contains(Device.PANAMA));
    }

    private ComputeEngine dummyEngine() {
        return new ComputeEngine() {
            @Override
            public ComputeBackend backendFor(Device device) {
                throw new UnsupportedOperationException("No backend for " + device);
            }

            @Override
            public KernelCache cache() {
                return DiskKernelCache.defaultCache();
            }
        };
    }

    private static final class StubBackend implements ai.qxotic.jota.backend.Backend {
        private final ai.qxotic.jota.memory.MemoryContext<?> context;
        private final ComputeEngine engine;

        private StubBackend(ai.qxotic.jota.memory.MemoryContext<?> context, ComputeEngine engine) {
            this.context = context;
            this.engine = engine;
        }

        @Override
        public Device device() {
            return context.device();
        }

        @Override
        public ai.qxotic.jota.memory.MemoryContext<?> memoryContext() {
            return context;
        }

        @Override
        public ComputeEngine computeEngine() {
            return engine;
        }

        @Override
        public java.util.Optional<KernelService> kernels() {
            return java.util.Optional.empty();
        }
    }
}
