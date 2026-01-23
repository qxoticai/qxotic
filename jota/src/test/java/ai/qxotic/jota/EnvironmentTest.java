package ai.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        Environment env = new Environment(Device.PANAMA, DataType.FP64, DeviceRegistry.global());

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
        DeviceRegistry registry =
                DeviceRegistry.builder()
                        .register(
                                ContextFactory.ofBytes(), dummyEngine())
                        .register(
                                ContextFactory.ofMemorySegment(),
                                dummyEngine())
                        .build();

        assertTrue(registry.devices().contains(Device.PANAMA));
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
}
