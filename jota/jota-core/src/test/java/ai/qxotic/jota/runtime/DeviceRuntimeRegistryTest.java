package ai.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.runtime.panama.PanamaDeviceRuntime;
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
}
