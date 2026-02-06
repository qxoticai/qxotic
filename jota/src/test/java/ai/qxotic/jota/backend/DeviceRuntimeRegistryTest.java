package ai.qxotic.jota.backend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.panama.PanamaDeviceRuntime;
import org.junit.jupiter.api.Test;

class DeviceRuntimeRegistryTest {

    @Test
    void registersNativeBackend() {
        DefaultBackendRegistry registry =
                DefaultBackendRegistry.withNative(new PanamaDeviceRuntime());
        assertNotNull(registry.nativeBackend());
        assertEquals(Device.PANAMA, registry.nativeBackend().device());
        assertNotNull(registry.backend(Device.PANAMA));
    }
}
