package ai.qxotic.jota.backend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.panama.PanamaBackend;
import org.junit.jupiter.api.Test;

class BackendRegistryTest {

    @Test
    void registersNativeBackend() {
        DefaultBackendRegistry registry =
                DefaultBackendRegistry.withNative(new PanamaBackend());
        assertNotNull(registry.nativeBackend());
        assertEquals(Device.PANAMA, registry.nativeBackend().device());
        assertNotNull(registry.backend(Device.PANAMA));
    }
}
