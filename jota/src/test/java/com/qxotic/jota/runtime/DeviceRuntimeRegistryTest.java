package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.qxotic.jota.Device;
import com.qxotic.jota.runtime.panama.PanamaDeviceRuntime;
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
