package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import java.util.List;
import org.junit.jupiter.api.Test;

class DeviceRuntimeProvidersTest {

    @Test
    void availableProvidersReturnsImmutableList() {
        List<DeviceRuntimeProvider> providers = DeviceRuntimeProvider.availableProviders();
        assertNotNull(providers);
        assertThrows(UnsupportedOperationException.class, () -> providers.add(null));
    }
}
