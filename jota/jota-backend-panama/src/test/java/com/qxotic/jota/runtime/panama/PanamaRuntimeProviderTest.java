package com.qxotic.jota.runtime.panama;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class PanamaRuntimeProviderTest {

    @Test
    void providerToStringIsCompactAndDescriptive() {
        String text = new PanamaRuntimeProvider().toString();
        assertEquals("DeviceRuntimeProvider{deviceType=panama, priority=1000}", text);
    }
}
