package com.qxotic.jota.runtime.panama;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class PanamaRuntimeProviderTest {

    @Test
    void providerToStringIsCompactAndDescriptive() {
        String text = new PanamaRuntimeProvider().toString();
        assertTrue(text.contains("deviceType=panama"));
        assertTrue(text.contains("priority="));
    }
}
