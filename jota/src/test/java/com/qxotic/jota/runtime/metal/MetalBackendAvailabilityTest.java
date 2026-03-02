package com.qxotic.jota.runtime.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.testutil.ExternalToolChecks;
import org.junit.jupiter.api.Test;

class MetalBackendAvailabilityTest {

    @Test
    void metalBackendMustBeAvailableInMetalProfile() {
        boolean xcrunAvailable = ExternalToolChecks.hasVersionCommand("xcrun");
        String details = MetalTestAssumptions.diagnosticsSummary(xcrunAvailable);
        assertTrue(MetalRuntime.isAvailable(), "Metal JNI runtime not available\n" + details);
        assertTrue(
                Environment.current().runtimes().hasRuntime(Device.METAL),
                "Metal runtime is not registered\n" + details);
        assertTrue(xcrunAvailable, "xcrun not available\n" + details);
        assertTrue(MetalRuntime.deviceCount() > 0, "No Metal device visible\n" + details);
        assertEquals(Device.METAL, Environment.current().runtimeFor(Device.METAL).device());
    }
}
