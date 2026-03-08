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
        boolean metalCompilerAvailable =
                ExternalToolChecks.hasCommand("xcrun", "-sdk", "macosx", "-find", "metal");
        boolean metallibToolAvailable =
                ExternalToolChecks.hasCommand("xcrun", "-sdk", "macosx", "-find", "metallib");
        String details =
                MetalTestAssumptions.diagnosticsSummary(
                        xcrunAvailable, metalCompilerAvailable, metallibToolAvailable);
        assertTrue(
                MetalRuntime.isAvailable(),
                canaryFailureMessage(
                        "Metal JNI runtime not available", "mvnd -Pmetal test", details));
        assertTrue(
                Environment.current().runtimes().hasRuntime(Device.METAL),
                canaryFailureMessage(
                        "Metal runtime is not registered", "mvnd -Pmetal test", details));
        assertTrue(
                xcrunAvailable,
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] xcrun not available", "mvnd -Pmetal test", details));
        assertTrue(
                metalCompilerAvailable,
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] xcrun metal tool not available",
                        "mvnd -Pmetal test",
                        details));
        assertTrue(
                metallibToolAvailable,
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] xcrun metallib tool not available",
                        "mvnd -Pmetal test",
                        details));
        assertTrue(
                MetalRuntime.deviceCount() > 0,
                canaryFailureMessage(
                        "[UNSUPPORTED_HARDWARE] no Metal device visible",
                        "mvnd -Pmetal test",
                        details));
        assertEquals(Device.METAL, Environment.current().runtimeFor(Device.METAL).device());
    }

    private static String canaryFailureMessage(String reason, String command, String details) {
        return "[BACKEND CANARY] "
                + reason
                + "\nRequested profile command: "
                + command
                + "\n"
                + "Install/enable Metal runtime support on this machine, or run without"
                + " -Pmetal.\n\n"
                + details;
    }
}
