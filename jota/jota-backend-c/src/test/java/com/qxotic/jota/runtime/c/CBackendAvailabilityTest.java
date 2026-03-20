package com.qxotic.jota.runtime.c;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class CBackendAvailabilityTest {

    @Test
    void cBackendMustBeAvailableInCProfile() {
        String details = diagnosticsSummary();
        assertTrue(
                CNative.isAvailable(),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] C JNI runtime not available",
                        "mvnd -Pc test",
                        details));
        assertTrue(
                ConfiguredTestDevice.hasRuntime(DeviceType.C),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] C runtime is not registered",
                        "mvnd -Pc test",
                        details));
        assertTrue(
                isGccAvailable(),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] gcc not available", "mvnd -Pc test", details));
        assertEquals(
                DeviceType.C.deviceIndex(0),
                Environment.current()
                        .runtimeFor(ConfiguredTestDevice.resolve(DeviceType.C))
                        .device());
    }

    private static String canaryFailureMessage(String reason, String command, String details) {
        return "[BACKEND CANARY] "
                + reason
                + "\nRequested profile command: "
                + command
                + "\nInstall/enable the C toolchain and JNI runtime, or run without -Pc.\n\n"
                + details;
    }

    private static String diagnosticsSummary() {
        String cDiagnostics =
                Environment.current().runtimeDiagnostics().stream()
                        .filter(d -> d.deviceType().equals(DeviceType.C))
                        .map(CBackendAvailabilityTest::formatDiagnostic)
                        .collect(Collectors.joining("\n"));
        if (cDiagnostics.isBlank()) {
            cDiagnostics = "<none>";
        }
        return "C diagnostics:\n"
                + cDiagnostics
                + "\nCNative.isAvailable="
                + CNative.isAvailable()
                + "\nC runtime registered="
                + ConfiguredTestDevice.hasRuntime(DeviceType.C)
                + "\ngcc available="
                + isGccAvailable();
    }

    private static String formatDiagnostic(RuntimeDiagnostic d) {
        String cause =
                d.probe().cause() == null
                        ? ""
                        : " cause="
                                + d.probe().cause().getClass().getSimpleName()
                                + ":"
                                + d.probe().cause().getMessage();
        String hint = d.probe().hint() == null ? "" : " hint=" + d.probe().hint();
        return "- provider="
                + d.providerId()
                + " status="
                + d.probe().status()
                + " message="
                + d.probe().message()
                + hint
                + cause;
    }

    private static boolean isGccAvailable() {
        return ExternalToolChecks.hasVersionCommand("gcc");
    }
}
