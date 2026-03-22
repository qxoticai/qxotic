package com.qxotic.jota.runtime.hip;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class HipBackendAvailabilityTest {

    @Test
    void hipBackendMustBeAvailableInHipProfile() {
        boolean hipccAvailable = isHipccAvailable();
        String details = diagnosticsSummary(hipccAvailable);
        assertTrue(
                HipRuntime.isAvailable(),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] HIP JNI runtime not available",
                        "mvnd -Phip test",
                        details));
        assertTrue(
                Environment.hasRuntimeFor(DeviceType.HIP.deviceIndex(0)),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] HIP runtime is not registered",
                        "mvnd -Phip test",
                        details));
        assertTrue(
                hipccAvailable,
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] hipcc not available", "mvnd -Phip test", details));

        int deviceCount = HipRuntime.deviceCount();
        assertTrue(
                deviceCount > 0,
                canaryFailureMessage(
                        "[UNSUPPORTED_HARDWARE] HIP reported no visible device",
                        "mvnd -Phip test",
                        details));
        assertTrue(
                Environment.runtimeFor(DeviceType.HIP.deviceIndex(0))
                        .device()
                        .belongsTo(DeviceType.HIP));
    }

    private static String canaryFailureMessage(String reason, String command, String details) {
        return "[BACKEND CANARY] "
                + reason
                + "\nRequested profile command: "
                + command
                + "\n"
                + "Install/enable the HIP runtime toolchain for this machine, or run without"
                + " -Phip.\n\n"
                + details;
    }

    private static String diagnosticsSummary() {
        return diagnosticsSummary(isHipccAvailable());
    }

    private static String diagnosticsSummary(boolean hipccAvailable) {
        String hipDiagnostics =
                Environment.runtimeDiagnostics().stream()
                        .filter(d -> d.deviceType().equals(DeviceType.HIP))
                        .map(HipBackendAvailabilityTest::formatDiagnostic)
                        .collect(Collectors.joining("\n"));
        if (hipDiagnostics.isBlank()) {
            hipDiagnostics = "<none>";
        }
        return "HIP diagnostics:\n"
                + hipDiagnostics
                + "\nHipRuntime.isAvailable="
                + HipRuntime.isAvailable()
                + "\nHIP runtime registered="
                + Environment.hasRuntimeFor(DeviceType.HIP.deviceIndex(0))
                + "\nhipcc available="
                + hipccAvailable;
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

    private static boolean isHipccAvailable() {
        return ExternalToolChecks.hasVersionCommand("hipcc");
    }
}
