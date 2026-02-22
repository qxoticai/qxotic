package com.qxotic.jota.runtime.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class HipBackendAvailabilityTest {

    @Test
    void hipBackendMustBeAvailableInHipProfile() {
        String details = diagnosticsSummary();
        assertTrue(HipRuntime.isAvailable(), "HIP JNI runtime not available\n" + details);
        assertTrue(
                Environment.current().runtimes().hasRuntime(Device.HIP),
                "HIP runtime is not registered\n" + details);
        assertTrue(isHipccAvailable(), "hipcc not available\n" + details);

        int deviceCount = HipRuntime.deviceCount();
        assertTrue(deviceCount > 0, "HIP reported no visible device\n" + details);
        assertEquals(Device.HIP, Environment.current().runtimeFor(Device.HIP).device());
    }

    private static String diagnosticsSummary() {
        String hipDiagnostics =
                Environment.current().runtimeDiagnostics().stream()
                        .filter(d -> d.device().equals(Device.HIP))
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
                + Environment.current().runtimes().hasRuntime(Device.HIP)
                + "\nhipcc available="
                + isHipccAvailable();
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
        try {
            Process process = new ProcessBuilder("hipcc", "--version").start();
            int code = process.waitFor();
            return code == 0;
        } catch (Exception e) {
            return false;
        }
    }
}
