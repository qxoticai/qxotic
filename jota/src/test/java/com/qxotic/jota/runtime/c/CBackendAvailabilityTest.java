package com.qxotic.jota.runtime.c;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class CBackendAvailabilityTest {

    @Test
    void cBackendMustBeAvailableInCProfile() {
        String details = diagnosticsSummary();
        assertTrue(CNative.isAvailable(), "C JNI runtime not available\n" + details);
        assertTrue(
                Environment.current().runtimes().hasRuntime(Device.C),
                "C runtime is not registered\n" + details);
        assertTrue(isGccAvailable(), "gcc not available\n" + details);
        assertEquals(Device.C, Environment.current().runtimeFor(Device.C).device());
    }

    private static String diagnosticsSummary() {
        String cDiagnostics =
                Environment.current().runtimeDiagnostics().stream()
                        .filter(d -> d.device().equals(Device.C))
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
                + Environment.current().runtimes().hasRuntime(Device.C)
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
