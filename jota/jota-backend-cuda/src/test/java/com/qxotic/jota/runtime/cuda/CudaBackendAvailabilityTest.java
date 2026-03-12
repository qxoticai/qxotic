package com.qxotic.jota.runtime.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class CudaBackendAvailabilityTest {

    @Test
    void cudaBackendMustBeAvailableInCudaProfile() {
        boolean nvccAvailable = isNvccAvailable();
        String details = diagnosticsSummary(nvccAvailable);
        assertTrue(
                CudaRuntime.isAvailable(),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] CUDA JNI runtime not available",
                        "mvnd -Pcuda test",
                        details));
        assertTrue(
                Environment.current().runtimes().hasRuntime(Device.CUDA),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] CUDA runtime is not registered",
                        "mvnd -Pcuda test",
                        details));
        assertTrue(
                nvccAvailable,
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] nvcc not available", "mvnd -Pcuda test", details));

        int deviceCount = CudaRuntime.deviceCount();
        assertTrue(
                deviceCount > 0,
                canaryFailureMessage(
                        "[UNSUPPORTED_HARDWARE] CUDA reported no visible device",
                        "mvnd -Pcuda test",
                        details));
        assertEquals(Device.CUDA, Environment.current().runtimeFor(Device.CUDA).device());
    }

    private static String canaryFailureMessage(String reason, String command, String details) {
        return "[BACKEND CANARY] "
                + reason
                + "\nRequested profile command: "
                + command
                + "\n"
                + "Install/enable the CUDA runtime toolchain for this machine, or run without"
                + " -Pcuda.\n\n"
                + details;
    }

    private static String diagnosticsSummary() {
        return diagnosticsSummary(isNvccAvailable());
    }

    private static String diagnosticsSummary(boolean nvccAvailable) {
        String diagnostics =
                Environment.current().runtimeDiagnostics().stream()
                        .filter(d -> d.device().equals(Device.CUDA))
                        .map(CudaBackendAvailabilityTest::formatDiagnostic)
                        .collect(Collectors.joining("\n"));
        if (diagnostics.isBlank()) {
            diagnostics = "<none>";
        }
        return "CUDA diagnostics:\n"
                + diagnostics
                + "\nCudaRuntime.isAvailable="
                + CudaRuntime.isAvailable()
                + "\nCUDA runtime registered="
                + Environment.current().runtimes().hasRuntime(Device.CUDA)
                + "\nnvcc available="
                + nvccAvailable;
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

    private static boolean isNvccAvailable() {
        return ExternalToolChecks.hasVersionCommand("nvcc");
    }
}
