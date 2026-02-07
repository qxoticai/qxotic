package ai.qxotic.jota.runtime.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.runtime.RuntimeDiagnostic;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Assumptions;

final class HipTestAssumptions {

    private HipTestAssumptions() {}

    static void assumeHipReady() {
        String details = diagnosticsSummary();
        Assumptions.assumeTrue(HipRuntime.isAvailable(), "HIP runtime not available\n" + details);
        Assumptions.assumeTrue(
                Environment.current().runtimes().hasRuntime(Device.HIP),
                "HIP runtime is not registered\n" + details);
        Assumptions.assumeTrue(isHipccAvailable(), "hipcc not available\n" + details);
    }

    static String diagnosticsSummary() {
        String hipDiagnostics =
                Environment.current().runtimeDiagnostics().stream()
                        .filter(d -> d.device().equals(Device.HIP))
                        .map(HipTestAssumptions::formatDiagnostic)
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
