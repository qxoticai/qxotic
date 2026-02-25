package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Assumptions;

final class HipTestAssumptions {

    private HipTestAssumptions() {}

    static void assumeHipReady() {
        boolean hipccAvailable = isHipccAvailable();
        String details = diagnosticsSummary(hipccAvailable);
        Assumptions.assumeTrue(HipRuntime.isAvailable(), "HIP runtime not available\n" + details);
        Assumptions.assumeTrue(
                Environment.current().runtimes().hasRuntime(Device.HIP),
                "HIP runtime is not registered\n" + details);
        Assumptions.assumeTrue(hipccAvailable, "hipcc not available\n" + details);
    }

    static String diagnosticsSummary() {
        return diagnosticsSummary(isHipccAvailable());
    }

    static String diagnosticsSummary(boolean hipccAvailable) {
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
