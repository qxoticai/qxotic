package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Assumptions;

final class MetalTestAssumptions {

    private MetalTestAssumptions() {}

    static void assumeMetalReady() {
        boolean xcrunAvailable = isXcrunAvailable();
        String details = diagnosticsSummary(xcrunAvailable);
        Assumptions.assumeTrue(
                MetalRuntime.isAvailable(), "Metal runtime not available\n" + details);
        Assumptions.assumeTrue(
                Environment.current().runtimes().hasRuntime(Device.METAL),
                "Metal runtime is not registered\n" + details);
        Assumptions.assumeTrue(xcrunAvailable, "xcrun not available\n" + details);
        Assumptions.assumeTrue(
                MetalRuntime.deviceCount() > 0, "No Metal device visible\n" + details);
    }

    static String diagnosticsSummary() {
        return diagnosticsSummary(isXcrunAvailable());
    }

    static String diagnosticsSummary(boolean xcrunAvailable) {
        String diagnostics =
                Environment.current().runtimeDiagnostics().stream()
                        .filter(d -> d.device().equals(Device.METAL))
                        .map(MetalTestAssumptions::formatDiagnostic)
                        .collect(Collectors.joining("\n"));
        if (diagnostics.isBlank()) {
            diagnostics = "<none>";
        }
        return "Metal diagnostics:\n"
                + diagnostics
                + "\nMetalRuntime.isAvailable="
                + MetalRuntime.isAvailable()
                + "\nMetal runtime registered="
                + Environment.current().runtimes().hasRuntime(Device.METAL)
                + "\nxcrun available="
                + xcrunAvailable;
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

    private static boolean isXcrunAvailable() {
        return ExternalToolChecks.hasVersionCommand("xcrun");
    }
}
