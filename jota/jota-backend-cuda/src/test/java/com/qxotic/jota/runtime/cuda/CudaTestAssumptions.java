package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Assumptions;

final class CudaTestAssumptions {

    private CudaTestAssumptions() {}

    static void assumeCudaReady() {
        boolean nvccAvailable = isNvccAvailable();
        String details = diagnosticsSummary(nvccAvailable);
        Assumptions.assumeTrue(CudaRuntime.isAvailable(), "CUDA runtime not available\n" + details);
        Assumptions.assumeTrue(
                Environment.current().runtimes().hasRuntime("cuda"),
                "CUDA runtime is not registered\n" + details);
        Assumptions.assumeTrue(nvccAvailable, "nvcc not available\n" + details);
    }

    static String diagnosticsSummary() {
        return diagnosticsSummary(isNvccAvailable());
    }

    static String diagnosticsSummary(boolean nvccAvailable) {
        String diagnostics =
                Environment.current().runtimeDiagnostics().stream()
                        .filter(d -> d.deviceType().equals(DeviceType.CUDA))
                        .map(CudaTestAssumptions::formatDiagnostic)
                        .collect(Collectors.joining("\n"));
        if (diagnostics.isBlank()) {
            diagnostics = "<none>";
        }
        return "CUDA diagnostics:\n"
                + diagnostics
                + "\nCudaRuntime.isAvailable="
                + CudaRuntime.isAvailable()
                + "\nCUDA runtime registered="
                + Environment.current().runtimes().hasRuntime("cuda")
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
