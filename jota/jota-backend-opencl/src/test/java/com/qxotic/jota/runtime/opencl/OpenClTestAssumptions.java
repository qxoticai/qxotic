package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Assumptions;

final class OpenClTestAssumptions {

    private OpenClTestAssumptions() {}

    static void assumeOpenClReady() {
        String details = diagnosticsSummary();
        Assumptions.assumeTrue(
                OpenClRuntime.isAvailable(), "OpenCL runtime not available\n" + details);
        Assumptions.assumeTrue(
                Environment.current().runtimes().hasRuntime(Device.OPENCL),
                "OpenCL runtime is not registered\n" + details);
        Assumptions.assumeTrue(
                OpenClRuntime.deviceCount() > 0, "No OpenCL CPU/GPU device visible\n" + details);
    }

    static String diagnosticsSummary() {
        String diagnostics =
                Environment.current().runtimeDiagnostics().stream()
                        .filter(d -> d.device().equals(Device.OPENCL))
                        .map(OpenClTestAssumptions::formatDiagnostic)
                        .collect(Collectors.joining("\n"));
        if (diagnostics.isBlank()) {
            diagnostics = "<none>";
        }
        return "OpenCL diagnostics:\n"
                + diagnostics
                + "\nOpenCLRuntime.isAvailable="
                + OpenClRuntime.isAvailable()
                + "\nOpenCL runtime registered="
                + Environment.current().runtimes().hasRuntime(Device.OPENCL)
                + "\nOpenCL device count="
                + (OpenClRuntime.isAvailable() ? OpenClRuntime.deviceCount() : 0)
                + "\nOpenCL selected device type="
                + (OpenClRuntime.isAvailable()
                        ? OpenClRuntime.selectedDeviceType()
                        : "<unavailable>")
                + "\nOpenCL selected platform name="
                + (OpenClRuntime.isAvailable()
                        ? OpenClRuntime.selectedPlatformName()
                        : "<unavailable>")
                + "\nOpenCL selected device name="
                + (OpenClRuntime.isAvailable()
                        ? OpenClRuntime.selectedDeviceName()
                        : "<unavailable>")
                + "\nOpenCL selection properties="
                + OpenClRuntime.selectionPropertiesSummary()
                + "\nOpenCL device list=\n"
                + (OpenClRuntime.isAvailable() ? OpenClRuntime.listDevices() : "<unavailable>")
                + "\nOpenCL init failure reason="
                + OpenClRuntime.initFailureReason();
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
}
