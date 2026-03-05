package com.qxotic.jota.runtime.opencl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import org.junit.jupiter.api.Test;

class OpenClBackendAvailabilityTest {

    @Test
    void openclBackendMustBeAvailableInOpenClProfile() {
        String details = OpenClTestAssumptions.diagnosticsSummary();
        assertTrue(
                OpenClRuntime.isAvailable(),
                canaryFailureMessage(
                        "OpenCL JNI runtime not available", "mvnd -Popencl test", details));
        assertTrue(
                Environment.current().runtimes().hasRuntime(Device.OPENCL),
                canaryFailureMessage(
                        "OpenCL runtime is not registered", "mvnd -Popencl test", details));
        assertTrue(
                OpenClRuntime.deviceCount() > 0,
                canaryFailureMessage(
                        "[UNSUPPORTED_HARDWARE] no OpenCL CPU/GPU device visible",
                        "mvnd -Popencl test",
                        details));
        String selectedType = OpenClRuntime.selectedDeviceType();
        assertTrue(
                "GPU".equals(selectedType) || "CPU".equals(selectedType),
                canaryFailureMessage(
                        "Unexpected selected OpenCL device type: " + selectedType,
                        "mvnd -Popencl test",
                        details));
        assertFalse(
                OpenClRuntime.selectedDeviceName().isBlank(),
                canaryFailureMessage(
                        "Selected OpenCL device name is blank", "mvnd -Popencl test", details));
        assertFalse(
                OpenClRuntime.selectedPlatformName().isBlank(),
                canaryFailureMessage(
                        "Selected OpenCL platform name is blank", "mvnd -Popencl test", details));
        assertTrue(
                OpenClRuntime.listDevices().contains("platform["),
                canaryFailureMessage(
                        "OpenCL device listing is empty", "mvnd -Popencl test", details));
        assertEquals(Device.OPENCL, Environment.current().runtimeFor(Device.OPENCL).device());
    }

    private static String canaryFailureMessage(String reason, String command, String details) {
        return "[BACKEND CANARY] "
                + reason
                + "\nRequested profile command: "
                + command
                + "\nInstall/enable OpenCL runtime support on this machine, or run without -Popencl.\n\n"
                + details;
    }
}
