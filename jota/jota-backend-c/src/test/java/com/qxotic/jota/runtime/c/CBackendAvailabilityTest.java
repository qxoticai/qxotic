package com.qxotic.jota.runtime.c;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.testutil.ExternalToolChecks;
import org.junit.jupiter.api.Test;

class CBackendAvailabilityTest {

    @Test
    void cBackendMustBeAvailableInCProfile() {
        String details = diagnosticsSummary();
        assertTrue(
                CNative.isAvailable(),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] C JNI runtime not available",
                        "mvnd -Pc test",
                        details));
        assertTrue(
                hasRuntime(DeviceType.C),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] C runtime is not registered",
                        "mvnd -Pc test",
                        details));
        assertTrue(
                isCompilerAvailable(),
                canaryFailureMessage(
                        "[MISSING_SOFTWARE] C compiler not available", "mvnd -Pc test", details));
        assertEquals(
                DeviceType.C.deviceIndex(0),
                Environment.runtimeFor(DeviceType.C.deviceIndex(0)).device());
    }

    private static String canaryFailureMessage(String reason, String command, String details) {
        return "[BACKEND CANARY] "
                + reason
                + "\nRequested profile command: "
                + command
                + "\nInstall/enable the C toolchain and JNI runtime, or run without -Pc.\n\n"
                + details;
    }

    private static String diagnosticsSummary() {
        return "C diagnostics:\n"
                + "<see runtimeDiagnostics()>"
                + "\nCNative.isAvailable="
                + CNative.isAvailable()
                + "\nC runtime registered="
                + hasRuntime(DeviceType.C)
                + "\ncompiler available="
                + isCompilerAvailable();
    }

    private static boolean hasRuntime(DeviceType type) {
        return Environment.hasRuntimeFor(type.deviceIndex(0));
    }

    private static boolean isCompilerAvailable() {
        return ExternalToolChecks.hasVersionCommand(CKernelCompiler.resolveCompilerExecutable());
    }
}
