package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Assumptions;

final class MetalTestAssumptions {

    private MetalTestAssumptions() {}

    static void assumeMetalReady() {
        boolean xcrunAvailable = isXcrunAvailable();
        boolean metalCompilerAvailable = isMetalCompilerAvailable();
        boolean metallibToolAvailable = isMetallibToolAvailable();
        String details =
                diagnosticsSummary(xcrunAvailable, metalCompilerAvailable, metallibToolAvailable);
        Assumptions.assumeTrue(
                MetalRuntime.isAvailable(), "Metal runtime not available\n" + details);
        Assumptions.assumeTrue(
                Environment.hasRuntimeFor(DeviceType.METAL.deviceIndex(0)),
                "Metal runtime is not registered\n" + details);
        Assumptions.assumeTrue(xcrunAvailable, "xcrun not available\n" + details);
        Assumptions.assumeTrue(
                metalCompilerAvailable, "xcrun metal tool not available\n" + details);
        Assumptions.assumeTrue(
                metallibToolAvailable, "xcrun metallib tool not available\n" + details);
        Assumptions.assumeTrue(
                MetalRuntime.deviceCount() > 0, "No Metal device visible\n" + details);
    }

    static String diagnosticsSummary() {
        return diagnosticsSummary(
                isXcrunAvailable(), isMetalCompilerAvailable(), isMetallibToolAvailable());
    }

    static String diagnosticsSummary(boolean xcrunAvailable) {
        return diagnosticsSummary(
                xcrunAvailable, isMetalCompilerAvailable(), isMetallibToolAvailable());
    }

    static String diagnosticsSummary(
            boolean xcrunAvailable, boolean metalCompilerAvailable, boolean metallibToolAvailable) {
        String diagnostics =
                Environment.runtimeDiagnostics().stream()
                        .filter(d -> d.deviceType().equals(DeviceType.METAL))
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
                + Environment.hasRuntimeFor(DeviceType.METAL.deviceIndex(0))
                + "\nxcrun available="
                + xcrunAvailable
                + "\nmetal tool available="
                + metalCompilerAvailable
                + "\nmetallib tool available="
                + metallibToolAvailable;
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

    private static boolean isMetalCompilerAvailable() {
        return ExternalToolChecks.hasCommand("xcrun", "-sdk", "macosx", "-find", "metal");
    }

    private static boolean isMetallibToolAvailable() {
        return ExternalToolChecks.hasCommand("xcrun", "-sdk", "macosx", "-find", "metallib");
    }
}
