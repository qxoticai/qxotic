package com.qxotic.jota.examples;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

final class DemoDevices {

    private static final List<Device> SUPPORTED_BACKENDS =
            List.of(
                    Device.PANAMA,
                    Device.C,
                    Device.HIP,
                    Device.CUDA,
                    Device.OPENCL,
                    Device.METAL,
                    Device.MOJO);

    private DemoDevices() {}

    static boolean hasListDevicesFlag(String[] args) {
        for (String arg : args) {
            if ("--list-devices".equalsIgnoreCase(arg)) {
                return true;
            }
        }
        return false;
    }

    static Device resolveDevice(Environment environment, String requested) {
        if (requested == null || requested.isBlank()) {
            return environment.nativeRuntime().device();
        }
        String normalized = requested.trim().toLowerCase(Locale.ROOT);
        Device parsed =
                switch (normalized) {
                    case "native", "default", "auto" -> Device.NATIVE;
                    case "panama", "ffm", "jvm" -> Device.PANAMA;
                    case "c" -> Device.C;
                    case "hip" -> Device.HIP;
                    case "cuda" -> Device.CUDA;
                    case "opencl", "cl" -> Device.OPENCL;
                    case "metal" -> Device.METAL;
                    case "mojo" -> Device.MOJO;
                    default ->
                            throw new IllegalArgumentException(
                                    "Unknown backend/device '"
                                            + requested
                                            + "'. Use one of: native, panama, c, hip, cuda, opencl,"
                                            + " metal, mojo");
                };
        if (parsed.equals(Device.NATIVE)) {
            return environment.nativeRuntime().device();
        }
        if (!environment.runtimes().hasRuntime(parsed)) {
            throw new IllegalStateException(unavailableDeviceMessage(environment, parsed));
        }
        return parsed;
    }

    static String listDevices(Environment environment) {
        Set<Device> available = new LinkedHashSet<>(environment.runtimes().devices());
        StringBuilder output = new StringBuilder();
        output.append("Available and usable devices:");
        output.append('\n')
                .append("  - native -> ")
                .append(environment.nativeRuntime().device().leafName());

        for (Device backend : SUPPORTED_BACKENDS) {
            if (available.contains(backend)) {
                output.append('\n').append("  - ").append(backend.leafName());
            }
        }

        Set<Device> unavailable = new LinkedHashSet<>();
        List<RuntimeDiagnostic> diagnostics = environment.runtimeDiagnostics();
        for (RuntimeDiagnostic diagnostic : diagnostics) {
            if (!diagnostic.probe().isAvailable()
                    && SUPPORTED_BACKENDS.contains(diagnostic.device())) {
                unavailable.add(diagnostic.device());
            }
        }

        if (!unavailable.isEmpty()) {
            output.append('\n').append('\n').append("Unavailable devices:");
            for (Device device : unavailable) {
                RuntimeDiagnostic diagnostic = latestDiagnosticFor(diagnostics, device);
                if (diagnostic == null) {
                    output.append('\n').append("  - ").append(device.leafName());
                    continue;
                }
                output.append('\n')
                        .append("  - ")
                        .append(device.leafName())
                        .append(": ")
                        .append(diagnostic.probe().message());
                String hint = diagnostic.probe().hint();
                if (hint != null && !hint.isBlank()) {
                    output.append(" (").append(hint).append(')');
                }
            }
        }

        return output.toString();
    }

    private static String unavailableDeviceMessage(Environment environment, Device device) {
        StringBuilder message =
                new StringBuilder("Requested device runtime is unavailable: ")
                        .append(device.name());
        List<RuntimeDiagnostic> diagnostics = environment.runtimes().diagnosticsFor(device);
        if (!diagnostics.isEmpty()) {
            message.append('\n').append("Diagnostics:");
            for (RuntimeDiagnostic diagnostic : diagnostics) {
                message.append('\n')
                        .append("- ")
                        .append(diagnostic.probe().status().name().toLowerCase(Locale.ROOT))
                        .append(": ")
                        .append(diagnostic.probe().message());
                String hint = diagnostic.probe().hint();
                if (hint != null) {
                    message.append(" | hint: ").append(hint);
                }
            }
        }
        message.append('\n').append("Use --list-devices to inspect available runtimes.");
        return message.toString();
    }

    private static RuntimeDiagnostic latestDiagnosticFor(
            List<RuntimeDiagnostic> diagnostics, Device device) {
        for (int i = diagnostics.size() - 1; i >= 0; i--) {
            RuntimeDiagnostic diagnostic = diagnostics.get(i);
            if (diagnostic.device().equals(device)) {
                return diagnostic;
            }
        }
        return null;
    }
}
