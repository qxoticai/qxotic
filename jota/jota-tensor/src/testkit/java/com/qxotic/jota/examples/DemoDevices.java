package com.qxotic.jota.examples;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

final class DemoDevices {

    private static final List<DeviceType> SUPPORTED_BACKENDS =
            List.of(
                    DeviceType.PANAMA,
                    DeviceType.C,
                    DeviceType.HIP,
                    DeviceType.CUDA,
                    DeviceType.OPENCL,
                    DeviceType.METAL,
                    DeviceType.MOJO);

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
            return Environment.nativeRuntime().device();
        }
        String normalized = requested.trim().toLowerCase(Locale.ROOT);
        DeviceType type =
                switch (normalized) {
                    case "native", "default", "auto" -> null;
                    case "panama", "ffm", "jvm" -> DeviceType.PANAMA;
                    case "c" -> DeviceType.C;
                    case "hip" -> DeviceType.HIP;
                    case "cuda" -> DeviceType.CUDA;
                    case "opencl", "cl" -> DeviceType.OPENCL;
                    case "metal" -> DeviceType.METAL;
                    case "mojo" -> DeviceType.MOJO;
                    default ->
                            throw new IllegalArgumentException(
                                    "Unknown backend/device '"
                                            + requested
                                            + "'. Use one of: native, panama, c, hip, cuda, opencl,"
                                            + " metal, mojo");
                };
        if (type == null) {
            return Environment.nativeRuntime().device();
        }
        Device parsed = type.deviceIndex(0);
        if (!environment.runtimes().hasRuntimeFor(parsed)) {
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
                .append(Environment.nativeRuntime().device().runtimeId());

        for (DeviceType backend : SUPPORTED_BACKENDS) {
            if (environment.runtimes().hasRuntimeFor(backend.deviceIndex(0))) {
                output.append('\n').append("  - ").append(backend.id());
            }
        }

        Set<DeviceType> unavailable = new LinkedHashSet<>();
        List<RuntimeDiagnostic> diagnostics = Environment.runtimeDiagnostics();
        for (RuntimeDiagnostic diagnostic : diagnostics) {
            if (!diagnostic.probe().isAvailable()
                    && SUPPORTED_BACKENDS.contains(diagnostic.deviceType())) {
                unavailable.add(diagnostic.deviceType());
            }
        }

        if (!unavailable.isEmpty()) {
            output.append('\n').append('\n').append("Unavailable devices:");
            for (DeviceType deviceType : unavailable) {
                RuntimeDiagnostic diagnostic = latestDiagnosticFor(diagnostics, deviceType);
                if (diagnostic == null) {
                    output.append('\n').append("  - ").append(deviceType.id());
                    continue;
                }
                output.append('\n')
                        .append("  - ")
                        .append(deviceType.id())
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
                new StringBuilder("Requested device runtime is unavailable: ").append(device);
        List<RuntimeDiagnostic> diagnostics =
                environment.runtimes().diagnostics().stream()
                        .filter(d -> d.deviceType().equals(device.type()))
                        .toList();
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
            List<RuntimeDiagnostic> diagnostics, DeviceType deviceType) {
        for (int i = diagnostics.size() - 1; i >= 0; i--) {
            RuntimeDiagnostic diagnostic = diagnostics.get(i);
            if (diagnostic.deviceType().equals(deviceType)) {
                return diagnostic;
            }
        }
        return null;
    }
}
