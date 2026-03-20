package com.qxotic.jota.testutil;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class AvailableBackends {

    public static final String TEST_BACKENDS_MODE_PROPERTY = "jota.test.backends";

    private AvailableBackends() {}

    public static List<Device> resolveTargets() {
        Environment current = Environment.current();
        // Hard baseline requirement for tests.
        current.nativeRuntime();

        Device configured = ConfiguredTestDevice.resolve();
        String modeRaw = System.getProperty(TEST_BACKENDS_MODE_PROPERTY, "single");
        String mode = modeRaw == null ? "single" : modeRaw.trim().toLowerCase();
        if (mode.equals("single")) {
            return List.of(configured);
        }
        if (!mode.equals("available")) {
            throw new IllegalArgumentException(
                    "Unsupported "
                            + TEST_BACKENDS_MODE_PROPERTY
                            + "='"
                            + modeRaw
                            + "'. Supported values: single, available");
        }

        Set<Device> targets = new LinkedHashSet<>();
        targets.add(current.resolveRuntime("native"));
        for (DeviceType type :
                List.of(
                        DeviceType.PANAMA,
                        DeviceType.C,
                        DeviceType.HIP,
                        DeviceType.CUDA,
                        DeviceType.MOJO,
                        DeviceType.OPENCL,
                        DeviceType.METAL)) {
            if (!current.runtimes().hasRuntime(type.id())) {
                continue;
            }
            Device device = current.resolveRuntime(type);
            if (isAvailable(current, device)) {
                targets.add(device);
            }
        }
        if (isAvailable(current, configured)) {
            targets.add(configured);
        }
        return new ArrayList<>(targets);
    }

    private static boolean isAvailable(Environment environment, Device device) {
        if (device.belongsTo(DeviceType.METAL)
                && !(ExternalToolChecks.hasVersionCommand("xcrun")
                        && ExternalToolChecks.hasCommand(
                                "xcrun", "-sdk", "macosx", "-find", "metal")
                        && ExternalToolChecks.hasCommand(
                                "xcrun", "-sdk", "macosx", "-find", "metallib"))) {
            return false;
        }
        if (!environment.runtimes().hasRuntime(device)) {
            return false;
        }
        List<RuntimeDiagnostic> diagnostics =
                environment.runtimes().diagnostics().stream()
                        .filter(d -> d.deviceType().equals(device.type()))
                        .toList();
        if (diagnostics.isEmpty()) {
            return true;
        }
        return diagnostics.stream().anyMatch(d -> d.probe().isAvailable());
    }
}
