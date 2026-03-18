package com.qxotic.jota.testutil;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;

public final class ConfiguredTestDevice {

    public static final String TEST_DEVICE_PROPERTY = "jota.test.device";

    private ConfiguredTestDevice() {}

    public static Device resolve() {
        Device contextual = TestBackendContext.current().orElse(null);
        if (contextual != null) {
            return contextual;
        }
        String raw = System.getProperty(TEST_DEVICE_PROPERTY, "panama");
        String normalized = raw == null ? "panama" : raw.trim().toLowerCase();
        return switch (normalized) {
            case "native" -> new Device(DeviceType.PANAMA, 0);
            case "panama" -> new Device(DeviceType.PANAMA, 0);
            case "c" -> new Device(DeviceType.C, 0);
            case "hip" -> new Device(DeviceType.HIP, 0);
            case "cuda" -> new Device(DeviceType.CUDA, 0);
            case "mojo" -> new Device(DeviceType.MOJO, 0);
            case "opencl" -> new Device(DeviceType.OPENCL, 0);
            case "metal" -> new Device(DeviceType.METAL, 0);
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported "
                                    + TEST_DEVICE_PROPERTY
                                    + "='"
                                    + raw
                                    + "'. Supported values: native, panama, c, hip, cuda, mojo,"
                                    + " opencl, metal");
        };
    }
}
