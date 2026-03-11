package com.qxotic.jota.testutil;

import com.qxotic.jota.Device;

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
            case "native" -> Device.NATIVE;
            case "panama" -> Device.PANAMA;
            case "c" -> Device.C;
            case "hip" -> Device.HIP;
            case "mojo" -> Device.MOJO;
            case "opencl" -> Device.OPENCL;
            case "metal" -> Device.METAL;
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported "
                                    + TEST_DEVICE_PROPERTY
                                    + "='"
                                    + raw
                                    + "'. Supported values: native, panama, c, hip, mojo, opencl,"
                                    + " metal");
        };
    }
}
