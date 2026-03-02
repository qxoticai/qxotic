package com.qxotic.jota.runtime.metal;

import java.util.Locale;

enum MetalMemoryPolicy {
    FAST_DEVICE,
    HOST_VISIBLE,
    AUTO;

    static final String PROPERTY = "jota.metal.memory.policy";

    static MetalMemoryPolicy current() {
        String value = System.getProperty(PROPERTY, "fast").trim().toLowerCase(Locale.ROOT);
        return switch (value) {
            case "fast", "private", "device" -> FAST_DEVICE;
            case "host", "shared", "host_visible" -> HOST_VISIBLE;
            case "auto" -> AUTO;
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported "
                                    + PROPERTY
                                    + " value: "
                                    + value
                                    + " (expected fast|host|auto)");
        };
    }

    int storageMode() {
        return switch (this) {
            case FAST_DEVICE, AUTO -> MetalRuntime.STORAGE_MODE_PRIVATE;
            case HOST_VISIBLE -> MetalRuntime.STORAGE_MODE_SHARED;
        };
    }
}
