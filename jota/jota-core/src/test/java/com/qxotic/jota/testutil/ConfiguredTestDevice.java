package com.qxotic.jota.testutil;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;

public final class ConfiguredTestDevice {

    public static final String TEST_DEVICE_PROPERTY = "jota.test.device";

    private ConfiguredTestDevice() {}

    public static Device resolve() {
        Device contextual = TestBackendContext.current().orElse(null);
        if (contextual != null) {
            return contextual;
        }
        Environment environment = Environment.current();
        String raw = System.getProperty(TEST_DEVICE_PROPERTY, "panama");
        String normalized = raw == null ? "panama" : raw.trim().toLowerCase();
        return switch (normalized) {
            case "native" -> environment.resolveRuntime("native");
            case "default" -> environment.defaultDevice();
            case "panama" -> resolvePreferredOrZero(environment, DeviceType.PANAMA);
            case "c" -> resolvePreferredOrZero(environment, DeviceType.C);
            case "hip" -> resolvePreferredOrZero(environment, DeviceType.HIP);
            case "cuda" -> resolvePreferredOrZero(environment, DeviceType.CUDA);
            case "mojo" -> resolvePreferredOrZero(environment, DeviceType.MOJO);
            case "opencl" -> resolvePreferredOrZero(environment, DeviceType.OPENCL);
            case "metal" -> resolvePreferredOrZero(environment, DeviceType.METAL);
            default -> parseLogicalDevice(environment, raw);
        };
    }

    public static Device resolve(DeviceType type) {
        return resolvePreferredOrZero(Environment.current(), type);
    }

    public static boolean hasRuntime(DeviceType type) {
        return Environment.current().runtimes().hasRuntime(type.id());
    }

    private static Device resolvePreferredOrZero(Environment environment, DeviceType type) {
        if (environment.runtimes().hasRuntime(type.id())) {
            return environment.resolveRuntime(type);
        }
        return type.deviceIndex(0);
    }

    private static Device parseLogicalDevice(Environment environment, String raw) {
        if (raw == null) {
            throw invalidDevice(raw);
        }
        String value = raw.trim();
        int separator = value.indexOf(':');
        if (separator <= 0 || separator == value.length() - 1) {
            throw invalidDevice(raw);
        }
        String runtime = value.substring(0, separator).trim().toLowerCase();
        String indexToken = value.substring(separator + 1).trim();
        int index;
        try {
            index = Integer.parseInt(indexToken);
        } catch (NumberFormatException e) {
            throw invalidDevice(raw);
        }
        if (index < 0) {
            throw invalidDevice(raw);
        }
        DeviceType type =
                switch (runtime) {
                    case "panama" -> DeviceType.PANAMA;
                    case "c" -> DeviceType.C;
                    case "hip" -> DeviceType.HIP;
                    case "cuda" -> DeviceType.CUDA;
                    case "mojo" -> DeviceType.MOJO;
                    case "opencl" -> DeviceType.OPENCL;
                    case "metal" -> DeviceType.METAL;
                    case "java" -> DeviceType.JAVA;
                    default -> null;
                };
        if (type == null) {
            throw invalidDevice(raw);
        }
        Device resolved = resolvePreferredOrZero(environment, type);
        if (resolved.index() == index) {
            return resolved;
        }
        return type.deviceIndex(index);
    }

    private static IllegalArgumentException invalidDevice(String raw) {
        return new IllegalArgumentException(
                "Unsupported "
                        + TEST_DEVICE_PROPERTY
                        + "='"
                        + raw
                        + "'. Supported values: native, default, panama, c, hip, cuda, mojo,"
                        + " opencl, metal, <runtime>:<index>");
    }
}
